// =========================================================
// trt_engine.cpp — TensorRT 8.x 단일-입력 엔진 래퍼 구현
// =========================================================
#include "autonomous_rc/trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <cstring>
#include <cassert>

namespace arc {

// ─── Logger ───
void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << "\n";
    }
}

// ─── Helper: dtype → byte size ───
static size_t dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        assert(d.d[i] > 0);
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

// ─── TrtEngine ───
TrtEngine::~TrtEngine() { cleanup(); }

void TrtEngine::cleanup() {
    // device 메모리 해제
    if (input_.d_ptr)  { cudaFree(input_.d_ptr);  input_.d_ptr = nullptr; }
    if (input_.h_ptr)  { cudaFreeHost(input_.h_ptr); input_.h_ptr = nullptr; }
    for (auto& o : outputs_) {
        if (o.d_ptr) { cudaFree(o.d_ptr);      o.d_ptr = nullptr; }
        if (o.h_ptr) { cudaFreeHost(o.h_ptr);   o.h_ptr = nullptr; }
    }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

bool TrtEngine::load(const std::string& engine_path) {
    // 파일 읽기
    std::ifstream ifs(engine_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        std::cerr << "[TRT] Cannot open engine file: " << engine_path << "\n";
        return false;
    }
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    ifs.read(blob.data(), size);
    ifs.close();

    // Runtime + Engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) { std::cerr << "[TRT] createInferRuntime failed\n"; return false; }

    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) { std::cerr << "[TRT] deserializeCudaEngine failed\n"; return false; }

    context_.reset(engine_->createExecutionContext());
    if (!context_) { std::cerr << "[TRT] createExecutionContext failed\n"; return false; }

    cudaStreamCreate(&stream_);

    // 바인딩 할당
    int nb = engine_->getNbBindings();
    bindings_.resize(nb, nullptr);
    bool found_input = false;

    for (int i = 0; i < nb; ++i) {
        Binding b;
        b.index    = i;
        b.is_input = engine_->bindingIsInput(i);
        b.name     = engine_->getBindingName(i);
        b.dims     = engine_->getBindingDimensions(i);
        b.dtype    = engine_->getBindingDataType(i);

        // 동적 shape 검사
        for (int d = 0; d < b.dims.nbDims; ++d) {
            if (b.dims.d[d] <= 0) {
                std::cerr << "[TRT] Dynamic shape at binding " << i
                          << " dim " << d << " → rebuild fixed engine\n";
                return false;
            }
        }

        b.vol   = volume(b.dims);
        b.bytes = b.vol * dtype_size(b.dtype);

        // device 메모리
        cudaMalloc(&b.d_ptr, b.bytes);
        // host pinned 메모리
        cudaMallocHost(&b.h_ptr, b.bytes);

        bindings_[i] = b.d_ptr;

        if (b.is_input) {
            if (found_input) {
                std::cerr << "[TRT] Multi-input engine not supported\n";
                return false;
            }
            found_input = true;
            input_ = b;
        } else {
            outputs_.push_back(b);
        }
    }

    if (!found_input || outputs_.empty()) {
        std::cerr << "[TRT] Need 1 input and >=1 output binding\n";
        return false;
    }

    std::cout << "[TRT] Loaded: " << engine_path << "\n";
    std::cout << "      Input  [" << input_.name << "] vol=" << input_.vol << "\n";
    for (auto& o : outputs_)
        std::cout << "      Output [" << o.name << "] vol=" << o.vol << "\n";

    return true;
}

float* TrtEngine::infer(const float* input) {
    // Host → Device
    size_t in_bytes = input_.vol * sizeof(float);

    // 엔진이 FP16 이어도 FP32 입력 → FP16 변환은 TRT 내부에서 처리됨
    // 단, 입력 바인딩 dtype이 kHALF면 직접 변환 필요.
    // 여기서는 두 엔진 모두 FP32 입력으로 가정 (일반적 패턴).
    if (input_.dtype == nvinfer1::DataType::kHALF) {
        // FP32→FP16 변환을 host 에서 수행 (간단 구현)
        auto* dst = reinterpret_cast<uint16_t*>(input_.h_ptr);
        for (size_t i = 0; i < input_.vol; ++i) {
            // __float2half_rn 대신 간단 변환
            float v = input[i];
            // 간이 fp16 변환 (precision 충분)
            uint32_t f32;
            std::memcpy(&f32, &v, 4);
            uint16_t sign = (f32 >> 16) & 0x8000;
            int32_t  exp  = ((f32 >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = f32 & 0x7FFFFF;
            if (exp <= 0) { dst[i] = sign; }
            else if (exp >= 31) { dst[i] = static_cast<uint16_t>(sign | 0x7C00); }
            else { dst[i] = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13)); }
        }
        cudaMemcpyAsync(input_.d_ptr, input_.h_ptr, input_.bytes,
                         cudaMemcpyHostToDevice, stream_);
    } else {
        cudaMemcpyAsync(input_.d_ptr, input, in_bytes,
                         cudaMemcpyHostToDevice, stream_);
    }

    // Execute
    context_->enqueueV2(bindings_.data(), stream_, nullptr);

    // Device → Host (첫 번째 출력만)
    auto& out0 = outputs_[0];
    cudaMemcpyAsync(out0.h_ptr, out0.d_ptr, out0.bytes,
                     cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // FP16 출력 → FP32 변환
    if (out0.dtype == nvinfer1::DataType::kHALF) {
        // 별도 버퍼가 필요 → 정적 버퍼 사용 (thread-unsafe, 단일 노드 사용 전제)
        static std::vector<float> fp32_buf;
        fp32_buf.resize(out0.vol);
        auto* src16 = reinterpret_cast<const uint16_t*>(out0.h_ptr);
        for (size_t i = 0; i < out0.vol; ++i) {
            uint16_t h = src16[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t expo = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (expo == 0) { f32 = sign; }
            else if (expo == 31) { f32 = sign | 0x7F800000 | (mant << 13); }
            else { f32 = sign | ((expo - 15 + 127) << 23) | (mant << 13); }
            std::memcpy(&fp32_buf[i], &f32, 4);
        }
        return fp32_buf.data();
    }

    return reinterpret_cast<float*>(out0.h_ptr);
}

}  // namespace arc
