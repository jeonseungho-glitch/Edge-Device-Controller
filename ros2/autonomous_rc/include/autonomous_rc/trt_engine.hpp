// =========================================================
// trt_engine.hpp — TensorRT 8.x 단일-입력 엔진 래퍼
// =========================================================
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace arc {

// TensorRT 로그 콜백
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// 바인딩 정보
struct Binding {
    int         index     = -1;
    bool        is_input  = false;
    std::string name;
    nvinfer1::Dims     dims{};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    size_t      vol       = 0;     // element count
    size_t      bytes     = 0;
    void*       d_ptr     = nullptr;  // device
    void*       h_ptr     = nullptr;  // host (page-locked)
};

class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine();

    // non-copyable
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    /// 엔진 파일 로드 + context 생성 + 메모리 할당
    bool load(const std::string& engine_path);

    /// float 입력 → 추론 → 첫 번째 출력 버퍼(float*) 반환
    /// @param input  호스트 float 배열 (크기 == input_volume())
    /// @return  호스트 float* (내부 버퍼, 다음 infer 호출까지 유효)
    float* infer(const float* input);

    // 조회
    size_t input_volume()  const { return input_.vol;  }
    size_t output_volume() const { return outputs_.empty() ? 0 : outputs_[0].vol; }
    nvinfer1::Dims input_dims()  const { return input_.dims;  }
    nvinfer1::Dims output_dims() const { return outputs_.empty() ? nvinfer1::Dims{} : outputs_[0].dims; }

    const std::vector<Binding>& outputs() const { return outputs_; }

private:
    TrtLogger                          logger_;
    std::shared_ptr<nvinfer1::IRuntime>           runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine>        engine_;
    std::shared_ptr<nvinfer1::IExecutionContext>   context_;
    cudaStream_t                       stream_ = nullptr;

    Binding              input_;
    std::vector<Binding> outputs_;
    std::vector<void*>   bindings_;      // [num_bindings]

    void cleanup();
};

}  // namespace arc
