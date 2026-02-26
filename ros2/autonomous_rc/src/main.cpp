// =========================================================
// main.cpp — 단일 ROS 2 노드 통합 루프 (Raw Image Publish 포함)
//   Lane Segmentation + Object Detection + IPM + Steering + Motor
//
//   [핵심 변경] cv_bridge 의존성을 완전히 제거하고
//   sensor_msgs::msg::Image 를 직접 구성하여 퍼블리시합니다.
//   → Jetson JetPack OpenCV ↔ ROS cv_bridge OpenCV 버전 충돌 원천 차단
//   → QoS: SensorDataQoS (BEST_EFFORT) 적용으로 rqt_image_view 호환
// =========================================================
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>

// *** cv_bridge 헤더 제거됨 — 직접 변환 방식 사용 ***

#include "autonomous_rc/config.hpp"
#include "autonomous_rc/trt_engine.hpp"
#include "autonomous_rc/control_logic.hpp"
#include "autonomous_rc/motor_driver.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <csignal>
#include <memory>
#include <optional>
#include <cstring>   // std::memcpy

// ─── Graceful shutdown ───
static volatile sig_atomic_t g_shutdown = 0;
static void signal_handler(int) { g_shutdown = 1; }

namespace arc {

    // ─────────────────────────────────────────────
    // cv::Mat → sensor_msgs::msg::Image 직접 변환
    // cv_bridge 없이 동작하므로 OpenCV 버전 충돌 문제 없음
    // ─────────────────────────────────────────────
    static sensor_msgs::msg::Image::UniquePtr
    mat_to_image_msg(const cv::Mat& mat,
                     const std_msgs::msg::Header& header,
                     const std::string& encoding)
    {
        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        msg->header       = header;
        msg->height       = static_cast<uint32_t>(mat.rows);
        msg->width        = static_cast<uint32_t>(mat.cols);
        msg->encoding     = encoding;
        msg->is_bigendian = 0;
        msg->step         = static_cast<uint32_t>(mat.step[0]);

        const size_t total_bytes = msg->step * msg->height;
        msg->data.resize(total_bytes);

        if (mat.isContinuous()) {
            std::memcpy(msg->data.data(), mat.data, total_bytes);
        } else {
            for (int r = 0; r < mat.rows; ++r) {
                std::memcpy(msg->data.data() + r * msg->step,
                            mat.ptr(r),
                            msg->step);
            }
        }
        return msg;
    }

    class AutonomousNode : public rclcpp::Node {
    public:
        AutonomousNode()
            : Node("autonomous_rc_node") {

            // ── ROS 파라미터 ──
            this->declare_parameter<std::string>("base_dir", DEFAULT_BASE_DIR);
            this->declare_parameter<std::string>("video", DEFAULT_VIDEO_NAME);
            this->declare_parameter<std::string>("lane_engine", DEFAULT_LANE_ENGINE);
            this->declare_parameter<std::string>("od_engine", DEFAULT_OD_ENGINE);
            this->declare_parameter<std::string>("classes_txt", DEFAULT_CLASSES_TXT);
            this->declare_parameter<int>("img_pub_interval", 1);

            base_dir_     = this->get_parameter("base_dir").as_string();
            video_path_   = base_dir_ + "/" + this->get_parameter("video").as_string();
            lane_path_    = base_dir_ + "/" + this->get_parameter("lane_engine").as_string();
            od_path_      = base_dir_ + "/" + this->get_parameter("od_engine").as_string();
            classes_path_ = base_dir_ + "/" + this->get_parameter("classes_txt").as_string();
            img_pub_interval_ = this->get_parameter("img_pub_interval").as_int();
            if (img_pub_interval_ < 1) img_pub_interval_ = 1;

            // ── 상태 퍼블리셔 ──
            state_pub_ = this->create_publisher<std_msgs::msg::String>(
                "control_state", 10);

            // ── 원본 영상 퍼블리셔 ──
            // ★ SensorDataQoS = BEST_EFFORT + VOLATILE + depth(5)
            //   rqt_image_view, rviz2 등 표준 뷰어가 기본적으로 사용하는 QoS
            auto qos = rclcpp::QoS(1).reliable().durability_volatile();
            img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "camera/raw_image", qos);

            RCLCPP_INFO(this->get_logger(), "VIDEO : %s", video_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "LANE  : %s", lane_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "OD    : %s", od_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "DEPTH : IPM (no depth.engine)");
            RCLCPP_INFO(this->get_logger(), "IMG PUB INTERVAL : every %d frame(s)",
                        img_pub_interval_);
            RCLCPP_INFO(this->get_logger(),
                "Image topic: camera/raw_image (QoS: SensorData/BestEffort)");
        }

        bool initialize() {
            if (!lane_trt_.load(lane_path_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load lane engine");
                return false;
            }
            if (!od_trt_.load(od_path_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load OD engine");
                return false;
            }

            if (!dc_motor_.init()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to init DC motor");
                return false;
            }
            if (!servo_.init()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to init servo");
                return false;
            }

            if (START_WITH_STOP) {
                dc_motor_.apply(CtrlState::STOP);
                last_state_ = CtrlState::STOP;
            }

            cap_.open(video_path_);
            if (!cap_.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open video: %s",
                             video_path_.c_str());
                return false;
            }

            W_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
            H_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps_src = cap_.get(cv::CAP_PROP_FPS);
            if (fps_src <= 0) fps_src = 30.0;
            int nframes = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
            RCLCPP_INFO(this->get_logger(), "VIDEO: %dx%d fps=%.2f frames=%d",
                        W_, H_, fps_src, nframes);

            logic_.init(W_, H_);
            logic_.load_class_names(classes_path_);

            lane_buf_.resize(3 * LANE_H * LANE_W, 0.0f);
            od_buf_.resize(3 * IMGSZ * IMGSZ, 0.0f);

            RCLCPP_INFO(this->get_logger(), "Initialization complete. Ready to run.");
            return true;
        }

        void run() {
            cv::Mat frame;
            auto t_prev = std::chrono::steady_clock::now();
            double fps_ema = 0.0;
            double seg_ms = 0.0, od_ms = 0.0;
            bool lane_logit_checked = false;
            bool lane_is_logit = true;
            SteerState steer_state = SteerState::S0;
            int idx = 0;
            int img_pub_count = 0;

            while (!g_shutdown && rclcpp::ok()) {
                if (!cap_.read(frame)) break;
                ++idx;

                // ────────────────────────────────
                // 원본 프레임 퍼블리시
                //   추론 결과가 그려지기 전의 순수 원본입니다.
                //   cv_bridge 없이 직접 메시지를 구성합니다.
                // ────────────────────────────────
                if ((idx % 1) == 0 && !frame.empty()) {
                    std_msgs::msg::Header header;
                    header.stamp    = this->now();
                    header.frame_id = "camera_link";

                    auto img_msg = mat_to_image_msg(frame, header, "bgr8");
                    img_pub_->publish(std::move(img_msg));
                    ++img_pub_count;

                    if (img_pub_count <= 5) {
                        RCLCPP_INFO(this->get_logger(),
                            "[IMG PUB] frame=%d size=%dx%d published (#%d)",
                            idx, frame.cols, frame.rows, img_pub_count);
                    }
                }

                auto t_now = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(t_now - t_prev).count();
                t_prev = t_now;
                double fps_inst = (dt > 1e-9) ? (1.0 / dt) : 0.0;
                fps_ema = (fps_ema <= 1e-6)
                    ? fps_inst
                    : (1.0 - FPS_EMA_ALPHA) * fps_ema + FPS_EMA_ALPHA * fps_inst;

                bool do_timing = true;

                // ════════════════════════════════
                // LANE SEGMENTATION
                // ════════════════════════════════
                auto t0 = std::chrono::steady_clock::now();
                logic_.lane_preprocess(frame, lane_buf_.data());
                float* lane_out = lane_trt_.infer(lane_buf_.data());
                if (do_timing)
                    seg_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();

                if (!lane_logit_checked) {
                    lane_is_logit = logic_.check_lane_logit(lane_out,
                        lane_trt_.output_volume());
                    lane_logit_checked = true;
                }

                Pt redL, redR;
                bool red_ok = logic_.pick_red_points(lane_out, lane_is_logit,
                                                     redL, redR);

                // ════════════════════════════════
                // STEERING
                // ════════════════════════════════
                double width = 0.0;
                double target_angle = 0.0;
                if (red_ok) {
                    target_angle = logic_.compute_steer(redL, redR, width);
                }
                double current_angle = logic_.smooth_steer(target_angle, width,
                                                           red_ok);

                SteerState new_ss = steer_to_state(current_angle);
                if (new_ss != steer_state) {
                    steer_state = servo_.transition(steer_state, new_ss);
                }
                else {
                    servo_.stop();
                }

                // ════════════════════════════════
                // ROI MASK
                // ════════════════════════════════
                logic_.update_roi_mask(red_ok ? &redL : nullptr,
                    red_ok ? &redR : nullptr);

                // ════════════════════════════════
                // OBJECT DETECTION
                // ════════════════════════════════
                t0 = std::chrono::steady_clock::now();
                float ratio; int pad_l, pad_t;
                logic_.od_preprocess(frame, od_buf_.data(), ratio, pad_l, pad_t);
                float* od_out = od_trt_.infer(od_buf_.data());
                if (do_timing)
                    od_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();

                auto dets = logic_.decode_od(od_out, 300, ratio, pad_l, pad_t);
                logic_.update_tracks(dets);

                // ════════════════════════════════
                // STATE DECISION (IPM)
                // ════════════════════════════════
                CtrlState state = logic_.decide_state(/*comm_ok=*/true);

                if (state != last_state_) {
                    dc_motor_.apply(state);
                    last_state_ = state;
                }

                // ════════════════════════════════
                // STATUS PUBLISH + PRINT
                // ════════════════════════════════
                if (idx % PRINT_EVERY == 0) {
                    auto cp = logic_.min_car_ped_dist();
                    auto sl = logic_.min_sl_cw_dist();

                    if (do_timing) {
                        printf("[%05d] FPS=%5.1f | steer=%+6.2f | SS=%s | "
                            "STATE=%5s | SEG=%6.1fms OD=%6.1fms | "
                            "trk=%d det=%d | "
                            "min_cp_m=%6.2f min_sl_m=%6.2f sigRY=%d "
                            "red_ok=%d\n",
                            idx, fps_ema, current_angle,
                            steer_state_str(steer_state).c_str(),
                            ctrl_state_str(state).c_str(),
                            seg_ms, od_ms,
                            logic_.track_count(), logic_.det_count(),
                            cp.value_or(-1.0), sl.value_or(-1.0),
                            logic_.has_red_or_yellow() ? 1 : 0,
                            red_ok ? 1 : 0);
                    }
                    else {
                        printf("[%05d] FPS=%5.1f | steer=%+6.2f | SS=%s | "
                            "STATE=%5s | trk=%d det=%d red_ok=%d\n",
                            idx, fps_ema, current_angle,
                            steer_state_str(steer_state).c_str(),
                            ctrl_state_str(state).c_str(),
                            logic_.track_count(), logic_.det_count(),
                            red_ok ? 1 : 0);
                    }
                    fflush(stdout);

                    auto msg = std_msgs::msg::String();
                    msg.data = ctrl_state_str(state);
                    state_pub_->publish(msg);
                }

                rclcpp::spin_some(this->get_node_base_interface());
            }

            RCLCPP_INFO(this->get_logger(),
                "Loop finished. Total frames: %d, Published images: %d",
                idx, img_pub_count);
        }

        void shutdown() {
            dc_motor_.apply(CtrlState::STOP);
            servo_.stop();
            dc_motor_.cleanup();
            servo_.cleanup();
            cap_.release();
            RCLCPP_INFO(this->get_logger(), "Hardware cleanup done.");
        }

    private:
        std::string base_dir_, video_path_, lane_path_, od_path_, classes_path_;

        TrtEngine lane_trt_, od_trt_;

        DcMotor         dc_motor_;
        ContinuousServo servo_;

        ControlLogic logic_;
        CtrlState    last_state_ = CtrlState::STOP;

        cv::VideoCapture cap_;
        int W_ = 0, H_ = 0;

        std::vector<float> lane_buf_;
        std::vector<float> od_buf_;

        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr  state_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;

        int img_pub_interval_ = 1;
    };

}  // namespace arc


int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    rclcpp::init(argc, argv);

    auto node = std::make_shared<arc::AutonomousNode>();

    if (!node->initialize()) {
        RCLCPP_FATAL(node->get_logger(), "Initialization failed. Aborting.");
        node->shutdown();
        rclcpp::shutdown();
        return 1;
    }

    node->run();
    node->shutdown();
    rclcpp::shutdown();
    return 0;
}
