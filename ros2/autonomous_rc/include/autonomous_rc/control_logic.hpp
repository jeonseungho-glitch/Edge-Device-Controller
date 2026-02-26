// =========================================================
// control_logic.hpp — 차선·IPM·조향·상태 판단·트래킹
// =========================================================
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace arc {

// ─── 2D 점 ───
struct Pt { int x = 0, y = 0; };

// ─── 세그먼트 ───
struct Segment { int x0, x1; float center; int length; };

// ─── OD 디텍션 ───
struct Detection {
    int   cid;
    float x1, y1, x2, y2;
    float conf;
};

// ─── 칼만 4D ───
class Kalman4D {
public:
    Kalman4D();
    void update(float cx, float cy, float w, float h,
                float& out_cx, float& out_cy, float& out_w, float& out_h);
private:
    cv::KalmanFilter kf_;
    bool initialized_ = false;
};

// ─── 트랙 ───
struct Track {
    int   tid;
    int   cid;
    float box[4];       // x1,y1,x2,y2
    int   missed = 0;
    Kalman4D kf;
};

// ─── Steering State ───
enum class SteerState { S0, L1, L2, R1, R2 };

SteerState steer_to_state(double steer_deg);
std::string steer_state_str(SteerState s);

// ─── Control State ───
enum class CtrlState { STOP, SLOW, DRIVE };

std::string ctrl_state_str(CtrlState s);

// ─── 클래스 ───
class ControlLogic {
public:
    ControlLogic() = default;

    // ── 초기화 (프레임 크기 기반) ──
    void init(int frame_w, int frame_h);

    // ── 클래스 이름 로드 ──
    bool load_class_names(const std::string& path);

    // ── Lane 전처리: BGR → NCHW float (caller 가 버퍼 준비) ──
    void lane_preprocess(const cv::Mat& bgr, float* out_nchw);

    // ── Lane 후처리 → red 포인트 ──
    bool pick_red_points(const float* lane_out,
                         bool is_logit,
                         Pt& redL, Pt& redR);

    // ── 스티어링 각도 계산 ──
    double compute_steer(const Pt& redL, const Pt& redR,
                         double& out_width);

    // ── 스티어링 스무딩 ──
    double smooth_steer(double target_angle, double width, bool red_ok);

    // ── OD letterbox 전처리: BGR → NCHW float ──
    void od_preprocess(const cv::Mat& bgr, float* out_nchw,
                       float& out_ratio, int& out_pad_left, int& out_pad_top);

    // ── OD 디코드 (1×300×6) ──
    std::vector<Detection> decode_od(const float* output, int rows,
                                     float ratio, int pad_left, int pad_top);

    // ── 트래킹 업데이트 ──
    void update_tracks(const std::vector<Detection>& dets);

    // ── ROI 마스크 생성 ──
    void update_roi_mask(const Pt* redL, const Pt* redR);

    // ── 상태 결정 (IPM) ──
    CtrlState decide_state(bool comm_ok);

    // ── Lane 출력이 logit 인지 판별 (최초 1회) ──
    bool check_lane_logit(const float* lane_out, size_t numel);

    // ── Steer state ──
    SteerState steer_state() const { return steer_state_; }
    double     prev_angle()  const { return prev_angle_;  }

    // ── 디버그 정보 ──
    int  track_count() const { return static_cast<int>(tracks_.size()); }
    int  det_count()   const { return last_det_count_; }
    bool has_red_or_yellow() const { return has_red_or_yellow_; }
    std::optional<double> min_car_ped_dist() const { return min_car_ped_dist_m_; }
    std::optional<double> min_sl_cw_dist()   const { return min_sl_cw_dist_m_; }

private:
    // ── 프레임 파라미터 ──
    int W_ = 0, H_ = 0;
    int y_line_ = 0, mid_x_ = 0, xL_ = 0, xR_ = 0;
    int y_purple_ = 0;
    Pt  purpleL_{}, purpleR_{};

    // ── ROI ──
    cv::Mat default_roi_mask_;
    cv::Mat roi_mask_;

    // ── Steer ──
    double prev_angle_  = 0.0;
    double prev_width_  = -1.0;
    SteerState steer_state_ = SteerState::S0;

    // ── Tracking ──
    std::vector<Track> tracks_;
    int  next_tid_      = 0;
    int  last_det_count_ = 0;

    // ── State inputs (매 프레임 갱신) ──
    std::optional<double> min_car_ped_dist_m_;
    std::optional<double> min_sl_cw_dist_m_;
    bool has_yellow_       = false;
    bool has_red_or_yellow_ = false;

    // ── 클래스 이름 ──
    std::vector<std::string> class_names_;

    // ── 유틸 ──
    std::string get_class_name(int cid) const;
    std::string normalize_name(const std::string& s) const;
    bool is_allowed_vehicle_signal(const std::string& cname) const;

    static double ipm_distance(int y_pix, int img_h);
    static float sigmoid_safe(float x);
    static double bbox_overlap_ratio(const cv::Mat& mask,
                                     int x1, int y1, int x2, int y2,
                                     int stride);

    static std::vector<Segment> segments_from_row(const std::vector<int>& xs, int gap);
};

}  // namespace arc
