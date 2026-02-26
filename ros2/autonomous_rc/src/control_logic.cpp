// =========================================================
// control_logic.cpp — 차선·IPM·조향·상태 판단·트래킹 구현
// Python engine_control_ipm.py 로직 1:1 이식
// =========================================================
#include "autonomous_rc/control_logic.hpp"
#include "autonomous_rc/config.hpp"

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cctype>

namespace arc {

// =========================================================
// SteerState helpers
// =========================================================
SteerState steer_to_state(double deg) {
    double s = deg;
    if (s >= -STEER_DEAD && s <= STEER_DEAD) return SteerState::S0;
    bool right = s > 0;
    double mag = std::abs(s);
    if (right) return (mag <= STEER_L1) ? SteerState::R1 : SteerState::R2;
    else       return (mag <= STEER_L1) ? SteerState::L1 : SteerState::L2;
}

std::string steer_state_str(SteerState s) {
    switch(s) {
        case SteerState::S0: return "S0";
        case SteerState::L1: return "L1";
        case SteerState::L2: return "L2";
        case SteerState::R1: return "R1";
        case SteerState::R2: return "R2";
    }
    return "??";
}

std::string ctrl_state_str(CtrlState s) {
    switch(s) {
        case CtrlState::STOP:  return "STOP";
        case CtrlState::SLOW:  return "SLOW";
        case CtrlState::DRIVE: return "DRIVE";
    }
    return "??";
}

// =========================================================
// Kalman4D
// =========================================================
Kalman4D::Kalman4D() {
    kf_ = cv::KalmanFilter(4, 4, 0, CV_32F);
    cv::setIdentity(kf_.transitionMatrix);
    cv::setIdentity(kf_.measurementMatrix);
    cv::setIdentity(kf_.processNoiseCov,   cv::Scalar(0.03));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(0.6));
}

void Kalman4D::update(float cx, float cy, float w, float h,
                      float& out_cx, float& out_cy, float& out_w, float& out_h) {
    cv::Mat z = (cv::Mat_<float>(4,1) << cx, cy, w, h);
    if (!initialized_) {
        z.copyTo(kf_.statePre);
        z.copyTo(kf_.statePost);
        initialized_ = true;
    }
    kf_.correct(z);
    cv::Mat pred = kf_.predict();
    out_cx = pred.at<float>(0);
    out_cy = pred.at<float>(1);
    out_w  = pred.at<float>(2);
    out_h  = pred.at<float>(3);
}

// =========================================================
// Init
// =========================================================
void ControlLogic::init(int frame_w, int frame_h) {
    W_ = frame_w;
    H_ = frame_h;

    y_line_ = static_cast<int>(H_ * Y_LINE_RATIO);
    mid_x_  = W_ / 2;
    xL_ = std::max(0, mid_x_ - SEARCH_HALF_WIDTH);
    xR_ = std::min(W_ - 1, mid_x_ + SEARCH_HALF_WIDTH);

    y_purple_ = static_cast<int>(H_ * PURPLE_Y_RATIO);
    int pxL = std::max(0, mid_x_ - DEFAULT_LANE_HALF_WIDTH);
    int pxR = std::min(W_ - 1, mid_x_ + DEFAULT_LANE_HALF_WIDTH);
    purpleL_ = {pxL, y_purple_};
    purpleR_ = {pxR, y_purple_};

    // default ROI polygon
    int y_top = static_cast<int>(H_ * 0.70);
    int y_bot = static_cast<int>(H_ * PURPLE_Y_RATIO);
    int tl = static_cast<int>(W_ * 0.45), tr = static_cast<int>(W_ * 0.55);
    int bl = static_cast<int>(W_ * 0.32), br = static_cast<int>(W_ * 0.68);
    std::vector<cv::Point> poly = {{tl, y_top}, {tr, y_top}, {br, y_bot}, {bl, y_bot}};
    default_roi_mask_ = cv::Mat::zeros(H_, W_, CV_8U);
    cv::fillPoly(default_roi_mask_, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(1));

    roi_mask_ = default_roi_mask_.clone();

    prev_angle_ = 0.0;
    prev_width_ = -1.0;
    steer_state_ = SteerState::S0;
    tracks_.clear();
    next_tid_ = 0;

    std::cout << "[CTRL] init W=" << W_ << " H=" << H_
              << " y_line=" << y_line_ << " mid_x=" << mid_x_ << "\n";
}

// =========================================================
// Class names
// =========================================================
bool ControlLogic::load_class_names(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    class_names_.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        // trim
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (!line.empty()) class_names_.push_back(line);
    }
    std::cout << "[CTRL] Loaded " << class_names_.size() << " class names\n";
    return !class_names_.empty();
}

std::string ControlLogic::get_class_name(int cid) const {
    if (cid >= 0 && cid < static_cast<int>(class_names_.size()))
        return class_names_[cid];
    return std::to_string(cid);
}

std::string ControlLogic::normalize_name(const std::string& s) const {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '-' || c == ' ') out += '_';
        else out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool ControlLogic::is_allowed_vehicle_signal(const std::string& cname) const {
    return VEH_SIGNAL_ALLOW_N.count(normalize_name(cname)) > 0;
}

// =========================================================
// Sigmoid
// =========================================================
float ControlLogic::sigmoid_safe(float x) {
    x = std::max(-50.0f, std::min(50.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

// =========================================================
// Lane preprocess
// =========================================================
void ControlLogic::lane_preprocess(const cv::Mat& bgr, float* out_nchw) {
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(LANE_W, LANE_H));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // HWC → NCHW (float, ImageNet norm)
    int hw = LANE_H * LANE_W;
    for (int y = 0; y < LANE_H; ++y) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < LANE_W; ++x) {
            int idx = y * LANE_W + x;
            float r = row[x * 3 + 0] / 255.0f;
            float g = row[x * 3 + 1] / 255.0f;
            float b = row[x * 3 + 2] / 255.0f;
            out_nchw[0 * hw + idx] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            out_nchw[1 * hw + idx] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            out_nchw[2 * hw + idx] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
    }
}

// =========================================================
// Segments from row
// =========================================================
std::vector<Segment> ControlLogic::segments_from_row(const std::vector<int>& xs, int gap) {
    std::vector<Segment> segs;
    if (xs.empty()) return segs;

    std::vector<int> sorted_xs = xs;
    std::sort(sorted_xs.begin(), sorted_xs.end());

    int start = 0;
    for (int i = 1; i < static_cast<int>(sorted_xs.size()); ++i) {
        if (sorted_xs[i] - sorted_xs[i-1] > gap) {
            int x0 = sorted_xs[start], x1 = sorted_xs[i-1];
            int len = i - start;
            float center = 0.5f * (x0 + x1);
            segs.push_back({x0, x1, center, len});
            start = i;
        }
    }
    int x0 = sorted_xs[start], x1 = sorted_xs.back();
    int len = static_cast<int>(sorted_xs.size()) - start;
    float center = 0.5f * (x0 + x1);
    segs.push_back({x0, x1, center, len});

    return segs;
}

// =========================================================
// Check lane logit
// =========================================================
bool ControlLogic::check_lane_logit(const float* data, size_t numel) {
    float mn = data[0], mx = data[0];
    for (size_t i = 1; i < numel; ++i) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    bool is_logit = !(mn >= -0.05f && mx <= 1.05f);
    std::cout << "[CTRL] Lane output min=" << mn << " max=" << mx
              << " is_logit=" << is_logit << "\n";
    return is_logit;
}

// =========================================================
// Pick red points
// =========================================================
bool ControlLogic::pick_red_points(const float* lane_out,
                                    bool is_logit,
                                    Pt& redL, Pt& redR) {
    // lane_out: (1,1,LANE_H,LANE_W)
    float sx = static_cast<float>(LANE_W) / W_;
    float sy = static_cast<float>(LANE_H) / H_;

    int y_lane = std::max(0, std::min(LANE_H - 1, static_cast<int>(std::round(y_line_ * sy))));
    int xL_lane = std::max(0, std::min(LANE_W - 1, static_cast<int>(std::round(xL_ * sx))));
    int xR_lane = std::max(xL_lane + 1, std::min(LANE_W, static_cast<int>(std::round(xR_ * sx))));

    // row pointer: lane_out[0, 0, y_lane, xL_lane .. xR_lane)
    const float* row = lane_out + y_lane * LANE_W;

    std::vector<int> xs_lane;
    for (int x = xL_lane; x < xR_lane; ++x) {
        float v = row[x];
        float p = is_logit ? sigmoid_safe(v) : v;
        if (p >= LANE_THR) xs_lane.push_back(x);
    }
    if (xs_lane.empty()) return false;

    int seg_gap_lane = std::max(1, static_cast<int>(std::round(SEG_GAP_PX * sx)));
    auto segs = segments_from_row(xs_lane, seg_gap_lane);

    int min_seg_lane = std::max(1, static_cast<int>(std::round(MIN_SEG_LEN_PX * sx)));
    segs.erase(std::remove_if(segs.begin(), segs.end(),
        [min_seg_lane](const Segment& s){ return s.length < min_seg_lane; }), segs.end());
    if (segs.empty()) return false;

    float mid_lane = mid_x_ * sx;
    std::vector<Segment> left_segs, right_segs;
    for (auto& s : segs) {
        if (s.center < mid_lane) left_segs.push_back(s);
        else right_segs.push_back(s);
    }
    if (left_segs.empty() || right_segs.empty()) return false;

    // closest to midline
    auto bestL = *std::min_element(left_segs.begin(), left_segs.end(),
        [mid_lane](const Segment& a, const Segment& b){
            return std::abs(mid_lane - a.center) < std::abs(mid_lane - b.center);
        });
    auto bestR = *std::min_element(right_segs.begin(), right_segs.end(),
        [mid_lane](const Segment& a, const Segment& b){
            return std::abs(a.center - mid_lane) < std::abs(b.center - mid_lane);
        });

    float lx_lane = bestL.center;
    float rx_lane = bestR.center;
    if (rx_lane <= lx_lane) return false;

    int lx = std::max(0, std::min(W_ - 1, static_cast<int>(std::round(lx_lane / sx))));
    int rx = std::max(0, std::min(W_ - 1, static_cast<int>(std::round(rx_lane / sx))));
    int y  = std::max(0, std::min(H_ - 1, y_line_));

    int width = rx - lx;
    if (width < MIN_WIDTH_PX || width > MAX_WIDTH_PX) return false;

    redL = {lx, y};
    redR = {rx, y};
    return true;
}

// =========================================================
// Compute steer from midline
// =========================================================
double ControlLogic::compute_steer(const Pt& redL, const Pt& redR, double& out_width) {
    double c1x = 0.5 * (redL.x + redR.x);
    double c1y = 0.5 * (redL.y + redR.y);
    double c0x = 0.5 * (purpleL_.x + purpleR_.x);
    double c0y = 0.5 * (purpleL_.y + purpleR_.y);

    double dy = c0y - c1y;
    if (std::abs(dy) < 1e-6) { out_width = 0; return 0.0; }
    double dx = c1x - c0x;
    double slope = dx / dy;
    double steer = std::max(-STEER_CLIP_DEG, std::min(STEER_CLIP_DEG, slope * STEER_GAIN_DEG));
    out_width = static_cast<double>(redR.x - redL.x);
    return steer;
}

// =========================================================
// Smooth steer
// =========================================================
double ControlLogic::smooth_steer(double target_angle, double width, bool red_ok) {
    double current;
    if (red_ok) {
        // dynamic smoothing
        double alpha;
        if (width < 0 || prev_width_ < 1e-6) {
            alpha = SMOOTH_MIN;
        } else {
            double rel = std::abs(width - prev_width_) / std::max(prev_width_, 1.0);
            double t = std::min(1.0, rel / 0.5);
            alpha = (1.0 - t) * SMOOTH_MAX + t * SMOOTH_MIN;
        }
        if (width > 0) prev_width_ = width;

        // clamp step
        double d = target_angle - prev_angle_;
        if (d > MAX_STEER_STEP_DEG) target_angle = prev_angle_ + MAX_STEER_STEP_DEG;
        else if (d < -MAX_STEER_STEP_DEG) target_angle = prev_angle_ - MAX_STEER_STEP_DEG;

        current = prev_angle_ * (1.0 - alpha) + target_angle * alpha;
    } else {
        current = prev_angle_ * (1.0 - SMOOTH_MIN);
    }
    prev_angle_ = current;
    return current;
}

// =========================================================
// OD preprocess (letterbox)
// =========================================================
void ControlLogic::od_preprocess(const cv::Mat& bgr, float* out_nchw,
                                  float& out_ratio, int& out_pad_left, int& out_pad_top) {
    int h0 = bgr.rows, w0 = bgr.cols;
    float r = std::min(static_cast<float>(IMGSZ) / h0,
                       static_cast<float>(IMGSZ) / w0);
    int new_w = static_cast<int>(std::round(w0 * r));
    int new_h = static_cast<int>(std::round(h0 * r));
    float dw = (IMGSZ - new_w) * 0.5f;
    float dh = (IMGSZ - new_h) * 0.5f;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int top    = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left   = static_cast<int>(std::round(dw - 0.1f));
    int right  = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // HWC → NCHW float32
    int hw = IMGSZ * IMGSZ;
    for (int y = 0; y < IMGSZ; ++y) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < IMGSZ; ++x) {
            int idx = y * IMGSZ + x;
            out_nchw[0 * hw + idx] = row[x * 3 + 0] / 255.0f;
            out_nchw[1 * hw + idx] = row[x * 3 + 1] / 255.0f;
            out_nchw[2 * hw + idx] = row[x * 3 + 2] / 255.0f;
        }
    }

    out_ratio    = r;
    out_pad_left = left;
    out_pad_top  = top;
}

// =========================================================
// OD decode (1×300×6)
// =========================================================
std::vector<Detection> ControlLogic::decode_od(const float* output, int rows,
                                                float ratio, int pad_left, int pad_top) {
    std::vector<Detection> dets;
    for (int i = 0; i < rows; ++i) {
        const float* row = output + i * 6;
        float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        float score = row[4];
        int   cls   = static_cast<int>(row[5]);
        if (score < CONF_THRESH) continue;

        x1 = (x1 - pad_left) / ratio;
        x2 = (x2 - pad_left) / ratio;
        y1 = (y1 - pad_top)  / ratio;
        y2 = (y2 - pad_top)  / ratio;
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(W_ - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(W_ - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(H_ - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(H_ - 1)));
        if (x2 <= x1 + 1 || y2 <= y1 + 1) continue;

        dets.push_back({cls, x1, y1, x2, y2, score});
    }
    return dets;
}

// =========================================================
// IOU
// =========================================================
static float iou_xyxy(const float* a, const float* b) {
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float iw  = std::max(0.0f, ix2 - ix1);
    float ih  = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = std::max(0.0f, a[2]-a[0]) * std::max(0.0f, a[3]-a[1]);
    float area_b = std::max(0.0f, b[2]-b[0]) * std::max(0.0f, b[3]-b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

// =========================================================
// Tracking
// =========================================================
void ControlLogic::update_tracks(const std::vector<Detection>& dets) {
    last_det_count_ = static_cast<int>(dets.size());
    std::vector<bool> used(dets.size(), false);

    for (auto& tr : tracks_) {
        float best_iou = 0.0f;
        int   best_j   = -1;
        for (int j = 0; j < static_cast<int>(dets.size()); ++j) {
            if (used[j]) continue;
            if (dets[j].cid != tr.cid) continue;
            float det_box[4] = {dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2};
            float v = iou_xyxy(tr.box, det_box);
            if (v > best_iou) { best_iou = v; best_j = j; }
        }
        if (best_j >= 0 && best_iou >= TRACK_IOU_TH) {
            tr.box[0] = dets[best_j].x1; tr.box[1] = dets[best_j].y1;
            tr.box[2] = dets[best_j].x2; tr.box[3] = dets[best_j].y2;
            tr.missed = 0;
            used[best_j] = true;
        } else {
            tr.missed++;
        }
    }

    // remove dead
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [](const Track& t){ return t.missed > TRACK_MAX_MISS; }), tracks_.end());

    // spawn new
    for (int j = 0; j < static_cast<int>(dets.size()); ++j) {
        if (used[j]) continue;
        Track t;
        t.tid = next_tid_++;
        t.cid = dets[j].cid;
        t.box[0] = dets[j].x1; t.box[1] = dets[j].y1;
        t.box[2] = dets[j].x2; t.box[3] = dets[j].y2;
        t.missed = 0;
        tracks_.push_back(std::move(t));
    }
}

// =========================================================
// ROI mask
// =========================================================
void ControlLogic::update_roi_mask(const Pt* redL, const Pt* redR) {
    roi_mask_ = default_roi_mask_.clone();
    if (redL && redR) {
        std::vector<cv::Point> trap = {
            {redL->x, redL->y}, {redR->x, redR->y},
            {purpleR_.x, purpleR_.y}, {purpleL_.x, purpleL_.y}
        };
        cv::fillPoly(roi_mask_, std::vector<std::vector<cv::Point>>{trap}, cv::Scalar(1));
    }
}

// =========================================================
// Bbox overlap ratio
// =========================================================
double ControlLogic::bbox_overlap_ratio(const cv::Mat& mask,
                                         int x1, int y1, int x2, int y2,
                                         int stride) {
    int h = mask.rows, w = mask.cols;
    x1 = std::max(0, std::min(w - 1, x1));
    x2 = std::max(0, std::min(w,     x2));
    y1 = std::max(0, std::min(h - 1, y1));
    y2 = std::max(0, std::min(h,     y2));
    if (x2 <= x1 + 1 || y2 <= y1 + 1) return 0.0;

    int total = 0, hits = 0;
    for (int yy = y1; yy < y2; yy += stride) {
        const uint8_t* row = mask.ptr<uint8_t>(yy);
        for (int xx = x1; xx < x2; xx += stride) {
            ++total;
            if (row[xx]) ++hits;
        }
    }
    return (total > 0) ? static_cast<double>(hits) / total : 0.0;
}

// =========================================================
// IPM distance
// =========================================================
double ControlLogic::ipm_distance(int y_pix, int img_h) {
    double cy = img_h * 0.5;
    double fov_rad = FOV_VERTICAL_DEG * M_PI / 180.0;
    fov_rad = std::max(10.0 * M_PI / 180.0, std::min(170.0 * M_PI / 180.0, fov_rad));
    double fy = (img_h * 0.5) / std::tan(fov_rad * 0.5);

    double theta = std::atan((static_cast<double>(y_pix) - cy) / std::max(fy, 1e-6));
    double pitch  = PITCH_DOWN_DEG * M_PI / 180.0;
    double alpha  = theta + pitch;

    if (alpha <= 0.5 * M_PI / 180.0) return 1e6;

    double dist = CAMERA_HEIGHT_M / std::tan(alpha);
    return (dist < 0) ? 1e6 : dist;
}

// =========================================================
// Decide state
// =========================================================
CtrlState ControlLogic::decide_state(bool comm_ok) {
    // 매 프레임 호출 전에 update_tracks → state inputs 갱신 필요
    // 여기서는 tracks_ 을 순회하며 inputs 재계산

    min_car_ped_dist_m_.reset();
    min_sl_cw_dist_m_.reset();
    has_yellow_        = false;
    has_red_or_yellow_ = false;

    for (auto& trk : tracks_) {
        if (trk.missed > 0) continue;

        std::string cname = get_class_name(trk.cid);
        std::string cn = normalize_name(cname);

        float cx = 0.5f * (trk.box[0] + trk.box[2]);
        float cy_box = 0.5f * (trk.box[1] + trk.box[3]);
        float wb = std::max(1.0f, trk.box[2] - trk.box[0]);
        float hb = std::max(1.0f, trk.box[3] - trk.box[1]);

        float scx, scy, sw, sh;
        trk.kf.update(cx, cy_box, wb, hb, scx, scy, sw, sh);

        int sx1 = std::max(0, std::min(W_ - 1, static_cast<int>(scx - 0.5f * sw)));
        int sy1 = std::max(0, std::min(H_ - 1, static_cast<int>(scy - 0.5f * sh)));
        int sx2 = std::max(0, std::min(W_ - 1, static_cast<int>(scx + 0.5f * sw)));
        int sy2 = std::max(0, std::min(H_ - 1, static_cast<int>(scy + 0.5f * sh)));
        if (sx2 <= sx1 + 1 || sy2 <= sy1 + 1) continue;

        bool force_sig = is_allowed_vehicle_signal(cname);
        double overlap = bbox_overlap_ratio(roi_mask_, sx1, sy1, sx2, sy2, ROI_OVERLAP_STRIDE);
        if (!force_sig && overlap < OVERLAP_RATIO_TH) continue;

        // 대표점: bbox 밑변 중점
        int rep_y = sy2;
        double dist_m = ipm_distance(rep_y, H_);

        // signals
        if (SIG_YEL_N.count(cn)) { has_yellow_ = true; has_red_or_yellow_ = true; }
        else if (SIG_RED_N.count(cn)) { has_red_or_yellow_ = true; }

        // min distances
        if (CAR_NAMES_N.count(cn) || PED_NAMES_N.count(cn)) {
            if (!min_car_ped_dist_m_ || dist_m < *min_car_ped_dist_m_)
                min_car_ped_dist_m_ = dist_m;
        }
        if (STOPLINE_NAMES_N.count(cn) || CROSSWALK_NAMES_N.count(cn)) {
            if (!min_sl_cw_dist_m_ || dist_m < *min_sl_cw_dist_m_)
                min_sl_cw_dist_m_ = dist_m;
        }
    }

    // decision
    if (!comm_ok) return CtrlState::STOP;

    if (min_car_ped_dist_m_ && *min_car_ped_dist_m_ <= STOP_DIST_M) return CtrlState::STOP;
    if (min_car_ped_dist_m_ && *min_car_ped_dist_m_ <= SLOW_DIST_M) return CtrlState::SLOW;

    if (has_red_or_yellow_ && min_sl_cw_dist_m_) {
        if (*min_sl_cw_dist_m_ <= STOP_DIST_M) return CtrlState::STOP;
        if (*min_sl_cw_dist_m_ <= SLOW_DIST_M) return CtrlState::SLOW;
    }
    if (has_red_or_yellow_) return CtrlState::SLOW;

    return CtrlState::DRIVE;
}

}  // namespace arc
