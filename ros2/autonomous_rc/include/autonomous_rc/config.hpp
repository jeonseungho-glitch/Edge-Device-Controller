// =========================================================
// config.hpp — 전체 상수 정의 (Python engine_control_ipm.py 1:1 이식)
// Jetson Xavier NX · JetPack 5.x · ROS 2 Foxy
// =========================================================
#pragma once

#include <string>
#include <set>
#include <cmath>

namespace arc {  // autonomous_rc

// ─────────────────────────────────────────────
// 0) 파일 경로 (launch 파라미터로 오버라이드 가능)
// ─────────────────────────────────────────────
inline constexpr char DEFAULT_BASE_DIR[]       = "/home/user/Downloads";
inline constexpr char DEFAULT_VIDEO_NAME[]     = "starting.mp4";
inline constexpr char DEFAULT_LANE_ENGINE[]    = "seg_regnety_fp16.engine";
inline constexpr char DEFAULT_OD_ENGINE[]      = "OD.engine";
inline constexpr char DEFAULT_CLASSES_TXT[]    = "classes.txt";

// ─────────────────────────────────────────────
// 1) GPIO 핀 (sysfs Linux GPIO 번호)
//    Xavier NX BOARD→Linux GPIO 매핑 (JetPack 5.x / L4T R35)
//
//    BOARD 11 → GPIO 447   (IN1)
//    BOARD 12 → GPIO 462   (IN2)
//    BOARD 16 → GPIO 321   (IN3)
//    BOARD 15 → GPIO 488   (IN4)
//
//    ※ Python Jetson.GPIO BOARD 기준 11,12,16,15 순서와 동일.
//      jetson-io 또는 /opt/nvidia/jetson-io/jetson-io.py 로
//      핀 기능(GPIO) 이 먼저 활성화되어 있어야 합니다.
// ─────────────────────────────────────────────
inline constexpr char GPIO_IN1[] = "PR.04";    // BOARD 11
inline constexpr char GPIO_IN2[] = "PT.05";    // BOARD 12
inline constexpr char GPIO_IN3[] = "PY.04";    // BOARD 16
inline constexpr char GPIO_IN4[] = "PCC.04";   // BOARD 15

// ─────────────────────────────────────────────
// 2) PWM (sysfs)
//    Xavier NX 에서 BOARD 32, 33 은 각각 별도의 pwmchip 에 매핑됩니다.
//    `ls /sys/class/pwm/` 로 확인 후 아래 값을 맞추십시오.
//
//    BOARD 33 (DC Motor Speed) → pwmchip0 ch0  (일반적)
//    BOARD 32 (Servo Steer)    → pwmchip2 ch0  (일반적)
//
//    ※ 칩 번호가 환경마다 다를 수 있으므로 아래 경로를 반드시
//      실제 보드에서 확인하십시오.
//      확인법:
//        busybox devmem 0x2434060 w 1   # pin33 mux → pwm
//        busybox devmem 0x2434040 w 1   # pin32 mux → pwm
//        cat /sys/kernel/debug/pwm      # chip/channel 확인
// ─────────────────────────────────────────────
inline constexpr char PWM_DC_CHIP[]    = "/sys/class/pwm/pwmchip0";
inline constexpr int  PWM_DC_CHANNEL   = 0;

inline constexpr char PWM_SERVO_CHIP[] = "/sys/class/pwm/pwmchip2";
inline constexpr int  PWM_SERVO_CHANNEL = 0;

// ─────────────────────────────────────────────
// 3) DC Motor 파라미터
// ─────────────────────────────────────────────
inline constexpr int  PWM_HZ         = 1000;
inline constexpr int  DUTY_SLOW      = 40;    // %
inline constexpr int  DUTY_DRIVE     = 60;    // %
inline constexpr bool START_WITH_STOP = true;

// ─────────────────────────────────────────────
// 4) Servo (continuous rotation) 파라미터
// ─────────────────────────────────────────────
inline constexpr int    SERVO_HZ       = 50;
inline constexpr double SERVO_NEUTRAL  = 7.039065;  // duty%
inline constexpr double SERVO_SPAN     = 1.6;       // duty% range

inline constexpr double STEER_DEAD     = 4.0;   // deg
inline constexpr double STEER_L1       = 8.0;   // deg

// Steering pulse 시간(초)
inline constexpr double L1_GO  = 0.18;
inline constexpr double L2_GO  = 0.30;
inline constexpr double R1_GO  = 0.20;
inline constexpr double R2_GO  = 0.45;
inline constexpr double L1_RET = 0.12;
inline constexpr double L2_RET = 0.20;
inline constexpr double R1_RET = 0.11;
inline constexpr double R2_RET = 0.18;

// ─────────────────────────────────────────────
// 5) Lane 세그멘테이션 파라미터
// ─────────────────────────────────────────────
inline constexpr int    LANE_W = 960;
inline constexpr int    LANE_H = 544;
inline constexpr float  LANE_THR = 0.5f;

inline constexpr double ROI_Y0_RATIO  = 0.45;
inline constexpr double Y_LINE_RATIO  = 0.62;
inline constexpr int    SEARCH_HALF_WIDTH = 210;

inline constexpr int    MIN_WIDTH_PX   = 220;
inline constexpr int    MAX_WIDTH_PX   = 1100;
inline constexpr int    MIN_SEG_LEN_PX = 8;
inline constexpr int    SEG_GAP_PX     = 3;

inline constexpr double PURPLE_Y_RATIO = 0.93;
inline constexpr int    DEFAULT_LANE_HALF_WIDTH = 600;

// ─────────────────────────────────────────────
// 6) 조향 (Steer) 파라미터
// ─────────────────────────────────────────────
inline constexpr double STEER_GAIN_DEG     = 55.0;
inline constexpr double STEER_CLIP_DEG     = 70.0;
inline constexpr double MAX_STEER_STEP_DEG = 6.0;
inline constexpr double SMOOTH_MIN = 0.08;
inline constexpr double SMOOTH_MAX = 0.25;

// ─────────────────────────────────────────────
// 7) OD 파라미터
// ─────────────────────────────────────────────
inline constexpr float CONF_THRESH = 0.25f;
inline constexpr int   IMGSZ       = 960;

// ─────────────────────────────────────────────
// 8) IPM (Inverse Perspective Mapping) 거리 파라미터
// ─────────────────────────────────────────────
inline constexpr double CAMERA_HEIGHT_M  = 1.20;
inline constexpr double FOV_VERTICAL_DEG = 60.0;
inline constexpr double PITCH_DOWN_DEG   = 0.0;

inline constexpr double STOP_DIST_M = 5.63;
inline constexpr double SLOW_DIST_M = 6.5;

// ─────────────────────────────────────────────
// 9) ROI / Tracking
// ─────────────────────────────────────────────
inline constexpr int    ROI_OVERLAP_STRIDE = 6;
inline constexpr double OVERLAP_RATIO_TH   = 0.10;

inline constexpr double TRACK_IOU_TH  = 0.30;
inline constexpr int    TRACK_MAX_MISS = 10;

// ─────────────────────────────────────────────
// 10) 출력 주기
// ─────────────────────────────────────────────
inline constexpr int TIMING_EVERY = 10;
inline constexpr int PRINT_EVERY  = 1;

// ─────────────────────────────────────────────
// 11) ImageNet 정규화
// ─────────────────────────────────────────────
inline constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
inline constexpr float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// ─────────────────────────────────────────────
// 12) 클래스 이름 집합
// ─────────────────────────────────────────────
inline const std::set<std::string> CAR_NAMES_N =
    {"car", "vehicle", "truck", "bus"};

inline const std::set<std::string> PED_NAMES_N =
    {"person", "pedestrian", "ped"};

inline const std::set<std::string> STOPLINE_NAMES_N =
    {"stop_line", "stopline", "stop_line"};

inline const std::set<std::string> CROSSWALK_NAMES_N =
    {"crosswalk", "zebra_crossing", "zebra"};

inline const std::set<std::string> SIG_RED_N =
    {"vehicular_signal_red"};

inline const std::set<std::string> SIG_YEL_N =
    {"vehicular_signal_yellow"};

inline const std::set<std::string> SIG_GRN_N =
    {"vehicular_signal_green"};

inline const std::set<std::string> VEH_SIGNAL_ALLOW_N =
    {"vehicular_signal_red", "vehicular_signal_yellow", "vehicular_signal_green"};

// ─────────────────────────────────────────────
// FPS EMA
// ─────────────────────────────────────────────
inline constexpr double FPS_EMA_ALPHA = 0.12;

}  // namespace arc
