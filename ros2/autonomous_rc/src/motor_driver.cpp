// =========================================================
// motor_driver.cpp — sysfs GPIO + PWM 제어 구현
//   Xavier NX: GPIO는 Tegra 이름(PR.04 등)으로 접근
// =========================================================
#include "autonomous_rc/motor_driver.hpp"
#include "autonomous_rc/config.hpp"

#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

namespace arc {

// ─── sysfs 파일 쓰기 헬퍼 ───
static bool write_file(const std::string& path, const std::string& val) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[GPIO/PWM] Cannot write " << path << "\n";
        return false;
    }
    ofs << val;
    ofs.close();
    return true;
}

// ═════════════════════════════════════════════
// SysfsGpio (Tegra 이름 기반)
// ═════════════════════════════════════════════
SysfsGpio::SysfsGpio(const std::string& tegra_name)
    : name_(tegra_name) {
    base_path_ = "/sys/class/gpio/" + name_;
}

SysfsGpio::~SysfsGpio() {
    if (exported_) unexport_pin();
}

bool SysfsGpio::export_pin() {
    // 이미 존재하면 스킵
    std::ifstream test(base_path_ + "/direction");
    if (test.good()) {
        exported_ = true;
        return true;
    }

    // Tegra 이름으로는 export 불가 → Python Jetson.GPIO가 이미 export 해둔 상태여야 함
    // 또는 숫자 GPIO를 알아야 하는데, 이미 export 되어 있으므로 패스
    std::cerr << "[GPIO] " << name_ << " not found. Run Python GPIO setup first.\n";
    return false;
}

bool SysfsGpio::set_direction_out() {
    return write_file(base_path_ + "/direction", "out");
}

bool SysfsGpio::write_value(int val) {
    return write_file(base_path_ + "/value", std::to_string(val));
}

void SysfsGpio::unexport_pin() {
    // Tegra 이름 핀은 unexport하지 않음 (다른 프로세스와 공유 가능)
    exported_ = false;
}

// ═════════════════════════════════════════════
// SysfsPwm
// ═════════════════════════════════════════════
SysfsPwm::SysfsPwm(const std::string& chip_path, int channel)
    : channel_(channel) {
    base_ = chip_path + "/pwm" + std::to_string(channel_);
}

SysfsPwm::~SysfsPwm() {
    cleanup();
}

bool SysfsPwm::export_channel() {
    std::ifstream test(base_ + "/period");
    if (test.good()) {
        exported_ = true;
        return true;
    }

    std::string chip_path = base_.substr(0, base_.rfind("/pwm"));
    if (!write_file(chip_path + "/export", std::to_string(channel_))) {
        std::cerr << "[PWM] Failed to export ch " << channel_ << " on " << chip_path << "\n";
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    exported_ = true;
    return true;
}

bool SysfsPwm::init(int freq_hz) {
    if (!export_channel()) return false;

    period_ns_ = static_cast<int>(1e9 / freq_hz);

    write_attr("duty_cycle", "0");
    write_attr("period", std::to_string(period_ns_));
    write_attr("duty_cycle", "0");
    write_attr("enable", "1");

    std::cout << "[PWM] " << base_ << " init: freq=" << freq_hz
              << "Hz period=" << period_ns_ << "ns\n";
    return true;
}

void SysfsPwm::set_duty_pct(double duty_pct) {
    duty_pct = std::max(0.0, std::min(100.0, duty_pct));
    int duty_ns = static_cast<int>(period_ns_ * duty_pct / 100.0);
    write_attr("duty_cycle", std::to_string(duty_ns));
}

void SysfsPwm::enable() {
    write_attr("enable", "1");
}

void SysfsPwm::disable() {
    write_attr("duty_cycle", "0");
    write_attr("enable", "0");
}

void SysfsPwm::cleanup() {
    if (exported_) {
        disable();
        std::string chip_path = base_.substr(0, base_.rfind("/pwm"));
        write_file(chip_path + "/unexport", std::to_string(channel_));
        exported_ = false;
    }
}

void SysfsPwm::write_attr(const std::string& attr, const std::string& val) {
    write_file(base_ + "/" + attr, val);
}

// ═════════════════════════════════════════════
// DcMotor
// ═════════════════════════════════════════════
DcMotor::DcMotor()
    : in1_(GPIO_IN1), in2_(GPIO_IN2),
      in3_(GPIO_IN3), in4_(GPIO_IN4),
      pwm_(PWM_DC_CHIP, PWM_DC_CHANNEL) {}

DcMotor::~DcMotor() { cleanup(); }

bool DcMotor::init() {
    for (auto* pin : {&in1_, &in2_, &in3_, &in4_}) {
        if (!pin->export_pin())       return false;
        if (!pin->set_direction_out()) return false;
        pin->write_value(0);
    }
    if (!pwm_.init(PWM_HZ)) return false;
    pwm_.set_duty_pct(0.0);

    std::cout << "[MOTOR] DC motor initialized. GPIO="
              << GPIO_IN1 << "," << GPIO_IN2 << ","
              << GPIO_IN3 << "," << GPIO_IN4 << "\n";
    return true;
}

void DcMotor::set_forward(int duty_pct) {
    in1_.write_value(1);
    in2_.write_value(0);
    in3_.write_value(1);
    in4_.write_value(0);
    pwm_.set_duty_pct(static_cast<double>(duty_pct));
}

void DcMotor::stop() {
    pwm_.set_duty_pct(0.0);
    in1_.write_value(0);
    in2_.write_value(0);
    in3_.write_value(0);
    in4_.write_value(0);
}

void DcMotor::apply(CtrlState state) {
    switch (state) {
        case CtrlState::STOP:  stop(); break;
        case CtrlState::SLOW:  set_forward(DUTY_SLOW); break;
        case CtrlState::DRIVE: set_forward(DUTY_DRIVE); break;
    }
}

void DcMotor::cleanup() {
    stop();
    pwm_.cleanup();
}

// ═════════════════════════════════════════════
// ContinuousServo
// ═════════════════════════════════════════════
ContinuousServo::ContinuousServo()
    : pwm_(PWM_SERVO_CHIP, PWM_SERVO_CHANNEL) {}

ContinuousServo::~ContinuousServo() { cleanup(); }

bool ContinuousServo::init() {
    if (!pwm_.init(SERVO_HZ)) return false;
    stop();
    std::cout << "[SERVO] Continuous servo initialized. neutral=" << SERVO_NEUTRAL
              << " span=" << SERVO_SPAN << "\n";
    return true;
}

void ContinuousServo::set_speed(double speed) {
    speed = std::max(-1.0, std::min(1.0, speed));
    double duty = SERVO_NEUTRAL + speed * SERVO_SPAN;
    pwm_.set_duty_pct(duty);
}

void ContinuousServo::stop() {
    pwm_.set_duty_pct(SERVO_NEUTRAL);
}

void ContinuousServo::pulse(char side, double seconds) {
    if (seconds <= 0.0) {
        stop();
        return;
    }
    double speed = (side == 'L') ? -1.0 : 1.0;
    set_speed(speed);
    std::this_thread::sleep_for(
        std::chrono::microseconds(static_cast<int64_t>(seconds * 1e6)));
    stop();
}

std::pair<char,int> ContinuousServo::parse_state(SteerState s) {
    switch (s) {
        case SteerState::S0: return {'S', 0};
        case SteerState::L1: return {'L', 1};
        case SteerState::L2: return {'L', 2};
        case SteerState::R1: return {'R', 1};
        case SteerState::R2: return {'R', 2};
    }
    return {'S', 0};
}

char ContinuousServo::opposite_side(char side) {
    return (side == 'L') ? 'R' : 'L';
}

double ContinuousServo::get_go_time(char side, int level) {
    if (side == 'L' && level == 1) return L1_GO;
    if (side == 'L' && level == 2) return L2_GO;
    if (side == 'R' && level == 1) return R1_GO;
    if (side == 'R' && level == 2) return R2_GO;
    return 0.0;
}

double ContinuousServo::get_ret_time(char side, int level) {
    if (side == 'L' && level == 1) return L1_RET;
    if (side == 'L' && level == 2) return L2_RET;
    if (side == 'R' && level == 1) return R1_RET;
    if (side == 'R' && level == 2) return R2_RET;
    return 0.0;
}

SteerState ContinuousServo::transition(SteerState prev, SteerState next) {
    if (prev == next) return prev;

    auto [ps, pl] = parse_state(prev);
    auto [ns, nl] = parse_state(next);

    if (prev == SteerState::S0 && next != SteerState::S0) {
        double t = get_go_time(ns, nl);
        pulse(ns, t);
        return next;
    }

    if (next == SteerState::S0 && prev != SteerState::S0) {
        double t = get_ret_time(ps, pl);
        pulse(opposite_side(ps), t);
        return next;
    }

    if (ps == ns) {
        if (nl > pl) {
            double t = get_go_time(ps, nl) - get_go_time(ps, pl);
            pulse(ps, t);
            return next;
        }
        if (nl < pl) {
            double t = get_ret_time(ps, pl) - get_ret_time(ps, nl);
            pulse(opposite_side(ps), t);
            return next;
        }
        return next;
    }

    double t1 = get_ret_time(ps, pl);
    pulse(opposite_side(ps), t1);

    double t2 = get_go_time(ns, nl);
    pulse(ns, t2);

    return next;
}

void ContinuousServo::cleanup() {
    stop();
    pwm_.cleanup();
}

}  // namespace arc
