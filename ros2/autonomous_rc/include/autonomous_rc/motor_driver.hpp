// =========================================================
// motor_driver.hpp — sysfs GPIO + PWM 제어
//   DC 모터 : 4×GPIO (IN1-IN4) + 1×PWM (speed)
//   서보     : 1×PWM (continuous rotation)
//
//   Xavier NX: GPIO는 Tegra 이름(PR.04 등)으로 접근
// =========================================================
#pragma once

#include <string>
#include "autonomous_rc/config.hpp"
#include "autonomous_rc/control_logic.hpp"

namespace arc {

// ─── sysfs GPIO (Tegra 이름 기반) ───
class SysfsGpio {
public:
    explicit SysfsGpio(const std::string& tegra_name);
    ~SysfsGpio();

    bool export_pin();
    bool set_direction_out();
    bool write_value(int val);
    void unexport_pin();

private:
    std::string name_;
    bool exported_ = false;
    std::string base_path_;
};

// ─── sysfs PWM ───
class SysfsPwm {
public:
    SysfsPwm(const std::string& chip_path, int channel);
    ~SysfsPwm();

    bool init(int freq_hz);
    void set_duty_pct(double duty_pct);
    void enable();
    void disable();
    void cleanup();

private:
    std::string base_;
    int  channel_;
    int  period_ns_ = 0;
    bool exported_  = false;

    bool export_channel();
    void write_attr(const std::string& attr, const std::string& val);
};

// ─── DC Motor ───
class DcMotor {
public:
    DcMotor();
    ~DcMotor();

    bool init();
    void set_forward(int duty_pct);
    void stop();
    void apply(CtrlState state);
    void cleanup();

private:
    SysfsGpio in1_{GPIO_IN1};
    SysfsGpio in2_{GPIO_IN2};
    SysfsGpio in3_{GPIO_IN3};
    SysfsGpio in4_{GPIO_IN4};
    SysfsPwm  pwm_{PWM_DC_CHIP, PWM_DC_CHANNEL};
};

// ─── Continuous Servo ───
class ContinuousServo {
public:
    ContinuousServo();
    ~ContinuousServo();

    bool init();
    void set_speed(double speed);
    void stop();
    SteerState transition(SteerState prev, SteerState next);
    void cleanup();

private:
    SysfsPwm pwm_{PWM_SERVO_CHIP, PWM_SERVO_CHANNEL};

    void pulse(char side, double seconds);

    static std::pair<char,int> parse_state(SteerState s);
    static char opposite_side(char side);
    static double get_go_time(char side, int level);
    static double get_ret_time(char side, int level);
};

}  // namespace arc
