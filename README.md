# autonomous_rc — ROS 2 Foxy C++ 자율주행 제어 노드

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                 autonomous_rc_node                       │
│  (단일 노드 · 단일 스레드 · 동기 루프)                    │
│                                                         │
│  ┌─────────┐  ┌─────────┐                               │
│  │ Lane    │  │ OD      │  TensorRT 8.x (FP16/FP32)    │
│  │ SegNet  │  │ YOLO    │                               │
│  └────┬────┘  └────┬────┘                               │
│       │            │                                    │
│  ┌────▼────────────▼────┐                               │
│  │   ControlLogic       │  차선→조향 / OD→IPM거리→상태   │
│  │   - Tracking         │                               │
│  │   - Kalman4D         │                               │
│  │   - IPM Distance     │                               │
│  └────┬────────────┬────┘                               │
│       │            │                                    │
│  ┌────▼────┐  ┌────▼────┐                               │
│  │ Servo   │  │ DC Motor│  sysfs GPIO + PWM             │
│  │ (steer) │  │ (speed) │                               │
│  └─────────┘  └─────────┘                               │
└─────────────────────────────────────────────────────────┘
```

## 전제 조건

| 항목 | 요구 사항 |
|------|----------|
| 보드 | NVIDIA Jetson Xavier NX |
| OS | Ubuntu 20.04 (JetPack 5.x / L4T R35) |
| ROS | ROS 2 Foxy |
| CUDA | JetPack 기본 포함 |
| TensorRT | 8.4+ (JetPack 기본 포함) |
| OpenCV | 4.x (JetPack 기본 포함) |

## 1단계: 핀 MUX 설정 (최초 1회, 재부팅 시 재실행)

```bash
# PWM 핀 활성화 (BOARD 32, 33)
sudo busybox devmem 0x2434040 w 1   # Pin 32 → PWM (Servo)
sudo busybox devmem 0x2434060 w 1   # Pin 33 → PWM (DC Motor)
```

> **중요**: devmem 주소는 Xavier NX 기준입니다.
> 다른 Jetson 모델은 주소가 다릅니다.
> `jetson-io.py` 로 대체 가능:
> ```bash
> sudo /opt/nvidia/jetson-io/jetson-io.py
> ```

## 2단계: PWM 칩 번호 확인

```bash
cat /sys/kernel/debug/pwm
# 또는
ls /sys/class/pwm/
```

출력 예시:
```
pwmchip0  pwmchip2
```

`config.hpp`의 `PWM_DC_CHIP`, `PWM_SERVO_CHIP` 값을 실제 칩 경로와 일치시키십시오.

- **BOARD 33** (DC Motor Speed) → 보통 `pwmchip0`
- **BOARD 32** (Servo Steer) → 보통 `pwmchip2`

칩 할당은 디바이스 트리에 따라 달라질 수 있으므로 **반드시 실물 확인 필수**.

## 3단계: GPIO 권한 설정

```bash
# 방법 A: udev rule (권장, 재부팅 후 유지)
sudo tee /etc/udev/rules.d/99-gpio-pwm.rules << 'EOF'
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", \
    RUN+="/bin/chmod 0666 /sys/class/gpio/export /sys/class/gpio/unexport"
SUBSYSTEM=="gpio", KERNEL=="gpio*", ACTION=="add", \
    RUN+="/bin/chmod 0666 /sys/class/gpio/gpio%n/direction /sys/class/gpio/gpio%n/value"
SUBSYSTEM=="pwm", ACTION=="add", \
    RUN+="/bin/chmod -R 0666 /sys/class/pwm/"
EOF
sudo udevadm control --reload-rules

# 방법 B: 임시 (세션 한정)
sudo chmod -R 666 /sys/class/gpio/export /sys/class/gpio/unexport
sudo chmod -R 666 /sys/class/pwm/
```

## 4단계: 빌드

```bash
cd ~/ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select autonomous_rc \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 5단계: 엔진 파일 배치

```
/home/user/Downloads/
├── starting.mp4              # 입력 영상
├── seg_regnety_fp16.engine   # Lane 세그멘테이션 엔진
├── OD.engine                 # Object Detection 엔진 (1,3,960,960)→(1,300,6)
└── classes.txt               # 클래스 이름 목록
```

## 6단계: 실행

```bash
# 기본 경로 사용
ros2 launch autonomous_rc run_system.py

# 경로 오버라이드
ros2 launch autonomous_rc run_system.py \
  base_dir:=/home/user/models \
  video:=test_drive.mp4

# 직접 실행 (디버그)
ros2 run autonomous_rc autonomous_rc_node \
  --ros-args \
  -p base_dir:=/home/user/Downloads \
  -p video:=starting.mp4
```

## GPIO 핀 매핑 요약

| 기능 | Python (BOARD) | C++ (Linux GPIO) | 인터페이스 |
|------|---------------|-------------------|-----------|
| IN1 (Motor A+) | 11 | 447 | sysfs GPIO |
| IN2 (Motor A-) | 12 | 462 | sysfs GPIO |
| IN3 (Motor B+) | 16 | 321 | sysfs GPIO |
| IN4 (Motor B-) | 15 | 488 | sysfs GPIO |
| DC Speed | 33 | pwmchip0/pwm0 | sysfs PWM |
| Servo Steer | 32 | pwmchip2/pwm0 | sysfs PWM |

> BOARD→Linux GPIO 번호는 **JetPack 5.x (L4T R35)** 기준입니다.
> JetPack 4.x 에서는 GPIO 번호가 다릅니다.

## 파일 구조

```
autonomous_rc/
├── CMakeLists.txt              # TensorRT/CUDA 링크 설정
├── package.xml                 # ROS 2 의존성
├── launch/
│   └── run_system.py           # 파라미터 주입 런치 파일
├── include/autonomous_rc/
│   ├── config.hpp              # 모든 상수 (Python 변수 1:1)
│   ├── trt_engine.hpp          # TensorRT 래퍼 헤더
│   ├── control_logic.hpp       # 차선/조향/IPM/트래킹 헤더
│   └── motor_driver.hpp        # GPIO/PWM 드라이버 헤더
└── src/
    ├── trt_engine.cpp          # TensorRT 추론 구현
    ├── control_logic.cpp       # 제어 로직 구현
    ├── motor_driver.cpp        # 하드웨어 제어 구현
    └── main.cpp                # ROS 2 노드 + 통합 루프
```

## 트러블슈팅

### PWM 동작 안 함
1. `cat /sys/kernel/debug/pwm` 으로 칩/채널 확인
2. devmem 핀 mux 재실행
3. `config.hpp`의 `PWM_DC_CHIP`, `PWM_SERVO_CHIP` 경로 확인

### GPIO Permission denied
1. `sudo` 로 실행하거나 udev rule 적용
2. `ls -la /sys/class/gpio/export` 권한 확인

### TRT 엔진 로드 실패
1. 엔진 파일이 **동일 GPU + 동일 TensorRT 버전**에서 빌드되었는지 확인
2. 동적 shape 엔진은 지원하지 않음 → fixed shape 으로 재빌드

### 서보 방향 반전
`motor_driver.cpp`의 `ContinuousServo::pulse()` 함수에서:
```cpp
double speed = (side == 'L') ? -1.0 : 1.0;
// 반대면:
double speed = (side == 'L') ? 1.0 : -1.0;
```
