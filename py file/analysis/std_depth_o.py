#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================
# FINAL (JETSON FAST CONSOLE) - TRT8 + Servo(continuous) + DC Motor
# ✅ no saving
# ✅ 서보 로직: 사용자 제공 로직 + (전이 로직 그대로)
# ✅ STATE: (d가 클수록 멀다) 기준 그대로 유지
# ✅ [CHG] 실시간 프레임 표시(imshow) 제거
# ✅ [NEW] 첫 프레임(idx=1) 제외, 모든 프레임의 통계 출력
#     - LANE/OD: 매 프레임 측정
#     - DEPTH: 매 프레임 측정 (run_depth=False면 0ms 기록) + run_depth=True 프레임만 별도 통계
#     - FPS(inst): min/max/mean/std  (idx=1 제외)
# =========================================================

import os
import time
import cv2
import numpy as np
import warnings

# --- TensorRT <-> NumPy 호환 경고 억제용 (구버전 환경 대비) ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        _ = np.bool
    except Exception:
        np.bool = bool

import Jetson.GPIO as GPIO

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from transformers import AutoImageProcessor

# =========================================================
# 0) PATH / SETTINGS  (네 환경에 맞게 여기만 수정)
# =========================================================
BASE_DIR = "/home/user/Downloads"

VIDEO_PATH = os.path.join(BASE_DIR, "starting.mp4")

LANE_ENGINE_PATH  = os.path.join(BASE_DIR, "seg_regnety_fp16.engine")
DEPTH_ENGINE_PATH = os.path.join(BASE_DIR, "depth.engine")      # (1,3,518,518) -> (1,518,518)
OD_ENGINE_PATH    = os.path.join(BASE_DIR, "OD.engine")         # (1,3,960,960) -> (1,300,6)

CLASSES_TXT = os.path.join(BASE_DIR, "classes.txt")

# =========================================================
# 1) MOTOR PINS (DC motor)
# =========================================================
SPEED_PIN = 33
IN1, IN2 = 11, 12
IN3, IN4 = 16, 15

PWM_HZ = 1000
DUTY_SLOW = 30
DUTY_DRIVE = 50
START_WITH_STOP = True

# =========================================================
# 2) STEER SERVO (continuous)
# =========================================================
STEER_SERVO_PIN = 32
SERVO_HZ = 50

SERVO_NEUTRAL = 7.039065  # NEED TO BE EDITTED
SERVO_SPAN = 1.6          # STEERING SPEED, NEED TO BE EDITTED

STEER_DEAD = 3.0
STEER_L1 = 6.0

# Steering seconds (튜닝값)
L1_GO  = 0.18
L2_GO  = 0.30
R1_GO  = 0.20
R2_GO  = 0.45

L1_RET = 0.12
L2_RET = 0.20
R1_RET = 0.11
R2_RET = 0.18

# =========================================================
# 3) PARAMS
# =========================================================
LANE_W, LANE_H = 960, 544
LANE_THR = 0.5

ROI_Y0_RATIO = 0.45
Y_LINE_RATIO = 0.62
SEARCH_HALF_WIDTH = 210

MIN_WIDTH_PX = 220
MAX_WIDTH_PX = 1100
MIN_SEG_LEN_PX = 8
SEG_GAP_PX = 3

PURPLE_Y_RATIO = 0.93
DEFAULT_LANE_HALF_WIDTH = 600

STEER_GAIN_DEG = 55.0
STEER_CLIP_DEG = 70.0
MAX_STEER_STEP_DEG = 6.0
SMOOTH_MIN = 0.08
SMOOTH_MAX = 0.25

CONF = 0.25
IMGSZ = 960

# Depth/state thresholds
TH_RED = 0.25      # slow
TH_YELLOW = 0.32   # stop
DEPTH_INFER_EVERY = 2
DEPTH_INTERP_MODE = cv2.INTER_LINEAR

ROI_OVERLAP_STRIDE = 6
OVERLAP_RATIO_TH   = 0.10

CAR_NAMES = {"car", "vehicle", "truck", "bus"}
PED_NAMES = {"person", "pedestrian", "ped"}
STOPLINE_NAMES = {"stop_line", "stopline", "stop-line"}
CROSSWALK_NAMES = {"crosswalk", "zebra_crossing", "zebra"}

SIG_RED = {"vehicular_signal_red"}
SIG_YEL = {"vehicular_signal_yellow"}
SIG_GRN = {"vehicular_signal_green"}

VEH_SIGNAL_ALLOW = {"vehicular_signal_red", "vehicular_signal_yellow", "vehicular_signal_green"}

PRINT_EVERY  = 1

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_IN = 518

# =========================================================
# 4) MOTOR CONTROL
# =========================================================
def motor_gpio_init():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([SPEED_PIN, IN1, IN2, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)
    pwm = GPIO.PWM(SPEED_PIN, PWM_HZ)
    pwm.start(0)
    print("[MOTOR] GPIO initialized.")
    return pwm

def motor_set_forward(pwm, duty):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm.ChangeDutyCycle(float(duty))

def motor_stop(pwm):
    pwm.ChangeDutyCycle(0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

def apply_state_to_motor(state, pwm):
    if state == "STOP":
        motor_stop(pwm)
    elif state == "SLOW":
        motor_set_forward(pwm, DUTY_SLOW)
    elif state == "DRIVE":
        motor_set_forward(pwm, DUTY_DRIVE)
    else:
        motor_stop(pwm)

# =========================================================
# 5) SERVO (continuous)
# =========================================================
def servo_gpio_init_cont():
    GPIO.setup(STEER_SERVO_PIN, GPIO.OUT, initial=GPIO.LOW)
    spwm = GPIO.PWM(STEER_SERVO_PIN, SERVO_HZ)
    spwm.start(0)
    return spwm

def servo_set_speed(spwm, speed):
    """
    speed: -1.0 ~ +1.0
    """
    speed = float(max(-1.0, min(1.0, speed)))
    duty = SERVO_NEUTRAL + speed * SERVO_SPAN
    spwm.ChangeDutyCycle(duty)

def servo_stop(spwm):
    spwm.ChangeDutyCycle(SERVO_NEUTRAL)

def steer_to_state(steer_deg):
    """
    steer_deg: prev_angle (deg)
    return: 'S0', 'L1', 'L2', 'R1', 'R2'
    """
    s = float(steer_deg)
    if -STEER_DEAD <= s <= STEER_DEAD:
        return "S0"
    side = "R" if s > 0 else "L"
    mag = abs(s)
    if mag <= STEER_L1:
        return f"{side}1"
    else:
        return f"{side}2"

def parse_state(st):
    if st == "S0":
        return ("S", 0)
    return (st[0], int(st[1]))  # ('L' or 'R', 1 or 2)

def opposite_side(side: str) -> str:
    return "R" if side == "L" else "L"

def get_go_time(side, level):
    if side == "L" and level == 1:
        return L1_GO
    if side == "L" and level == 2:
        return L2_GO
    if side == "R" and level == 1:
        return R1_GO
    if side == "R" and level == 2:
        return R2_GO
    return 0.0

def get_ret_time(side, level):
    if side == "L" and level == 1:
        return L1_RET
    if side == "L" and level == 2:
        return L2_RET
    if side == "R" and level == 1:
        return R1_RET
    if side == "R" and level == 2:
        return R2_RET
    return 0.0

def servo_pulse(spwm, side, seconds):
    """
    side: 'L' or 'R' (이 방향으로 회전)
    seconds: duration to rotate
    """
    seconds = float(seconds)
    if seconds <= 0.0:
        servo_stop(spwm)
        return
    # NOTE: 좌/우가 실제 하드웨어와 반대면 아래 speed 부호만 swap 하면 됨.
    speed = -1.0 if side == "L" else +1.0
    servo_set_speed(spwm, speed)
    time.sleep(seconds)
    servo_stop(spwm)

def apply_steer_state_transition(spwm, prev_state, new_state):
    """
    - S0 -> L/R : GO(same)
    - L/R -> S0 : RET(opposite)
    - same side:
        * L1->L2, R1->R2 : (GO_high - GO_low) same side
        * L2->L1, R2->R1 : (RET_high - RET_low) opposite side
    - change side:
        prev -> center (RET opposite) then center -> new (GO)
    """
    if new_state == prev_state:
        return prev_state

    prev_side, prev_lv = parse_state(prev_state)
    new_side, new_lv = parse_state(new_state)

    # 1) S0 -> L/R : GO (same direction)
    if prev_state == "S0" and new_state != "S0":
        t = get_go_time(new_side, new_lv)
        servo_pulse(spwm, new_side, t)
        return new_state

    # 2) L/R -> S0 : RET (opposite direction)
    if new_state == "S0" and prev_state != "S0":
        t = get_ret_time(prev_side, prev_lv)
        servo_pulse(spwm, opposite_side(prev_side), t)
        return new_state

    # 3) both non-center
    # 3a) same direction
    if prev_side == new_side:
        # 더 꺾기: GO 차이만큼 같은 방향
        if new_lv > prev_lv:
            t = get_go_time(prev_side, new_lv) - get_go_time(prev_side, prev_lv)
            servo_pulse(spwm, prev_side, t)
            return new_state

        # 덜 꺾기: RET 차이만큼 반대 방향
        if new_lv < prev_lv:
            t = get_ret_time(prev_side, prev_lv) - get_ret_time(prev_side, new_lv)
            servo_pulse(spwm, opposite_side(prev_side), t)
            return new_state

        return new_state

    # 3b) different direction: prev -> center (RET opposite) then center -> new (GO)
    t1 = get_ret_time(prev_side, prev_lv)
    servo_pulse(spwm, opposite_side(prev_side), t1)

    t2 = get_go_time(new_side, new_lv)
    servo_pulse(spwm, new_side, t2)

    return new_state

# =========================================================
# 6) UTILS
# =========================================================
def clamp_step(prev, target, max_step):
    d = target - prev
    if d > max_step:  return prev + max_step
    if d < -max_step: return prev - max_step
    return target

def compute_dynamic_smoothing(width, prev_width):
    if width is None or prev_width is None or prev_width < 1e-6:
        return SMOOTH_MIN
    rel = abs(width - prev_width) / max(prev_width, 1.0)
    t = np.clip(rel / 0.5, 0.0, 1.0)
    return float((1.0 - t) * SMOOTH_MAX + t * SMOOTH_MIN)

def normalize_name(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def sigmoid_safe(x):
    x = np.clip(x.astype(np.float32), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

CAR_NAMES_N = {normalize_name(x) for x in CAR_NAMES}
PED_NAMES_N = {normalize_name(x) for x in PED_NAMES}
STOPLINE_NAMES_N = {normalize_name(x) for x in STOPLINE_NAMES}
CROSSWALK_NAMES_N = {normalize_name(x) for x in CROSSWALK_NAMES}
SIG_RED_N = {normalize_name(x) for x in SIG_RED}
SIG_YEL_N = {normalize_name(x) for x in SIG_YEL}
SIG_GRN_N = {normalize_name(x) for x in SIG_GRN}
VEH_SIGNAL_ALLOW_N = {normalize_name(x) for x in VEH_SIGNAL_ALLOW}

def is_allowed_vehicle_signal(cname: str) -> bool:
    return normalize_name(cname) in VEH_SIGNAL_ALLOW_N

def load_classes_txt(path):
    if not os.path.exists(path):
        return None
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                names.append(t)
    return names if names else None

def get_class_name(names, cid: int):
    if names is None:
        return str(cid)
    if 0 <= cid < len(names):
        return str(names[cid])
    return str(cid)

# =========================================================
# 7) TensorRT 8.x Runner
# =========================================================
class TRTInfer8:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create execution context")

        self.engine = engine
        self.context = context

        self.bindings = [None] * engine.num_bindings
        self.stream = cuda.Stream()

        self.in_idx = None
        self.out_idxs = []
        self.host_in = None
        self.dev_in = None
        self.host_out = []
        self.dev_out = []
        self.out_shapes = []
        self.out_dtypes = []

        for i in range(engine.num_bindings):
            is_input = engine.binding_is_input(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = tuple(engine.get_binding_shape(i))

            if any(d <= 0 for d in shape):
                raise RuntimeError(f"Dynamic shape binding found at {i}: {shape} (rebuild fixed engine)")

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype=dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[i] = int(dev_mem)

            if is_input:
                if self.in_idx is not None:
                    raise RuntimeError("Expected exactly 1 input binding per engine.")
                self.in_idx = i
                self.host_in = host_mem
                self.dev_in = dev_mem
                self.in_shape = shape
                self.in_dtype = dtype
            else:
                self.out_idxs.append(i)
                self.host_out.append(host_mem)
                self.dev_out.append(dev_mem)
                self.out_shapes.append(shape)
                self.out_dtypes.append(dtype)

        if self.in_idx is None or len(self.out_idxs) < 1:
            raise RuntimeError("Invalid engine bindings (need 1 input and >=1 output).")

    def infer(self, inp: np.ndarray):
        if inp.dtype != self.in_dtype:
            inp = inp.astype(self.in_dtype, copy=False)
        inp = np.ascontiguousarray(inp)

        if inp.size != self.host_in.size:
            raise ValueError(f"Input size mismatch: inp.size={inp.size} vs engine expects {self.host_in.size}")

        np.copyto(self.host_in, inp.ravel())
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)

        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v2 failed")

        for h, d in zip(self.host_out, self.dev_out):
            cuda.memcpy_dtoh_async(h, d, self.stream)

        self.stream.synchronize()

        outs = []
        for h, shp, dt in zip(self.host_out, self.out_shapes, self.out_dtypes):
            outs.append(np.array(h, dtype=dt).reshape(shp))
        return outs[0] if len(outs) == 1 else outs

# =========================================================
# 8) Lane preprocess + red points
# =========================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def lane_preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (LANE_W, LANE_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2,0,1)[None, ...]

def segments_from_row_xs(xs, gap=SEG_GAP_PX):
    if xs.size == 0:
        return []
    xs = np.sort(xs)
    cuts = np.where(np.diff(xs) > gap)[0]
    segs = []
    s = 0
    for c in cuts:
        part = xs[s:c+1]
        x0, x1 = int(part[0]), int(part[-1])
        seg_len = int(part.size)
        center = 0.5*(x0+x1)
        segs.append((x0, x1, center, seg_len))
        s = c+1
    part = xs[s:]
    x0, x1 = int(part[0]), int(part[-1])
    seg_len = int(part.size)
    center = 0.5*(x0+x1)
    segs.append((x0, x1, center, seg_len))
    return segs

def pick_red_points_from_lane_out(lane_out_1x1hw, orig_W, orig_H, y_line_orig, mid_x_orig, xL_orig, xR_orig,
                                  lane_out_is_logit=True):
    sx = LANE_W / float(orig_W)
    sy = LANE_H / float(orig_H)
    y_lane = int(np.clip(round(y_line_orig * sy), 0, LANE_H-1))
    xL_lane = int(np.clip(round(xL_orig * sx), 0, LANE_W-1))
    xR_lane = int(np.clip(round(xR_orig * sx), xL_lane+1, LANE_W))

    row = lane_out_1x1hw[0,0,y_lane, xL_lane:xR_lane].astype(np.float32)
    row_prob = sigmoid_safe(row) if lane_out_is_logit else row

    xs = np.where(row_prob >= LANE_THR)[0]
    if xs.size == 0:
        return None, None
    xs = xs + xL_lane

    seg_gap_lane = max(1, int(round(SEG_GAP_PX * sx)))
    segs = segments_from_row_xs(xs, gap=seg_gap_lane)

    min_seg_lane = max(1, int(round(MIN_SEG_LEN_PX * sx)))
    segs = [s for s in segs if s[3] >= min_seg_lane]
    if not segs:
        return None, None

    mid_lane = mid_x_orig * sx
    left_segs  = [s for s in segs if s[2] < mid_lane]
    right_segs = [s for s in segs if s[2] >= mid_lane]
    if (not left_segs) or (not right_segs):
        return None, None

    bestL = min(left_segs,  key=lambda s: abs(mid_lane - s[2]))
    bestR = min(right_segs, key=lambda s: abs(s[2] - mid_lane))

    lx_lane = float(bestL[2])
    rx_lane = float(bestR[2])
    if rx_lane <= lx_lane:
        return None, None

    lx = int(np.clip(round(lx_lane / sx), 0, orig_W-1))
    rx = int(np.clip(round(rx_lane / sx), 0, orig_W-1))
    y  = int(np.clip(y_line_orig, 0, orig_H-1))

    width = rx - lx
    if not (MIN_WIDTH_PX <= width <= MAX_WIDTH_PX):
        return None, None

    return (lx, y), (rx, y)

def compute_steer_from_midline(redL, redR, purpleL, purpleR):
    if redL is None or redR is None:
        return 0.0, None
    c1x = 0.5*(redL[0] + redR[0])
    c1y = 0.5*(redL[1] + redR[1])
    c0x = 0.5*(purpleL[0] + purpleR[0])
    c0y = 0.5*(purpleL[1] + purpleR[1])
    dy = (c0y - c1y)
    if abs(dy) < 1e-6:
        return 0.0, None
    dx = (c1x - c0x)
    slope = dx / dy
    steer = float(np.clip(slope * STEER_GAIN_DEG, -STEER_CLIP_DEG, STEER_CLIP_DEG))
    width = float(redR[0] - redL[0])
    return steer, width

# =========================================================
# 9) ROI mask
# =========================================================
def make_default_roi_polygon(W, H):
    y_top = int(H * 0.70)
    y_bot = int(H * PURPLE_Y_RATIO)
    tl = int(W * 0.45); tr = int(W * 0.55)
    bl = int(W * 0.32); br = int(W * 0.68)
    pts = np.array([[tl, y_top], [tr, y_top], [br, y_bot], [bl, y_bot]], dtype=np.int32).reshape(-1, 1, 2)
    return pts

def trapezoid_poly_from_points(redL, redR, purpleL, purpleR):
    if redL is None or redR is None:
        return None
    pts = np.array([redL, redR, purpleR, purpleL], dtype=np.int32).reshape(-1,1,2)
    return pts

def bbox_overlap_ratio_blue_roi(blue_roi_mask_u8, x1, y1, x2, y2, stride=ROI_OVERLAP_STRIDE):
    h, w = blue_roi_mask_u8.shape
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w,     int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return 0.0
    xs = np.arange(x1, x2, stride, dtype=np.int32)
    ys = np.arange(y1, y2, stride, dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    patch = blue_roi_mask_u8[np.ix_(ys, xs)]
    total = patch.size
    if total <= 0:
        return 0.0
    hits = int(np.count_nonzero(patch))
    return float(hits) / float(total)

# =========================================================
# 10) DEPTH preprocess (AutoImageProcessor)
# =========================================================
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)

def depth_preprocess_518(frame_bgr):
    small_bgr = cv2.resize(frame_bgr, (DEPTH_IN, DEPTH_IN), interpolation=DEPTH_INTERP_MODE)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    out = depth_processor(images=small_rgb, return_tensors="np")
    pv = out["pixel_values"].astype(np.float32)  # (1,3,518,518)
    return pv

def depth_normalize_percentile_518(pred_518):
    dmin, dmax = np.percentile(pred_518, 2), np.percentile(pred_518, 98)
    dn = (pred_518 - dmin) / (dmax - dmin + 1e-6)
    return np.clip(dn, 0.0, 1.0).astype(np.float32)

def sample_depth_from_518(depth518_norm, x_orig, y_orig, W0, H0):
    sx = DEPTH_IN / float(W0)
    sy = DEPTH_IN / float(H0)
    xd = int(np.clip(round(x_orig * sx), 0, DEPTH_IN-1))
    yd = int(np.clip(round(y_orig * sy), 0, DEPTH_IN-1))
    return float(depth518_norm[yd, xd])

# =========================================================
# 11) STATE logic (d가 클수록 멀어짐)  [원문 그대로]
# =========================================================
def decide_control_state(comm_ok,
                         min_car_or_ped_d,
                         has_yellow_signal,
                         has_red_or_yellow_signal,
                         min_stopline_or_crosswalk_d):
    if not comm_ok:
        return "STOP"

    # d >= TH_YELLOW -> STOP
    if (min_car_or_ped_d is not None) and (min_car_or_ped_d >= TH_YELLOW):
        return "STOP"
    # TH_RED <= d < TH_YELLOW -> SLOW
    if (min_car_or_ped_d is not None) and (TH_RED <= min_car_or_ped_d < TH_YELLOW):
        return "SLOW"

    if has_red_or_yellow_signal and (min_stopline_or_crosswalk_d is not None):
        if min_stopline_or_crosswalk_d >= TH_YELLOW:
            return "STOP"
        if min_stopline_or_crosswalk_d >= TH_RED:
            return "SLOW"
    if has_red_or_yellow_signal:
        return "SLOW"

    return "DRIVE"

# =========================================================
# 12) Tracking + Kalman
# =========================================================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class Kalman4D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.6
        self.initialized = False

    def update(self, cx, cy, w, h):
        z = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]])
        if not self.initialized:
            self.kf.statePre = z.copy()
            self.kf.statePost = z.copy()
            self.initialized = True
        self.kf.correct(z)
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])

class Track:
    def __init__(self, tid, cid, box_xyxy):
        self.tid = tid
        self.cid = cid
        self.kf = Kalman4D()
        self.last_box = box_xyxy
        self.missed = 0

    def update(self, box_xyxy):
        self.last_box = box_xyxy
        self.missed = 0

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(1, x2-x1)
    h = max(1, y2-y1)
    cx = x1 + 0.5*w
    cy = y1 + 0.5*h
    return cx, cy, w, h

def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return [x1,y1,x2,y2]

def update_tracks(tracks, dets, iou_th=0.3, max_missed=10):
    used_det = set()
    for tr in tracks:
        best_iou = 0.0
        best_j = -1
        for j, (cid, box, conf) in enumerate(dets):
            if j in used_det:
                continue
            if cid != tr.cid:
                continue
            v = iou_xyxy(tr.last_box, box)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_th:
            tr.update(dets[best_j][1])
            used_det.add(best_j)
        else:
            tr.missed += 1
    tracks = [t for t in tracks if t.missed <= max_missed]
    next_id = (max([t.tid for t in tracks]) + 1) if tracks else 0
    for j, (cid, box, conf) in enumerate(dets):
        if j in used_det:
            continue
        tracks.append(Track(next_id, cid, box))
        next_id += 1
    return tracks

# =========================================================
# 13) OD decode (letterbox + 1x300x6)
# =========================================================
def letterbox(im, new_shape=640, color=(114,114,114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def decode_od_300x6(output_1x300x6, orig_hw, r, pad, conf_th=0.25):
    H0, W0 = orig_hw
    left, top = pad
    a = output_1x300x6[0]
    dets = []
    for i in range(a.shape[0]):
        x1,y1,x2,y2,score,cls = a[i]
        score = float(score)
        if score < conf_th:
            continue
        x1 = (float(x1) - left) / r
        x2 = (float(x2) - left) / r
        y1 = (float(y1) - top)  / r
        y2 = (float(y2) - top)  / r
        x1 = float(np.clip(x1, 0, W0-1)); x2 = float(np.clip(x2, 0, W0-1))
        y1 = float(np.clip(y1, 0, H0-1)); y2 = float(np.clip(y2, 0, H0-1))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        dets.append((int(cls), [x1,y1,x2,y2], score))
    return dets

# =========================================================
# 14) TIMING/FPS STATS
# =========================================================
def stats_arr(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }

def print_stats_ms(title, s):
    if s is None:
        print(f"{title}: (no samples)")
        return
    print(
        f"{title}: n={s['n']} | min={s['min']:.3f}ms | max={s['max']:.3f}ms | "
        f"mean={s['mean']:.3f}ms | std={s['std']:.3f}ms"
    )

def print_stats_fps(title, s):
    if s is None:
        print(f"{title}: (no samples)")
        return
    print(
        f"{title}: n={s['n']} | min={s['min']:.3f}fps | max={s['max']:.3f}fps | "
        f"mean={s['mean']:.3f}fps | std={s['std']:.3f}fps"
    )

# =========================================================
# 15) MAIN
# =========================================================
def main():
    print("VIDEO :", VIDEO_PATH)
    print("OD    :", OD_ENGINE_PATH)
    print("LANE  :", LANE_ENGINE_PATH)
    print("DEPTH :", DEPTH_ENGINE_PATH)

    for p in [VIDEO_PATH, OD_ENGINE_PATH, LANE_ENGINE_PATH, DEPTH_ENGINE_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    class_names = load_classes_txt(CLASSES_TXT)
    if class_names is None:
        print("[WARN] classes.txt not found -> class-name matching may not work.")
    else:
        print(f"✅ Loaded class names from classes.txt (n={len(class_names)})")

    pwm = motor_gpio_init()
    spwm = None
    last_state = None
    steer_state = "S0"

    # timing arrays (idx=1 제외 -> idx>=2부터 append)
    lane_times_ms = []
    depth_times_ms = []          # 매 프레임 기록(run_depth=False면 0ms)
    depth_times_run_ms = []      # run_depth=True 프레임만 기록
    od_times_ms = []
    fps_inst_list = []           # ✅ [NEW] idx>=2 instantaneous FPS 기록

    try:
        spwm = servo_gpio_init_cont()
        servo_stop(spwm)

        if START_WITH_STOP:
            apply_state_to_motor("STOP", pwm)
            last_state = "STOP"

        lane_trt  = TRTInfer8(LANE_ENGINE_PATH)
        depth_trt = TRTInfer8(DEPTH_ENGINE_PATH)
        od_trt    = TRTInfer8(OD_ENGINE_PATH)

        print(f"[INFO] LANE  input: {lane_trt.in_shape} dtype={lane_trt.in_dtype} out_shapes={lane_trt.out_shapes}")
        print(f"[INFO] DEPTH input: {depth_trt.in_shape} dtype={depth_trt.in_dtype} out_shapes={depth_trt.out_shapes}")
        print(f"[INFO] OD    input: {od_trt.in_shape} dtype={od_trt.in_dtype} out_shapes={od_trt.out_shapes}")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_src = fps_src if fps_src and fps_src > 0 else 30.0

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nVIDEO: {W}x{H}, src_fps={fps_src:.2f}, frames={nframes}")

        y_line = int(H * Y_LINE_RATIO)
        mid_x  = W // 2
        xL = int(np.clip(mid_x - SEARCH_HALF_WIDTH, 0, W - 1))
        xR = int(np.clip(mid_x + SEARCH_HALF_WIDTH, 0, W - 1))

        y_purple = int(H * PURPLE_Y_RATIO)
        pxL = int(np.clip(mid_x - DEFAULT_LANE_HALF_WIDTH, 0, W - 1))
        pxR = int(np.clip(mid_x + DEFAULT_LANE_HALF_WIDTH, 0, W - 1))
        purpleL = (pxL, y_purple)
        purpleR = (pxR, y_purple)

        default_roi_poly = make_default_roi_polygon(W, H)
        default_roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(default_roi_mask, [default_roi_poly], 1)

        prev_angle = 0.0
        prev_width = None
        comm_ok = True
        tracks = []

        depth_cache_518 = None

        fps_ema = 0.0
        FPS_EMA_ALPHA = 0.12
        t_prev = time.perf_counter()

        lane_out_is_logit = True
        lane_out_checked = False

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            # FPS (inst + EMA)
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_inst = (1.0 / dt) if dt > 1e-9 else 0.0
            fps_ema = fps_inst if fps_ema <= 1e-6 else (1.0 - FPS_EMA_ALPHA) * fps_ema + FPS_EMA_ALPHA * fps_inst

            if idx >= 2:
                fps_inst_list.append(fps_inst)

            # ---- LANE (매 프레임 타이밍) ----
            t0 = time.perf_counter()
            x_lane = lane_preprocess(frame).astype(lane_trt.in_dtype, copy=False)
            out_lane = lane_trt.infer(x_lane)  # (1,1,544,960)
            seg_ms = (time.perf_counter() - t0) * 1000.0

            if not lane_out_checked:
                tmp = out_lane[0,0].astype(np.float32)
                mn, mx = float(tmp.min()), float(tmp.max())
                lane_out_is_logit = not (mn >= -0.05 and mx <= 1.05)
                lane_out_checked = True
                print(f"[INFO] Lane output stats: min={mn:.3f} max={mx:.3f} -> lane_out_is_logit={lane_out_is_logit}")

            redL, redR = pick_red_points_from_lane_out(
                out_lane, W, H, y_line, mid_x, xL, xR, lane_out_is_logit=lane_out_is_logit
            )
            red_ok = (redL is not None and redR is not None)

            # ---- DEPTH (매 프레임 타이밍: run_depth False면 0ms 기록) ----
            run_depth = (depth_cache_518 is None) or (idx % DEPTH_INFER_EVERY == 0)

            depth_ms = 0.0
            if run_depth:
                t0 = time.perf_counter()
                pv = depth_preprocess_518(frame).astype(depth_trt.in_dtype, copy=False)
                pred = depth_trt.infer(pv)  # (1,518,518)
                depth_cache_518 = depth_normalize_percentile_518(pred[0].astype(np.float32))
                depth_ms = (time.perf_counter() - t0) * 1000.0

            # ---- STEER ----
            if red_ok:
                target_angle, width = compute_steer_from_midline(redL, redR, purpleL, purpleR)
                alpha = compute_dynamic_smoothing(width, prev_width)
                prev_width = width if width is not None else prev_width
                target_angle = clamp_step(prev_angle, target_angle, MAX_STEER_STEP_DEG)
                current_angle = prev_angle * (1.0 - alpha) + target_angle * alpha
            else:
                current_angle = prev_angle * (1.0 - SMOOTH_MIN)
            prev_angle = float(current_angle)

            new_steer_state = steer_to_state(prev_angle)
            if new_steer_state != steer_state:
                steer_state = apply_steer_state_transition(spwm, steer_state, new_steer_state)
            else:
                servo_stop(spwm)

            # ---- ROI mask ----
            blue_roi_mask = default_roi_mask
            if red_ok:
                trap_poly = trapezoid_poly_from_points(redL, redR, purpleL, purpleR)
                if trap_poly is not None:
                    blue_roi_mask = default_roi_mask.copy()
                    cv2.fillPoly(blue_roi_mask, [trap_poly], 1)

            # ---- OD (매 프레임 타이밍) ----
            t0 = time.perf_counter()
            im_lb, r, (left, top) = letterbox(frame, new_shape=IMGSZ)
            im_rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
            x_od = (im_rgb.astype(np.float32) / 255.0).transpose(2,0,1)[None, ...]
            x_od = x_od.astype(od_trt.in_dtype, copy=False)
            out_od = od_trt.infer(x_od)  # (1,300,6)
            od_ms = (time.perf_counter() - t0) * 1000.0

            dets = decode_od_300x6(out_od, (H, W), r, (left, top), conf_th=CONF)
            tracks = update_tracks(tracks, dets, iou_th=0.30, max_missed=10)

            # ---- STATE INPUTS ----
            min_car_ped_d = None
            min_stopline_crosswalk_d = None
            has_yellow_signal = False
            has_red_or_yellow_signal = False

            for trk in tracks:
                if trk.missed > 0:
                    continue

                cid = trk.cid
                cname = get_class_name(class_names, cid)
                cn = normalize_name(cname)

                x1, y1, x2, y2 = trk.last_box
                cx, cy, w_box, h_box = xyxy_to_cxcywh([x1, y1, x2, y2])
                cx_s, cy_s, w_s, h_s = trk.kf.update(cx, cy, w_box, h_box)
                sx1, sy1, sx2, sy2 = cxcywh_to_xyxy(cx_s, cy_s, w_s, h_s)

                sx1 = int(max(0, min(W - 1, sx1)))
                sx2 = int(max(0, min(W - 1, sx2)))
                sy1 = int(max(0, min(H - 1, sy1)))
                sy2 = int(max(0, min(H - 1, sy2)))
                if sx2 <= sx1 + 1 or sy2 <= sy1 + 1:
                    continue

                force_vehicle_signal = is_allowed_vehicle_signal(cname)
                overlap_ratio = bbox_overlap_ratio_blue_roi(
                    blue_roi_mask, sx1, sy1, sx2, sy2, stride=ROI_OVERLAP_STRIDE
                )
                keep = force_vehicle_signal or (overlap_ratio >= OVERLAP_RATIO_TH)
                if not keep:
                    continue

                scx = int(0.5 * (sx1 + sx2))
                scy = int(0.5 * (sy1 + sy2))
                d = sample_depth_from_518(depth_cache_518, scx, scy, W, H)

                if cn in SIG_YEL_N:
                    has_yellow_signal = True
                    has_red_or_yellow_signal = True
                elif cn in SIG_RED_N:
                    has_red_or_yellow_signal = True

                if (cn in CAR_NAMES_N) or (cn in PED_NAMES_N):
                    if (min_car_ped_d is None) or (d < min_car_ped_d):
                        min_car_ped_d = d

                if (cn in STOPLINE_NAMES_N) or (cn in CROSSWALK_NAMES_N):
                    if (min_stopline_crosswalk_d is None) or (d < min_stopline_crosswalk_d):
                        min_stopline_crosswalk_d = d

            state = decide_control_state(
                comm_ok=comm_ok,
                min_car_or_ped_d=min_car_ped_d,
                has_yellow_signal=has_yellow_signal,
                has_red_or_yellow_signal=has_red_or_yellow_signal,
                min_stopline_or_crosswalk_d=min_stopline_crosswalk_d
            )

            if state != last_state:
                apply_state_to_motor(state, pwm)
                last_state = state

            # ---- TIMING RECORD (idx=1 제외) ----
            if idx >= 2:
                lane_times_ms.append(seg_ms)
                od_times_ms.append(od_ms)
                depth_times_ms.append(depth_ms if run_depth else 0.0)
                if run_depth:
                    depth_times_run_ms.append(depth_ms)

            # ---- PRINT ----
            if idx % PRINT_EVERY == 0:
                print(
                    f"[{idx:05d}] FPS={fps_ema:5.1f} | steer={prev_angle:+6.2f} | STEER_STATE={steer_state:2s} | "
                    f"STATE={state:5s} | SEG={seg_ms:6.1f}ms DEPTH={depth_ms:6.1f}ms OD={od_ms:6.1f}ms | "
                    f"run_depth={run_depth} tracks={len(tracks)} dets={len(dets)} | "
                    f"min_cp={-1 if min_car_ped_d is None else min_car_ped_d:.3f} "
                    f"min_slcw={-1 if min_stopline_crosswalk_d is None else min_stopline_crosswalk_d:.3f} "
                    f"sigRY={has_red_or_yellow_signal} red_ok={red_ok}"
                )

        cap.release()

        # ---- SUMMARY STATS ----
        print("\n==================== SUMMARY (exclude idx=1) ====================")
        print_stats_ms("LANE inference", stats_arr(lane_times_ms))
        print_stats_ms("OD inference", stats_arr(od_times_ms))
        print_stats_ms(
            f"DEPTH inference (per-frame, 0ms when skipped; every={DEPTH_INFER_EVERY})",
            stats_arr(depth_times_ms)
        )
        print_stats_ms("DEPTH inference (run_depth=True frames only)", stats_arr(depth_times_run_ms))
        print_stats_fps("FPS(inst) per-frame", stats_arr(fps_inst_list))
        print("=================================================================\n")

        print("✅ Done.")

    finally:
        try:
            apply_state_to_motor("STOP", pwm)
            pwm.stop()
        except Exception:
            pass

        try:
            if spwm is not None:
                servo_stop(spwm)
                spwm.stop()
        except Exception:
            pass

        GPIO.cleanup()

if __name__ == "__main__":
    main()