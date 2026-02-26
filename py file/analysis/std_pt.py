#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import Jetson.GPIO as GPIO
from ultralytics import YOLO
# from transformers import AutoImageProcessor#, DepthAnythingForDepthEstimation


BASE_DIR = "/home/user/Downloads"

VIDEO_PATH = os.path.join(BASE_DIR, "starting.mp4")
OD_PT_PATH = os.path.join(BASE_DIR, "OD.pt")
LANE_PT_PATH = os.path.join(BASE_DIR, "Lane.pt")
DEPTH_PTH_PATH = os.path.join(BASE_DIR, "depth.pth")


SPEED_PIN = 33
IN1, IN2 = 11, 12
IN3, IN4 = 16, 15

PWM_HZ = 1000
DUTY_SLOW = 35
DUTY_DRIVE = 80
START_WITH_STOP = True

# =========================
# STEER SERVO (continuous)
# =========================
STEER_SERVO_PIN = 32
SERVO_HZ = 50

SERVO_NEUTRAL = 7.039065  # NEED TO BE EDITTED
SERVO_SPAN = 1.6          # STEERING SPEED, NEED TO BE EDITTED

STEER_DEAD = 1.0
STEER_L1 = 2.0

# [MOD] Steering second
L1_GO  = 0.24  # left level 1
L2_GO  = 0.40  # left level 2
R1_GO  = 0.16  # right
R2_GO  = 0.25  #

L1_RET = 0.14  # left return level 1
L2_RET = 0.18  # left 2
R1_RET = 0.15  # right 1
R2_RET = 0.30  # right 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
IOU = 0.60
IMGSZ = 960

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

TH_RED = 0.20
TH_YELLOW = 0.30

DEPTH_INFER_EVERY = 2
DEPTH_INPUT_SIZE = 518
DEPTH_INTERP_MODE = cv2.INTER_LINEAR

ROI_OVERLAP_STRIDE = 6
OVERLAP_RATIO_TH = 0.10

VEH_SIGNAL_ALLOW = {"vehicular_signal_red", "vehicular_signal_yellow", "vehicular_signal_green"}

CAR_NAMES = {"car", "vehicle", "truck", "bus"}
PED_NAMES = {"person", "pedestrian", "ped"}
STOPLINE_NAMES = {"stop_line", "stopline", "stop-line"}
CROSSWALK_NAMES = {"crosswalk", "zebra_crossing", "zebra"}

SIG_RED = {"vehicular_signal_red"}
SIG_YEL = {"vehicular_signal_yellow"}
SIG_GRN = {"vehicular_signal_green"}

TIMING_EVERY = 10
PRINT_EVERY = 1

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================
# ✅ [ADDED] Online stats (Welford) : min/max/mean/std
# =========================================================
class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def update(self, x: float):
        x = float(x)
        self.n += 1
        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.n <= 0:
            return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
        var = (self.M2 / (self.n - 1)) if self.n > 1 else 0.0
        std = float(np.sqrt(max(var, 0.0)))
        return {
            "n": int(self.n),
            "min": float(self.min_v),
            "max": float(self.max_v),
            "mean": float(self.mean),
            "std": std,
        }


def motor_gpio_init():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([SPEED_PIN, IN1, IN2, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)
    pwm = GPIO.PWM(SPEED_PIN, PWM_HZ)
    pwm.start(0)
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
# Servo (continuous) - state based
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
    # Servo neutral value(stop)
    spwm.ChangeDutyCycle(SERVO_NEUTRAL)


# [MOD] transforming steer value to state value
def steer_to_state(steer_deg):
    """
    steer_deg: prev_angle (deg)
    return: 'S0', 'L1', 'L2', 'R1', 'R2'
    """
    s = float(steer_deg)

    if -STEER_DEAD <= s <= STEER_DEAD:
        return "S0"

    # NOTE: if direction opposite, edit here
    side = "R" if s > 0 else "L"
    mag = abs(s)

    if mag <= STEER_L1:
        return f"{side}1"
    else:
        return f"{side}2"


# [MOD] state parsing
def parse_state(st):
    if st == "S0":
        return ("S", 0)
    return (st[0], int(st[1]))  # ('L' or 'R', 1 or 2)


# [MOD] GO/RET time lookup
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


# [MOD] actual pulse function
def servo_pulse(spwm, side, seconds):
    """
    side: 'L' or 'R'
    seconds: duration to rotate
    """
    seconds = float(seconds)
    if seconds <= 0.0:
        servo_stop(spwm)
        return

    # NOTE: if direction opposite edit speed sign
    speed = -1.0 if side == "L" else +1.0
    servo_set_speed(spwm, speed)
    time.sleep(seconds)
    servo_stop(spwm)


# [MOD] prev_state -> new_state
def apply_steer_state_transition(spwm, prev_state, new_state):
    if new_state == prev_state:
        return prev_state

    prev_side, prev_lv = parse_state(prev_state)
    new_side, new_lv = parse_state(new_state)

    # 1) center -> L/R
    if prev_state == "S0" and new_state != "S0":
        t = get_go_time(new_side, new_lv)
        servo_pulse(spwm, new_side, t)
        return new_state

    # 2) L/R -> center
    if new_state == "S0" and prev_state != "S0":
        t = get_ret_time(prev_side, prev_lv)
        servo_pulse(spwm, prev_side, t)
        return new_state

    # 3) both non-center
    # 3a) if same direction
    if prev_side == new_side:
        if new_lv > prev_lv:
            # ex: L1 -> L2 : (L2_GO - L1_GO)
            t = get_go_time(prev_side, new_lv) - get_go_time(prev_side, prev_lv)
            servo_pulse(spwm, prev_side, t)
        else:
            # ex :  L2 -> L1 : (L2_RET - L1_RET)
            t = get_ret_time(prev_side, prev_lv) - get_ret_time(prev_side, new_lv)
            servo_pulse(spwm, prev_side, t)
        return new_state

    # 3b) if different direction : prev -> center (RET) and then center -> new (GO)
    # ex : L1 -> R2 : L1_RET + R2_GO
    t1 = get_ret_time(prev_side, prev_lv)
    servo_pulse(spwm, prev_side, t1)

    t2 = get_go_time(new_side, new_lv)
    servo_pulse(spwm, new_side, t2)

    return new_state


def _sync_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def clamp_step(prev, target, max_step):
    d = target - prev
    if d > max_step:
        return prev + max_step
    if d < -max_step:
        return prev - max_step
    return target


def compute_dynamic_smoothing(width, prev_width):
    if width is None or prev_width is None or prev_width < 1e-6:
        return SMOOTH_MIN
    rel = abs(width - prev_width) / max(prev_width, 1.0)
    t = np.clip(rel / 0.5, 0.0, 1.0)
    return float((1.0 - t) * SMOOTH_MAX + t * SMOOTH_MIN)


def normalize_name(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def get_class_name(names, cid):
    if isinstance(names, dict):
        return str(names.get(cid, cid))
    if isinstance(names, (list, tuple)) and cid < len(names):
        return str(names[cid])
    return str(cid)


def is_allowed_vehicle_signal(cname):
    return normalize_name(cname) in VEH_SIGNAL_ALLOW


def load_lane_model_from_pt(path, device):
    m = torch.load(path, map_location="cpu")
    if hasattr(m, "eval"):
        m.eval()
    if device == "cuda":
        m = m.to(device).half()
    else:
        m = m.to(device)
    return m


@torch.no_grad()
def lane_get_prediction_fixed(frame_bgr, model):
    h, w = frame_bgr.shape[:2]
    img = cv2.resize(frame_bgr, (LANE_W, LANE_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        x = x.half()
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 4 and logits.shape[1] != 1:
        logits = logits[:, :1, :, :]
    prob = torch.sigmoid(logits)
    prob_up = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    mask = (prob_up[0, 0].float().cpu().numpy() >= LANE_THR).astype(np.uint8)
    return mask


def road_roi(mask):
    h, w = mask.shape
    y0 = int(h * ROI_Y0_RATIO)
    out = np.zeros_like(mask)
    out[y0:, :] = mask[y0:, :]
    return out


def segments_from_row_xs(xs, gap=SEG_GAP_PX):
    if xs.size == 0:
        return []
    xs = np.sort(xs)
    cuts = np.where(np.diff(xs) > gap)[0]
    segs = []
    s = 0
    for c in cuts:
        part = xs[s:c + 1]
        x0, x1 = int(part[0]), int(part[-1])
        seg_len = int(part.size)
        center = 0.5 * (x0 + x1)
        segs.append((x0, x1, center, seg_len))
        s = c + 1
    part = xs[s:]
    x0, x1 = int(part[0]), int(part[-1])
    seg_len = int(part.size)
    center = 0.5 * (x0 + x1)
    segs.append((x0, x1, center, seg_len))
    return segs


def pick_red_points_on_scanline(mask_roi, y_line, mid_x, xL, xR):
    h, w = mask_roi.shape
    y = int(np.clip(y_line, 0, h - 1))
    xL = int(np.clip(xL, 0, w - 1))
    xR = int(np.clip(xR, xL + 1, w))
    xs = np.where(mask_roi[y, xL:xR] > 0)[0]
    if xs.size == 0:
        return None, None
    xs = xs + xL
    segs = segments_from_row_xs(xs)
    segs = [s for s in segs if s[3] >= MIN_SEG_LEN_PX]
    if not segs:
        return None, None
    left_segs = [s for s in segs if s[2] < mid_x]
    right_segs = [s for s in segs if s[2] >= mid_x]
    if not left_segs or not right_segs:
        return None, None
    bestL = min(left_segs, key=lambda s: abs(mid_x - s[2]))
    bestR = min(right_segs, key=lambda s: abs(s[2] - mid_x))
    lx = int(bestL[2])
    rx = int(bestR[2])
    if rx <= lx:
        return None, None
    width = rx - lx
    if not (MIN_WIDTH_PX <= width <= MAX_WIDTH_PX):
        return None, None
    return (lx, y), (rx, y)


def compute_steer_from_midline(redL, redR, purpleL, purpleR):
    if redL is None or redR is None:
        return 0.0, None
    c1x = 0.5 * (redL[0] + redR[0])
    c1y = 0.5 * (redL[1] + redR[1])
    c0x = 0.5 * (purpleL[0] + purpleR[0])
    c0y = 0.5 * (purpleL[1] + purpleR[1])
    dy = (c0y - c1y)
    if abs(dy) < 1e-6:
        return 0.0, None
    dx = (c1x - c0x)
    slope = dx / dy
    steer = float(np.clip(slope * STEER_GAIN_DEG, -STEER_CLIP_DEG, STEER_CLIP_DEG))
    width = float(redR[0] - redL[0])
    return steer, width


def make_default_roi_polygon(W, H):
    y_top = int(H * 0.70)
    y_bot = int(H * PURPLE_Y_RATIO)
    tl = int(W * 0.45)
    tr = int(W * 0.55)
    bl = int(W * 0.32)
    br = int(W * 0.68)
    pts = np.array([[tl, y_top], [tr, y_top], [br, y_bot], [bl, y_bot]], dtype=np.int32).reshape(-1, 1, 2)
    return pts


def trapezoid_poly_from_points(redL, redR, purpleL, purpleR):
    if redL is None or redR is None:
        return None
    pts = np.array([redL, redR, purpleR, purpleL], dtype=np.int32).reshape(-1, 1, 2)
    return pts


def bbox_overlap_ratio_blue_roi(blue_roi_mask_u8, x1, y1, x2, y2, stride=ROI_OVERLAP_STRIDE):
    h, w = blue_roi_mask_u8.shape
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
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


def load_depth_model(model_id, device):
    import sys

    repo_path = '/home/user/Downloads/Depth-Anything-V2'
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from depth_anything_v2.dpt import DepthAnythingV2

    depth_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    base_model = DepthAnythingV2(**depth_model_configs['vits'])

    weight_path = '/home/user/Downloads/depth.pth'
    base_model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    class CustomBatchFeature(dict):
        def to(self, device):
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    self[k] = v.to(device)
            return self

    class CustomProcessor:
        def __call__(self, images, return_tensors="pt"):
            img = images.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return CustomBatchFeature({"x": tensor})

    processor = CustomProcessor()

    class DummyOutput:
        def __init__(self, depth):
            self.predicted_depth = depth

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            depth = self.model(x)
            return DummyOutput(depth)

    model = ModelWrapper(base_model)
    model.to(device).eval()

    return processor, model


@torch.no_grad()
def infer_depth_map_bgr_fast(frame_bgr, processor, model, device, in_size=384):
    H, W = frame_bgr.shape[:2]
    small_bgr = cv2.resize(frame_bgr, (in_size, in_size), interpolation=DEPTH_INTERP_MODE)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=small_rgb, return_tensors="pt").to(device)
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
    pred = outputs.predicted_depth
    pred_up = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False).squeeze(1).squeeze(0)
    depth = pred_up.detach().float().cpu().numpy()
    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth_n = (depth - dmin) / (dmax - dmin + 1e-6)
    depth_n = np.clip(depth_n, 0.0, 1.0).astype(np.float32)
    return depth_n


def decide_control_state(comm_ok, min_car_or_ped_d, has_yellow_signal, has_red_or_yellow_signal, min_stopline_or_crosswalk_d):
    if not comm_ok:
        return "STOP"
    if (min_car_or_ped_d is not None) and (min_car_or_ped_d <= TH_RED):
        return "STOP"
    if has_red_or_yellow_signal and (min_stopline_or_crosswalk_d is not None) and (min_stopline_or_crosswalk_d <= TH_RED):
        return "STOP"
    if (min_car_or_ped_d is not None) and (TH_RED < min_car_or_ped_d <= TH_YELLOW):
        return "SLOW"
    if has_red_or_yellow_signal:
        return "SLOW"
    return "DRIVE"


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
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

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
    x1, y1, x2, y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return [x1, y1, x2, y2]


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
# ✅ [ADDED] pretty print for stats
# =========================================================
def _print_stats_block(title, stats_dict):
    s = stats_dict
    if s["n"] <= 0:
        print(f"{title:<10s}: n=0")
        return
    print(
        f"{title:<10s}: n={s['n']:6d} | "
        f"min={s['min']:8.3f} max={s['max']:8.3f} | "
        f"mean={s['mean']:8.3f} std={s['std']:8.3f}"
    )


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)
    if not os.path.exists(OD_PT_PATH):
        raise FileNotFoundError(OD_PT_PATH)
    if not os.path.exists(LANE_PT_PATH):
        raise FileNotFoundError(LANE_PT_PATH)

    pwm = motor_gpio_init()
    spwm = servo_gpio_init_cont()

    steer_state = "S0"
    servo_stop(spwm)

    last_state = None

    # =========================================================
    # ✅ [ADDED] stats collectors (first frame excluded)
    # =========================================================
    fps_stats = OnlineStats()
    steer_stats = OnlineStats()
    segms_stats = OnlineStats()
    depthms_stats = OnlineStats()
    odms_stats = OnlineStats()

    try:
        if START_WITH_STOP:
            apply_state_to_motor("STOP", pwm)
            last_state = "STOP"

        lane_model = load_lane_model_from_pt(LANE_PT_PATH, DEVICE)
        od_model = YOLO(OD_PT_PATH)
        depth_proc, depth_model = load_depth_model(DEPTH_MODEL_ID, DEVICE)

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError("cannot open video")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        y_line = int(H * Y_LINE_RATIO)
        mid_x = W // 2
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
        depth_cache = None

        fps_ema = 0.0
        FPS_EMA_ALPHA = 0.12
        t_prev = time.perf_counter()

        seg_ms = 0.0
        depth_ms = 0.0
        od_ms = 0.0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_inst = (1.0 / dt) if dt > 1e-9 else 0.0
            fps_ema = fps_inst if fps_ema <= 1e-6 else (1.0 - FPS_EMA_ALPHA) * fps_ema + FPS_EMA_ALPHA * fps_inst

            do_timing = (idx == 1) or (idx % TIMING_EVERY == 0)

            if do_timing:
                _sync_cuda()
                t0 = time.perf_counter()
            mask = lane_get_prediction_fixed(frame, lane_model)
            if do_timing:
                _sync_cuda()
                seg_ms = (time.perf_counter() - t0) * 1000.0
            mask_roi = road_roi(mask)

            run_depth = (depth_cache is None) or (idx % DEPTH_INFER_EVERY == 0)
            if do_timing:
                _sync_cuda()
                t0 = time.perf_counter()
            if run_depth:
                depth_cache = infer_depth_map_bgr_fast(frame, depth_proc, depth_model, DEVICE, in_size=DEPTH_INPUT_SIZE)
            depth_map = depth_cache
            if do_timing:
                _sync_cuda()
                depth_ms = (time.perf_counter() - t0) * 1000.0

            redL, redR = pick_red_points_on_scanline(mask_roi, y_line, mid_x, xL, xR)
            red_ok = (redL is not None and redR is not None)

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

            blue_roi_mask = default_roi_mask
            if red_ok:
                trap_poly = trapezoid_poly_from_points(redL, redR, purpleL, purpleR)
                if trap_poly is not None:
                    blue_roi_mask = default_roi_mask.copy()
                    cv2.fillPoly(blue_roi_mask, [trap_poly], 1)

            if do_timing:
                _sync_cuda()
                t0 = time.perf_counter()
            results = od_model.predict(
                source=frame,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                device=0 if DEVICE == "cuda" else "cpu",
                verbose=False
            )
            if do_timing:
                _sync_cuda()
                od_ms = (time.perf_counter() - t0) * 1000.0

            r = results[0]
            names = od_model.names

            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.detach().cpu().numpy()
                clss = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros((len(boxes),), int)
                confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.ones((len(boxes),), float)
                for (x1, y1, x2, y2), cid, cf in zip(boxes, clss, confs):
                    x1 = float(max(0, min(W - 1, x1)))
                    x2 = float(max(0, min(W - 1, x2)))
                    y1 = float(max(0, min(H - 1, y1)))
                    y2 = float(max(0, min(H - 1, y2)))
                    if x2 <= x1 + 1 or y2 <= y1 + 1:
                        continue
                    dets.append((int(cid), [x1, y1, x2, y2], float(cf)))

            tracks = update_tracks(tracks, dets, iou_th=0.30, max_missed=10)

            min_car_ped_d = None
            min_stopline_crosswalk_d = None
            has_yellow_signal = False
            has_red_or_yellow_signal = False

            for tr in tracks:
                if tr.missed > 0:
                    continue

                cid = tr.cid
                cname = get_class_name(names, cid)
                cn = normalize_name(cname)

                x1, y1, x2, y2 = tr.last_box
                cx, cy, w_box, h_box = xyxy_to_cxcywh([x1, y1, x2, y2])
                cx_s, cy_s, w_s, h_s = tr.kf.update(cx, cy, w_box, h_box)
                sx1, sy1, sx2, sy2 = cxcywh_to_xyxy(cx_s, cy_s, w_s, h_s)

                sx1 = int(max(0, min(W - 1, sx1)))
                sx2 = int(max(0, min(W - 1, sx2)))
                sy1 = int(max(0, min(H - 1, sy1)))
                sy2 = int(max(0, min(H - 1, sy2)))
                if sx2 <= sx1 + 1 or sy2 <= sy1 + 1:
                    continue

                force_vehicle_signal = is_allowed_vehicle_signal(cname)
                overlap_ratio = bbox_overlap_ratio_blue_roi(blue_roi_mask, sx1, sy1, sx2, sy2, stride=ROI_OVERLAP_STRIDE)
                keep = force_vehicle_signal or (overlap_ratio >= OVERLAP_RATIO_TH)
                if not keep:
                    continue

                scx = int(0.5 * (sx1 + sx2))
                scy = int(0.5 * (sy1 + sy2))
                scx = max(0, min(W - 1, scx))
                scy = max(0, min(H - 1, scy))
                d = float(depth_map[scy, scx])

                if cn in SIG_YEL:
                    has_yellow_signal = True
                    has_red_or_yellow_signal = True
                elif cn in SIG_RED:
                    has_red_or_yellow_signal = True

                if (cn in CAR_NAMES) or (cn in PED_NAMES):
                    if (min_car_ped_d is None) or (d < min_car_ped_d):
                        min_car_ped_d = d

                if (cn in STOPLINE_NAMES) or (cn in CROSSWALK_NAMES):
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

            # =========================================================
            # ✅ [ADDED] update stats (exclude first frame)
            # - FPS/steer: every frame (except idx==1)
            # - seg/depth/od ms: only when do_timing and idx!=1
            # =========================================================
            if idx >= 2:
                fps_stats.update(fps_ema)
                steer_stats.update(prev_angle)
                if do_timing:
                    segms_stats.update(seg_ms)
                    depthms_stats.update(depth_ms)
                    odms_stats.update(od_ms)

            if idx % PRINT_EVERY == 0:
                if do_timing:
                    print(
                        f"[{idx:05d}] FPS={fps_ema:5.1f} | steer={prev_angle:+6.2f} | STEER_STATE={steer_state:2s} | "
                        f"STATE={state:5s} | SEG={seg_ms:6.1f}ms DEPTH={depth_ms:6.1f}ms OD={od_ms:6.1f}ms | "
                        f"run_depth={run_depth} tracks={len(tracks)}"
                    )
                else:
                    print(
                        f"[{idx:05d}] FPS={fps_ema:5.1f} | steer={prev_angle:+6.2f} | STEER_STATE={steer_state:2s} | "
                        f"STATE={state:5s} | run_depth={run_depth} tracks={len(tracks)}"
                    )

        cap.release()

        # =========================================================
        # ✅ [ADDED] print summary stats at end
        # =========================================================
        print("\n==================== SUMMARY (exclude first frame) ====================")
        _print_stats_block("FPS(ema)", fps_stats.finalize())
        _print_stats_block("STEER(deg)", steer_stats.finalize())
        _print_stats_block("SEG(ms)", segms_stats.finalize())
        _print_stats_block("DEPTH(ms)", depthms_stats.finalize())
        _print_stats_block("OD(ms)", odms_stats.finalize())
        print("======================================================================\n")

    finally:
        try:
            apply_state_to_motor("STOP", pwm)
            pwm.stop()
        except Exception:
            pass

        try:
            servo_stop(spwm)
            spwm.stop()
        except Exception:
            pass

        GPIO.cleanup()


if __name__ == "__main__":
    main()