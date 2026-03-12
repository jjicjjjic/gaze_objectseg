#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import gc
import argparse
from collections import deque

import cv2
import numpy as np
import zmq
import msgpack
from PIL import Image

from ultralytics import YOLO
from nanosam.utils.predictor import Predictor

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Utils
# -----------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def recv_topic_payload(sub):
    topic = sub.recv_string()
    payload = msgpack.unpackb(sub.recv(), raw=False)

    extra = []
    while sub.get(zmq.RCVMORE):
        extra.append(sub.recv())
    if extra:
        payload["__raw_data__"] = extra
    return topic, payload


def crop_with_pad(img_bgr, cx, cy, roi):
    H, W = img_bgr.shape[:2]
    half = roi // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0
    y1 = y0
    x2 = x0 + roi
    y2 = y0 + roi

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    sx1 = clamp(x1, 0, W)
    sy1 = clamp(y1, 0, H)
    sx2 = clamp(x2, 0, W)
    sy2 = clamp(y2, 0, H)

    crop = img_bgr[sy1:sy2, sx1:sx2]
    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        roi_img = cv2.copyMakeBorder(
            crop,
            top=pad_top, bottom=pad_bottom,
            left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        roi_img = crop

    if roi_img.shape[0] != roi or roi_img.shape[1] != roi:
        roi_img = cv2.resize(roi_img, (roi, roi), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(roi_img), x0, y0


def overlay_mask(full_img, mask_bool, x0, y0, alpha=0.40, color=(0, 255, 0)):
    H, W = full_img.shape[:2]
    mh, mw = mask_bool.shape[:2]

    X1 = max(0, x0)
    Y1 = max(0, y0)
    X2 = min(W, x0 + mw)
    Y2 = min(H, y0 + mh)
    if X1 >= X2 or Y1 >= Y2:
        return

    mx1 = X1 - x0
    my1 = Y1 - y0
    mx2 = mx1 + (X2 - X1)
    my2 = my1 + (Y2 - Y1)

    sub = full_img[Y1:Y2, X1:X2]
    msk = mask_bool[my1:my2, mx1:mx2]
    if msk.size == 0:
        return

    overlay = sub.copy()
    overlay[msk] = color
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, dst=sub)


def nearest_gaze(gaze_buf, t_frame):
    if not gaze_buf:
        return None
    best = min(gaze_buf, key=lambda x: abs(x[0] - t_frame))
    return (best[1], best[2])


def gaze_fixation_ok(gaze_hist, std_th=0.010, min_n=12):
    if len(gaze_hist) < min_n:
        return False, None
    pts = np.array([(x, y) for (_, x, y) in gaze_hist], dtype=np.float32)
    std = pts.std(axis=0)
    if (std[0] < std_th) and (std[1] < std_th):
        mean = pts.mean(axis=0)
        return True, (float(mean[0]), float(mean[1]))
    return False, None


def cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_nanosam_once(enc_path, dec_path, refine_bgr, point_xy):
    predictor = None
    try:
        predictor = Predictor(
            image_encoder_engine=enc_path,
            mask_decoder_engine=dec_path
        )

        refine_rgb = cv2.cvtColor(refine_bgr, cv2.COLOR_BGR2RGB)
        refine_rgb = np.ascontiguousarray(refine_rgb)
        predictor.set_image(Image.fromarray(refine_rgb))

        H, W = refine_bgr.shape[:2]
        px = float(clamp(point_xy[0], 0, W - 1))
        py = float(clamp(point_xy[1], 0, H - 1))

        pts = np.array([[px, py]], dtype=np.float32)
        lbl = np.array([1], dtype=np.float32)

        masks_pred, iou_pred, logits_pred = predictor.predict(pts, lbl)

        if hasattr(masks_pred, "detach"):
            masks_np = masks_pred.detach().float().cpu().numpy()
        else:
            masks_np = np.asarray(masks_pred)

        if hasattr(iou_pred, "detach"):
            iou_np = iou_pred.detach().float().cpu().numpy()
        else:
            iou_np = np.asarray(iou_pred)

        masks_np = np.squeeze(masks_np)
        iou_np = np.squeeze(iou_np)

        if masks_np.ndim == 3:
            best_idx = int(np.argmax(iou_np)) if np.size(iou_np) > 0 else 0
            best_mask = (masks_np[best_idx] > 0.5)
        elif masks_np.ndim == 2:
            best_mask = (masks_np > 0.5)
        else:
            best_mask = None

        del predictor
        cuda_cleanup()
        return best_mask

    except Exception as e:
        print("[NanoSAM][ERR] one-shot refine failed:", e)
        try:
            del predictor
        except Exception:
            pass
        cuda_cleanup()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--req-port", type=int, default=50020)

    ap.add_argument("--model", type=str, default="yolo26n-seg.pt")
    ap.add_argument("--roi", type=int, default=160)
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", type=str, default="0")

    ap.add_argument("--gaze-buf-sec", type=float, default=0.6)
    ap.add_argument("--gaze-radius", type=int, default=120)

    ap.add_argument("--fix-std", type=float, default=0.010)
    ap.add_argument("--fix-minn", type=int, default=12)

    ap.add_argument("--nanosam-encoder", type=str, default="~/pupil/nanosam/data/resnet18_image_encoder.engine")
    ap.add_argument("--nanosam-decoder", type=str, default="~/pupil/nanosam/data/mobile_sam_mask_decoder.engine")
    ap.add_argument("--refine-size", type=int, default=96)
    ap.add_argument("--nanosam-only-on-fix", action="store_true")
    ap.add_argument("--nanosam-cooldown", type=float, default=1.5)
    ap.add_argument("--nanosam-cache-ttl", type=float, default=1.0)
    ap.add_argument("--stable-hit-min", type=int, default=3)

    ap.add_argument("--view", action="store_true")
    ap.add_argument("--max-drain", type=int, default=80)
    ap.add_argument("--conflate", action="store_true")
    args = ap.parse_args()

    if args.imgsz <= 0:
        args.imgsz = args.roi

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://{args.host}:{args.req_port}")

    req.send_string("SUB_PORT")
    sub_port = req.recv_string()
    print("[PUPIL] SUB_PORT =", sub_port)

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
    sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.3d.01")

    sub.setsockopt(zmq.RCVTIMEO, 5)
    if args.conflate:
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.RCVHWM, 1)
    else:
        sub.setsockopt(zmq.RCVHWM, 4)

    print("[YOLO] loading:", args.model)
    model = YOLO(args.model)

    enc_path = os.path.expanduser(args.nanosam_encoder)
    dec_path = os.path.expanduser(args.nanosam_decoder)

    gaze_buf = []
    last_gaze = None
    gaze_hist = deque(maxlen=40)

    ns_cache_mask = None
    ns_cache_x0 = None
    ns_cache_y0 = None
    ns_cache_t = 0.0
    ns_next_allowed_t = 0.0

    stable_hit = 0

    t_fps0 = time.time()
    n_frames = 0

    print("[RUN] press 'q' to quit")
    while True:
        latest_frame_msg = None
        drained = 0
        now_t = None

        while drained < args.max_drain:
            try:
                topic, msg = recv_topic_payload(sub)
            except zmq.Again:
                break
            drained += 1

            if topic.startswith("gaze.3d.01"):
                ts = msg.get("timestamp", None)
                if ts is None:
                    continue
                nx, ny = msg.get("norm_pos", (None, None))
                if nx is None or ny is None:
                    continue
                nx = float(nx)
                ny = float(ny)
                last_gaze = (nx, ny)
                gaze_buf.append((float(ts), nx, ny))
                gaze_hist.append((float(ts), nx, ny))

                t_now = float(ts)
                gaze_buf = [(t, x, y) for (t, x, y) in gaze_buf if (t_now - t) <= args.gaze_buf_sec]

            elif topic == "frame.world":
                latest_frame_msg = msg
                now_t = msg.get("timestamp", None)

        if latest_frame_msg is None:
            continue

        try:
            w = int(latest_frame_msg["width"])
            h = int(latest_frame_msg["height"])
            raw = latest_frame_msg["__raw_data__"][0]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frame = np.ascontiguousarray(frame)
        except Exception as e:
            print("[ERR] frame decode:", e)
            continue

        H, W = frame.shape[:2]

        # ===== 여기부터 gaze 처리 로직은 네 정상 코드 그대로 =====
        gaze = None
        if now_t is not None:
            try:
                gaze = nearest_gaze(gaze_buf, float(now_t))
            except Exception:
                gaze = None
        if gaze is None:
            gaze = last_gaze

        if gaze is None:
            gx, gy = W // 2, H // 2
        else:
            nx, ny = gaze
            gx = int(nx * W)
            gy = int((1.0 - ny) * H)
            gx = clamp(gx, 0, W - 1)
            gy = clamp(gy, 0, H - 1)
        # ===== 여기까지 절대 건드리지 말 것 =====

        roi_img, x0, y0 = crop_with_pad(frame, gx, gy, args.roi)
        gxr = gx - x0
        gyr = gy - y0

        try:
            res = model(
                roi_img,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                half=args.half,
                device=args.device,
                verbose=False,
            )[0]
        except Exception as e:
            print("[ERR] inference:", e)
            res = None
            cuda_cleanup()

        vis = frame.copy()
        cv2.circle(vis, (gx, gy), 7, (0, 0, 255), -1)
        cv2.rectangle(vis, (x0, y0), (x0 + args.roi, y0 + args.roi), (255, 0, 0), 2)

        keep_ids = []
        chosen_bbox = None

        if res is not None and res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.detach().cpu().numpy()
            if 0 <= int(gyr) < masks.shape[1] and 0 <= int(gxr) < masks.shape[2]:
                for i in range(masks.shape[0]):
                    if masks[i, int(gyr), int(gxr)] > 0.5:
                        keep_ids.append(i)

        if not keep_ids and res is not None and res.boxes is not None:
            R = int(args.gaze_radius)
            for i, b in enumerate(res.boxes):
                x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                inside = (x1 <= gxr <= x2) and (y1 <= gyr <= y2)
                near = ((cx - gxr) ** 2 + (cy - gyr) ** 2) <= (R * R)
                if inside or near:
                    keep_ids.append(i)

        if res is not None:
            if res.masks is not None and res.masks.data is not None:
                masks = res.masks.data.detach().cpu().numpy()
                for i in keep_ids:
                    if 0 <= i < masks.shape[0]:
                        overlay_mask(vis, masks[i] > 0.5, x0, y0, alpha=0.40, color=(0, 255, 0))

            if res.boxes is not None and len(res.boxes) > 0:
                for i in keep_ids:
                    if i >= len(res.boxes):
                        continue
                    b = res.boxes[i]
                    xyxy = b.xyxy[0].detach().cpu().numpy()
                    cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0

                    x1, y1, x2, y2 = xyxy
                    if chosen_bbox is None:
                        chosen_bbox = (float(x1), float(y1), float(x2), float(y2))

                    X1 = int(x1 + x0)
                    Y1 = int(y1 + y0)
                    X2 = int(x2 + x0)
                    Y2 = int(y2 + y0)

                    cv2.rectangle(vis, (X1, Y1), (X2, Y2), (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        f"{cls}:{conf:.2f}",
                        (X1, max(0, Y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        fix_ok, _ = gaze_fixation_ok(gaze_hist, std_th=args.fix_std, min_n=args.fix_minn)

        if len(keep_ids) > 0:
            stable_hit += 1
        else:
            stable_hit = 0

        now = time.time()
        if ns_cache_mask is not None and (now - ns_cache_t) <= args.nanosam_cache_ttl:
            overlay_mask(vis, ns_cache_mask, ns_cache_x0, ns_cache_y0, alpha=0.30, color=(0, 0, 255))
            cv2.putText(vis, "NanoSAM refine", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        can_run_ns = (
            chosen_bbox is not None and
            stable_hit >= args.stable_hit_min and
            now >= ns_next_allowed_t and
            ((not args.nanosam_only_on_fix) or fix_ok)
        )

        if can_run_ns:
            x1r, y1r, x2r, y2r = chosen_bbox
            cx_roi = int(round((x1r + x2r) / 2.0))
            cy_roi = int(round((y1r + y2r) / 2.0))
            cx_fr = x0 + cx_roi
            cy_fr = y0 + cy_roi

            refine_img, fx0, fy0 = crop_with_pad(frame, cx_fr, cy_fr, args.refine_size)

            px_ref = gx - fx0
            py_ref = gy - fy0
            if not (0 <= px_ref < args.refine_size and 0 <= py_ref < args.refine_size):
                px_ref = args.refine_size // 2
                py_ref = args.refine_size // 2

            ns_mask = run_nanosam_once(enc_path, dec_path, refine_img, (px_ref, py_ref))
            if ns_mask is not None:
                ns_cache_mask = ns_mask
                ns_cache_x0 = fx0
                ns_cache_y0 = fy0
                ns_cache_t = time.time()

            ns_next_allowed_t = time.time() + float(args.nanosam_cooldown)

        n_frames += 1
        if time.time() - t_fps0 >= 1.0:
            fps = n_frames / (time.time() - t_fps0)
            ns_on = int(ns_cache_mask is not None and (time.time() - ns_cache_t) <= args.nanosam_cache_ttl)
            print(
                f"[FPS] {fps:.1f} | keep={len(keep_ids)} | ROI={args.roi} imgsz={args.imgsz} "
                f"| fix={int(fix_ok)} | ns={ns_on}"
            )
            t_fps0 = time.time()
            n_frames = 0

        if args.view:
            cv2.imshow("Pupil + YOLO26-seg + NanoSAM", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()