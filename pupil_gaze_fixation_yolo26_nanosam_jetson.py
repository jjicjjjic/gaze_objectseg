#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import gc
import argparse

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


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def round_up_to_stride(x, stride=32):
    return int(math.ceil(float(x) / stride) * stride)


def recv_topic_payload(sub):
    topic = sub.recv_string()
    payload = msgpack.unpackb(sub.recv(), raw=False)
    extra = []
    while sub.get(zmq.RCVMORE):
        extra.append(sub.recv())
    if extra:
        payload["__raw_data__"] = extra
    return topic, payload


def nearest_gaze(gaze_buf, t_frame):
    if not gaze_buf:
        return None
    best = min(gaze_buf, key=lambda x: abs(x[0] - t_frame))
    return (best[1], best[2])


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
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    if crop.shape[0] != roi or crop.shape[1] != roi:
        crop = cv2.resize(crop, (roi, roi), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(crop), x0, y0


def crop_xyxy_with_pad(img_bgr, x1, y1, x2, y2):
    H, W = img_bgr.shape[:2]

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
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return np.ascontiguousarray(crop), x1, y1


def overlay_mask_roi(full_img, mask_bool, x0, y0, alpha=0.35, color=(0, 255, 0)):
    if mask_bool is None:
        return

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


def draw_label_box(img, x1, y1, text, box_color=(0, 255, 255), text_color=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx1 = x1
    ty2 = max(th + 8, y1)
    ty1 = ty2 - th - 8
    tx2 = tx1 + tw + 10

    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), box_color, -1)
    cv2.putText(
        img, text, (tx1 + 5, ty2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA
    )


def cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def pick_instance_from_roi(res, px, py, radius):
    if res is None:
        return None

    if res.masks is not None and res.masks.data is not None:
        masks = res.masks.data.detach().cpu().numpy()
        if 0 <= int(py) < masks.shape[1] and 0 <= int(px) < masks.shape[2]:
            for i in range(masks.shape[0]):
                if masks[i, int(py), int(px)] > 0.5:
                    return i

    if res.boxes is not None:
        best_i = None
        best_d2 = 1e18
        R2 = float(radius) ** 2
        for i, b in enumerate(res.boxes):
            x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            inside = (x1 <= px <= x2) and (y1 <= py <= y2)
            d2 = (cx - px) ** 2 + (cy - py) ** 2
            if inside or d2 <= R2:
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
        return best_i

    return None


def build_object_roi_from_seed_bbox(seed_bbox_roi, seed_x0, seed_y0, expand_scale=2.2, margin=24):
    x1r, y1r, x2r, y2r = seed_bbox_roi
    x1 = seed_x0 + x1r
    y1 = seed_y0 + y1r
    x2 = seed_x0 + x2r
    y2 = seed_y0 + y2r

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    side = int(max(bw, bh) * expand_scale + 2 * margin)
    side = max(side, 96)

    rx1 = int(round(cx - side / 2.0))
    ry1 = int(round(cy - side / 2.0))
    rx2 = rx1 + side
    ry2 = ry1 + side
    return rx1, ry1, rx2, ry2


def point_in_bbox(x, y, bbox, margin=0):
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def clamp_point_to_bbox(x, y, bbox, margin=0):
    x1, y1, x2, y2 = bbox
    return (
        clamp(int(x), int(x1 - margin), int(x2 + margin)),
        clamp(int(y), int(y1 - margin), int(y2 + margin)),
    )


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

        masks_pred, iou_pred, _ = predictor.predict(pts, lbl)

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

    except Exception:
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
    ap.add_argument("--seed-roi", type=int, default=220)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", type=str, default="0")

    ap.add_argument("--gaze-buf-sec", type=float, default=0.6)
    ap.add_argument("--gaze-radius", type=int, default=120)

    ap.add_argument("--lock-radius", type=float, default=60.0)
    ap.add_argument("--lock-duration", type=float, default=1.0)
    ap.add_argument("--unlock-radius", type=float, default=100.0)
    ap.add_argument("--unlock-duration", type=float, default=0.8)

    ap.add_argument("--object-expand-scale", type=float, default=2.2)
    ap.add_argument("--object-margin", type=int, default=24)
    ap.add_argument("--selected-keep-margin", type=int, default=36)
    ap.add_argument("--selected-timeout", type=float, default=1.0)

    ap.add_argument("--nanosam-encoder", type=str, default="~/pupil/nanosam/data/resnet18_image_encoder.engine")
    ap.add_argument("--nanosam-decoder", type=str, default="~/pupil/nanosam/data/mobile_sam_mask_decoder.engine")
    ap.add_argument("--refine-size", type=int, default=96)
    ap.add_argument("--nanosam-cache-ttl", type=float, default=10.0)

    ap.add_argument("--view", action="store_true")
    ap.add_argument("--max-drain", type=int, default=80)
    ap.add_argument("--conflate", action="store_true")
    args = ap.parse_args()

    args.imgsz = round_up_to_stride(args.imgsz, 32)

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://{args.host}:{args.req_port}")

    req.send_string("SUB_PORT")
    sub_port = req.recv_string()

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

    model = YOLO(args.model)
    names = model.names if hasattr(model, "names") else {}

    enc_path = os.path.expanduser(args.nanosam_encoder)
    dec_path = os.path.expanduser(args.nanosam_decoder)

    gaze_buf = []
    last_gaze = None

    fix_anchor_x = None
    fix_anchor_y = None
    fix_start_t = None
    lock_active = False
    lock_cx = None
    lock_cy = None
    outside_start_t = None
    lock_just_confirmed = False

    selected_bbox_global = None
    selected_mask_roi = None
    selected_roi_x0 = None
    selected_roi_y0 = None
    selected_label = None
    selected_conf = None
    selected_last_seen_t = 0.0

    last_logged_label = None
    last_logged_t = 0.0

    ns_cache_mask = None
    ns_cache_x0 = None
    ns_cache_y0 = None
    ns_cache_t = 0.0

    t_fps0 = time.time()
    n_frames = 0

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
        except Exception:
            continue

        H, W = frame.shape[:2]

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

        now = time.time()
        lock_just_confirmed = False

        # anchor + dwell fixation
        if not lock_active:
            if fix_anchor_x is None or fix_anchor_y is None:
                fix_anchor_x = gx
                fix_anchor_y = gy
                fix_start_t = now
            else:
                d = math.hypot(gx - fix_anchor_x, gy - fix_anchor_y)
                if d <= args.lock_radius:
                    if (now - fix_start_t) >= args.lock_duration:
                        lock_active = True
                        lock_cx = int(fix_anchor_x)
                        lock_cy = int(fix_anchor_y)
                        outside_start_t = None
                        lock_just_confirmed = True
                else:
                    fix_anchor_x = gx
                    fix_anchor_y = gy
                    fix_start_t = now
        else:
            d_unlock = math.hypot(gx - lock_cx, gy - lock_cy)
            if d_unlock <= args.unlock_radius:
                outside_start_t = None
            else:
                if outside_start_t is None:
                    outside_start_t = now
                elif (now - outside_start_t) >= args.unlock_duration:
                    lock_active = False
                    lock_cx = None
                    lock_cy = None
                    outside_start_t = None

                    fix_anchor_x = gx
                    fix_anchor_y = gy
                    fix_start_t = now

                    selected_bbox_global = None
                    selected_mask_roi = None
                    selected_roi_x0 = None
                    selected_roi_y0 = None
                    selected_label = None
                    selected_conf = None

                    ns_cache_mask = None
                    ns_cache_x0 = None
                    ns_cache_y0 = None
                    ns_cache_t = 0.0

        vis = frame.copy()

        status_text = "LOCKED" if lock_active else "WAIT_FIX"
        cv2.putText(
            vis,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not lock_active and fix_anchor_x is not None and fix_anchor_y is not None:
            cv2.circle(vis, (int(fix_anchor_x), int(fix_anchor_y)), 6, (255, 255, 0), 2)

        cv2.circle(vis, (gx, gy), 7, (0, 0, 255), -1)

        seed_cx = lock_cx if lock_active else gx
        seed_cy = lock_cy if lock_active else gy
        seed_roi_img, sx0, sy0 = crop_with_pad(frame, seed_cx, seed_cy, args.seed_roi)
        cv2.rectangle(vis, (sx0, sy0), (sx0 + args.seed_roi, sy0 + args.seed_roi), (255, 0, 0), 2)

        if lock_just_confirmed:
            gxr = gx - sx0
            gyr = gy - sy0

            # 이전 결과 초기화
            selected_bbox_global = None
            selected_mask_roi = None
            selected_roi_x0 = None
            selected_roi_y0 = None
            selected_label = None
            selected_conf = None

            ns_cache_mask = None
            ns_cache_x0 = None
            ns_cache_y0 = None
            ns_cache_t = 0.0

            try:
                seed_res = model(
                    seed_roi_img,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    half=args.half,
                    device=args.device,
                    verbose=False,
                )[0]
            except Exception as e:
                seed_res = None
                cuda_cleanup()
                print(f"[ERR] seed yolo: {e}")

            chosen_seed_id = pick_instance_from_roi(seed_res, gxr, gyr, args.gaze_radius)

            if chosen_seed_id is not None and seed_res is not None and seed_res.boxes is not None and chosen_seed_id < len(seed_res.boxes):
                seed_box = seed_res.boxes[chosen_seed_id]
                x1r, y1r, x2r, y2r = seed_box.xyxy[0].detach().cpu().numpy()

                ox1, oy1, ox2, oy2 = build_object_roi_from_seed_bbox(
                    seed_bbox_roi=(x1r, y1r, x2r, y2r),
                    seed_x0=sx0,
                    seed_y0=sy0,
                    expand_scale=args.object_expand_scale,
                    margin=args.object_margin
                )

                object_roi_img, object_roi_x0, object_roi_y0 = crop_xyxy_with_pad(frame, ox1, oy1, ox2, oy2)
                ogx = gx - object_roi_x0
                ogy = gy - object_roi_y0

                try:
                    obj_res = model(
                        object_roi_img,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        half=args.half,
                        device=args.device,
                        verbose=False,
                    )[0]
                except Exception as e:
                    obj_res = None
                    cuda_cleanup()
                    print(f"[ERR] object yolo: {e}")

                chosen_obj_id = pick_instance_from_roi(obj_res, ogx, ogy, args.gaze_radius)

                if chosen_obj_id is not None and obj_res is not None:
                    if obj_res.masks is not None and obj_res.masks.data is not None:
                        masks = obj_res.masks.data.detach().cpu().numpy()
                        if 0 <= chosen_obj_id < masks.shape[0]:
                            selected_mask_roi = (masks[chosen_obj_id] > 0.5)
                            selected_roi_x0 = object_roi_x0
                            selected_roi_y0 = object_roi_y0

                    if obj_res.boxes is not None and chosen_obj_id < len(obj_res.boxes):
                        b = obj_res.boxes[chosen_obj_id]
                        x1o, y1o, x2o, y2o = b.xyxy[0].detach().cpu().numpy()
                        selected_bbox_global = (
                            int(x1o + object_roi_x0),
                            int(y1o + object_roi_y0),
                            int(x2o + object_roi_x0),
                            int(y2o + object_roi_y0),
                        )
                        cls_id = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                        selected_conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                        selected_label = names.get(cls_id, str(cls_id))
                        selected_last_seen_t = now

                        # dwell 시점 ROI local nanosam 1회
                        local_cx, local_cy = clamp_point_to_bbox(
                            gx, gy, selected_bbox_global, margin=args.selected_keep_margin
                        )
                        refine_img, fx0, fy0 = crop_with_pad(frame, local_cx, local_cy, args.refine_size)
                        px_ref = local_cx - fx0
                        py_ref = local_cy - fy0

                        ns_mask = run_nanosam_once(enc_path, dec_path, refine_img, (px_ref, py_ref))
                        if ns_mask is not None:
                            ns_cache_mask = ns_mask
                            ns_cache_x0 = fx0
                            ns_cache_y0 = fy0
                            ns_cache_t = now

                        print(f"[SNAP] {selected_label} ({selected_conf:.2f})")
                else:
                    print("[SNAP] object yolo found nothing")
            else:
                print("[SNAP] seed yolo found nothing")

        selected_alive = selected_bbox_global is not None
        if selected_alive and lock_active:
            if point_in_bbox(gx, gy, selected_bbox_global, margin=args.selected_keep_margin):
                selected_last_seen_t = now
            elif (now - selected_last_seen_t) > args.selected_timeout:
                selected_alive = False
                selected_bbox_global = None
                selected_mask_roi = None
                selected_roi_x0 = None
                selected_roi_y0 = None
                selected_label = None
                selected_conf = None
                ns_cache_mask = None
                ns_cache_x0 = None
                ns_cache_y0 = None
                ns_cache_t = 0.0

        if selected_alive and selected_mask_roi is not None:
            overlay_mask_roi(vis, selected_mask_roi, selected_roi_x0, selected_roi_y0, alpha=0.35, color=(0, 255, 0))

        if selected_alive and selected_bbox_global is not None:
            x1, y1, x2, y2 = selected_bbox_global
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if selected_label is not None:
                draw_label_box(vis, x1, y1, f"{selected_label} {selected_conf:.2f}")
                if (selected_label != last_logged_label) or ((now - last_logged_t) > 1.0):
                    print(f"[OBJ] {selected_label} ({selected_conf:.2f})")
                    last_logged_label = selected_label
                    last_logged_t = now

        if selected_alive and ns_cache_mask is not None and (now - ns_cache_t) <= args.nanosam_cache_ttl:
            overlay_mask_roi(vis, ns_cache_mask, ns_cache_x0, ns_cache_y0, alpha=0.35, color=(0, 0, 255))

        n_frames += 1
        if time.time() - t_fps0 >= 1.0:
            fps = n_frames / (time.time() - t_fps0)
            keep = 1 if selected_alive else 0
            ns_on = int(selected_alive and ns_cache_mask is not None and (time.time() - ns_cache_t) <= args.nanosam_cache_ttl)
            print(f"[FPS] {fps:.1f} | lock={int(lock_active)} | keep={keep} | imgsz={args.imgsz} | ns={ns_on}")
            t_fps0 = time.time()
            n_frames = 0

        if args.view:
            cv2.imshow("Pupil + fixation-triggered YOLO/NanoSAM", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()