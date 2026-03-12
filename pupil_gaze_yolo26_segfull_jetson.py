#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pupil frame.world (BGR raw) + gaze.3d.01 (norm_pos)
-> Ultralytics YOLO26 Seg (ROI select + full-frame refine)

핵심:
  1) 최신 프레임만 처리 (ZMQ CONFLATE + drain loop)
  2) ROI는 "시선 기반 관심 물체 선택"용
  3) 선택된 물체는 full-frame에서 같은 물체 전체 bbox/mask로 복원
  4) bbox와 mask는 항상 같은 최종 인스턴스에서 같이 사용

종료: 'q'
"""

import argparse
import time
import datetime

import cv2
import numpy as np
import zmq
import msgpack

from ultralytics import YOLO


# -----------------------------
# Utils
# -----------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def recv_topic_payload(sub):
    """Receive multipart: topic(str) + msgpack(dict) + raw parts (bytes)."""
    topic = sub.recv_string()
    payload = msgpack.unpackb(sub.recv(), raw=False)

    extra = []
    while sub.get(zmq.RCVMORE):
        extra.append(sub.recv())
    if extra:
        payload["__raw_data__"] = extra
    return topic, payload


def crop_with_pad(img_bgr, cx, cy, roi):
    """
    Crop square ROI centered at (cx, cy). Pads with black if near border.
    Returns:
      roi_img: (roi, roi, 3)
      x0, y0: ROI top-left in original coords (can be negative)
    """
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
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        roi_img = crop

    if roi_img.shape[0] != roi or roi_img.shape[1] != roi:
        roi_img = cv2.resize(roi_img, (roi, roi), interpolation=cv2.INTER_LINEAR)

    return roi_img, x0, y0


def overlay_mask(full_img, mask_bool, x0, y0, alpha=0.40):
    """
    Overlay one boolean mask onto full_img at position (x0, y0).
    mask_bool shape: (h, w)
    """
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
    overlay[msk] = (0, 255, 0)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, dst=sub)


def draw_label_box(img, x1, y1, text, box_color=(0, 255, 255), text_color=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx1 = int(x1)
    ty2 = max(th + 8, int(y1))
    ty1 = ty2 - th - 8
    tx2 = tx1 + tw + 10

    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), box_color, -1)
    cv2.putText(
        img,
        text,
        (tx1 + 5, ty2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )


def nearest_gaze(gaze_buf, t_frame):
    """Return gaze (nx, ny) in gaze_buf with timestamp closest to t_frame."""
    if not gaze_buf:
        return None
    best = min(gaze_buf, key=lambda x: abs(x[0] - t_frame))
    return (best[1], best[2])


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


def box_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def choose_fullframe_instance(full_res, seed_cls, seed_box, gaze_xy_full):
    """
    Full-frame 2차 추론 결과에서 ROI로 고른 seed 물체와 가장 같은 인스턴스를 선택.

    보수적 규칙:
      1) 같은 class 우선
      2) gaze 또는 seed center를 포함하는 후보 우선
      3) seed보다 약간 큰 후보 우선
      4) 너무 큰 후보는 패널티
      5) IoU는 보조 점수
      6) 최종 bbox와 mask는 반드시 같은 인덱스 사용
    """
    if full_res is None or full_res.boxes is None or len(full_res.boxes) == 0:
        return None

    full_masks = None
    if full_res.masks is not None and full_res.masks.data is not None:
        full_masks = full_res.masks.data.detach().cpu().numpy()

    sx1, sy1, sx2, sy2 = seed_box
    scx = 0.5 * (sx1 + sx2)
    scy = 0.5 * (sy1 + sy2)
    gx, gy = gaze_xy_full

    seed_area = max(1.0, box_area_xyxy(seed_box))

    best_j = None
    best_score = -1e18

    for j, bb in enumerate(full_res.boxes):
        cls_j = int(bb.cls[0].item()) if hasattr(bb, "cls") else -1
        if cls_j != seed_cls:
            continue

        xx1, yy1, xx2, yy2 = bb.xyxy[0].detach().cpu().numpy()
        cand = (float(xx1), float(yy1), float(xx2), float(yy2))

        area = max(1.0, box_area_xyxy(cand))
        iou_val = box_iou_xyxy(seed_box, cand)
        area_ratio = area / seed_area

        contains_seed_center = (cand[0] <= scx <= cand[2]) and (cand[1] <= scy <= cand[3])
        contains_gaze = (cand[0] <= gx <= cand[2]) and (cand[1] <= gy <= cand[3])

        mask_hit = False
        if full_masks is not None and j < full_masks.shape[0]:
            mx = int(round(gx))
            my = int(round(gy))
            if 0 <= my < full_masks.shape[1] and 0 <= mx < full_masks.shape[2]:
                mask_hit = (full_masks[j, my, mx] > 0.5)

        score = 0.0

        # 같은 물체를 보수적으로 고르기 위한 우선순위
        if mask_hit:
            score += 1e7
        if contains_gaze:
            score += 1e6
        if contains_seed_center:
            score += 1e5

        # seed 박스보다 조금 큰 후보를 선호
        if area_ratio >= 1.05:
            score += 1e4
        else:
            score -= 5e4

        # 너무 거대한 후보는 패널티
        if area_ratio > 12.0:
            score -= 2e5
        elif area_ratio > 8.0:
            score -= 5e4

        # seed와의 overlap은 보조 점수
        score += 2e4 * iou_val

        # 같은 조건이면 더 작은 box 선호
        score -= 1e-3 * area

        if score > best_score:
            best_score = score
            best_j = j

    if best_j is not None:
        return best_j

    # fallback: class 무시하고 IoU 최대
    best_iou = -1.0
    best_j = None
    for j, bb in enumerate(full_res.boxes):
        xx1, yy1, xx2, yy2 = bb.xyxy[0].detach().cpu().numpy()
        cand = (float(xx1), float(yy1), float(xx2), float(yy2))
        iou_val = box_iou_xyxy(seed_box, cand)
        if iou_val > best_iou:
            best_iou = iou_val
            best_j = j

    return best_j


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--req-port", type=int, default=50020)

    ap.add_argument("--model", type=str, default="yolo26n-seg.pt")
    ap.add_argument("--roi", type=int, default=256, help="square ROI size")
    ap.add_argument("--imgsz", type=int, default=256, help="ultralytics imgsz for ROI pass")
    ap.add_argument("--full-imgsz", type=int, default=640, help="ultralytics imgsz for full-frame refine")
    ap.add_argument("--gaze-radius", type=int, default=80, help="fallback radius in ROI pixels (bbox-only case)")

    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", type=str, default="0", help="'0' or 'cpu'")

    ap.add_argument("--gaze-buf-sec", type=float, default=0.6, help="gaze buffer seconds")

    ap.add_argument("--view", action="store_true")
    ap.add_argument("--save-vid", action="store_true")
    ap.add_argument("--out-fps", type=float, default=15.0)
    ap.add_argument("--out-size", type=str, default="1280x720")

    ap.add_argument("--max-drain", type=int, default=80, help="how many msgs to drain per loop (latest-only)")
    args = ap.parse_args()

    roi = int(args.roi)
    if args.imgsz is None or args.imgsz <= 0:
        args.imgsz = roi

    # -----------------------------
    # ZMQ connect
    # -----------------------------
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

    # 최신 메시지 우선: 딜레이 방지
    sub.setsockopt(zmq.RCVTIMEO, 5)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)

    # -----------------------------
    # Model
    # -----------------------------
    print("[YOLO] loading:", args.model)
    model = YOLO(args.model)
    names = model.names if hasattr(model, "names") else {}

    # -----------------------------
    # Video writer (optional)
    # -----------------------------
    writer = None
    if args.save_vid:
        ow, oh = map(int, args.out_size.lower().split("x"))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"seg_roi_fullrefine_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(args.out_fps), (ow, oh))
        print("[VID] saving:", out_path)

    # -----------------------------
    # State: gaze buffer
    # -----------------------------
    gaze_buf = []
    last_gaze = None

    # -----------------------------
    # FPS logging
    # -----------------------------
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

                t_now = float(ts)
                gaze_buf = [(t, x, y) for (t, x, y) in gaze_buf if (t_now - t) <= args.gaze_buf_sec]

            elif topic == "frame.world":
                latest_frame_msg = msg
                now_t = msg.get("timestamp", None)

        if latest_frame_msg is None:
            continue

        msg = latest_frame_msg

        # decode frame.world BGR raw
        try:
            w = int(msg["width"])
            h = int(msg["height"])
            raw = msg["__raw_data__"][0]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        except Exception as e:
            print("[ERR] frame decode:", e)
            continue

        H, W = frame.shape[:2]

        # frame timestamp와 가장 가까운 gaze 사용
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
            gy = int((1.0 - ny) * H)  # Pupil y flip
            gx = clamp(gx, 0, W - 1)
            gy = clamp(gy, 0, H - 1)

        # ROI crop around gaze
        roi_img, x0, y0 = crop_with_pad(frame, gx, gy, roi)

        # gaze in ROI coords
        gxr = gx - x0
        gyr = gy - y0

        # 1차 ROI inference
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
            print("[ERR] ROI inference:", e)
            res = None

        vis = frame.copy()

        # gaze marker + ROI rectangle
        cv2.circle(vis, (gx, gy), 7, (0, 0, 255), -1)
        cv2.rectangle(vis, (x0, y0), (x0 + roi, y0 + roi), (255, 0, 0), 2)

        best_i = None
        best_score = -1e18
        masks = None

        if res is not None and res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.detach().cpu().numpy()

        # ROI 내에서 시선과 가장 관련 있는 인스턴스 1개 선택
        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            R2 = float(args.gaze_radius) ** 2

            for i, b in enumerate(res.boxes):
                x1r, y1r, x2r, y2r = b.xyxy[0].detach().cpu().numpy()
                cx = 0.5 * (x1r + x2r)
                cy = 0.5 * (y1r + y2r)

                inside = (x1r <= gxr <= x2r) and (y1r <= gyr <= y2r)
                d2 = (cx - gxr) ** 2 + (cy - gyr) ** 2

                mask_hit = False
                if masks is not None and i < masks.shape[0]:
                    if 0 <= int(gyr) < masks.shape[1] and 0 <= int(gxr) < masks.shape[2]:
                        mask_hit = (masks[i, int(gyr), int(gxr)] > 0.5)

                if mask_hit:
                    score = 1e9 - d2
                elif inside:
                    score = 1e6 - d2
                elif d2 <= R2:
                    score = -d2
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_i = i

        # 2차 full-frame refine
        if res is not None and best_i is not None and res.boxes is not None and best_i < len(res.boxes):
            b = res.boxes[best_i]
            xyxy = b.xyxy[0].detach().cpu().numpy()
            cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0

            x1r, y1r, x2r, y2r = xyxy
            seed_box = (
                float(x1r + x0),
                float(y1r + y0),
                float(x2r + x0),
                float(y2r + y0),
            )

            # fallback은 ROI 결과
            final_box = seed_box
            final_mask = masks[best_i] > 0.5 if (masks is not None and best_i < masks.shape[0]) else None
            final_mask_x0 = x0
            final_mask_y0 = y0
            final_cls = cls
            final_conf = conf

            try:
                full_res = model(
                    frame,
                    imgsz=args.full_imgsz,
                    conf=max(args.conf, 0.30),
                    iou=args.iou,
                    half=args.half,
                    device=args.device,
                    verbose=False,
                )[0]
            except Exception as e:
                print("[ERR] full-frame refine:", e)
                full_res = None

            full_i = choose_fullframe_instance(full_res, cls, seed_box, (gx, gy))

            if full_i is not None and full_res is not None and full_res.boxes is not None and full_i < len(full_res.boxes):
                fb = full_res.boxes[full_i]
                fxyxy = fb.xyxy[0].detach().cpu().numpy()
                final_box = (
                    float(fxyxy[0]),
                    float(fxyxy[1]),
                    float(fxyxy[2]),
                    float(fxyxy[3]),
                )
                final_cls = int(fb.cls[0].item()) if hasattr(fb, "cls") else cls
                final_conf = float(fb.conf[0].item()) if hasattr(fb, "conf") else conf

                if full_res.masks is not None and full_res.masks.data is not None and full_i < full_res.masks.data.shape[0]:
                    full_masks = full_res.masks.data.detach().cpu().numpy()
                    final_mask = full_masks[full_i] > 0.5
                    final_mask_x0 = 0
                    final_mask_y0 = 0

            if final_mask is not None:
                overlay_mask(vis, final_mask, final_mask_x0, final_mask_y0, alpha=0.40)

            x1, y1, x2, y2 = map(int, final_box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if isinstance(names, dict):
                cls_name = names.get(final_cls, str(final_cls))
            else:
                cls_name = names[final_cls] if 0 <= final_cls < len(names) else str(final_cls)

            draw_label_box(vis, x1, y1, f"{cls_name} {final_conf:.2f}")

        # FPS log
        n_frames += 1
        if time.time() - t_fps0 >= 1.0:
            fps = n_frames / (time.time() - t_fps0)
            print(
                f"[FPS] {fps:.1f} | hit={int(best_i is not None)} | "
                f"ROI={roi} roi_imgsz={args.imgsz} full_imgsz={args.full_imgsz}"
            )
            t_fps0 = time.time()
            n_frames = 0

        # show/save
        if args.view:
            cv2.imshow("Pupil + YOLO26-seg (Latest ROI)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        if writer is not None:
            ow, oh = map(int, args.out_size.lower().split("x"))
            writer.write(cv2.resize(vis, (ow, oh)))

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()