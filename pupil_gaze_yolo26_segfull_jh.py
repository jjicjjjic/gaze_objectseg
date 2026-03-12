#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pupil frame.world (BGR raw) + gaze.3d.01 (norm_pos) -> Ultralytics YOLO26 Seg (ROI, low-latency)

핵심 개선:
  1) 최신 프레임만 처리 (ZMQ CONFLATE + drain loop)  -> 딜레이 최소화
  2) "응시점 포함" mask만 유지 (없으면 gaze 근처 bbox만) -> 산발적 탐지 제거
  3) ROI를 작게(기본 384) + imgsz=ROI로 고정 -> 속도/안정성 향상

실행:
  터미널 A (Pupil Capture):
    cd ~/pupil/pupil_src
    source ~/pupil/Minki_pupilTest/bin/activate
    export DISPLAY=:1
    python main.py capture

  터미널 B (This script):
    cd ~/pupil
    source ~/pupil/Minki_pupilTest/bin/activate
    export DISPLAY=:1
    python pupil_gaze_yolo26_seg_latest_roi.py --view --half

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
            top=pad_top, bottom=pad_bottom,
            left=pad_left, right=pad_right,
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
    Overlay one boolean mask (roi space) onto full_img at position (x0,y0).
    mask_bool shape: (roi, roi)
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
        img, text, (tx1 + 5, ty2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA
    )


# def crop_box_expand(img_bgr, x1, y1, x2, y2, scale=2.2, margin=24):
#     """
#     선택된 ROI bbox를 기준으로 전체 프레임에서 더 큰 정사각 crop 생성.
#     이 crop에서 다시 YOLO-seg를 돌려 '물체 전체 mask'를 얻기 위함.
#     """
#     H, W = img_bgr.shape[:2]

#     bw = max(1.0, float(x2 - x1))
#     bh = max(1.0, float(y2 - y1))
#     cx = 0.5 * (x1 + x2)
#     cy = 0.5 * (y1 + y2)

#     side = int(max(bw, bh) * scale + 2 * margin)
#     side = max(side, 96)

#     rx1 = int(round(cx - side / 2.0))
#     ry1 = int(round(cy - side / 2.0))
#     rx2 = rx1 + side
#     ry2 = ry1 + side

#     pad_left = max(0, -rx1)
#     pad_top = max(0, -ry1)
#     pad_right = max(0, rx2 - W)
#     pad_bottom = max(0, ry2 - H)

#     sx1 = clamp(rx1, 0, W)
#     sy1 = clamp(ry1, 0, H)
#     sx2 = clamp(rx2, 0, W)
#     sy2 = clamp(ry2, 0, H)

#     crop = img_bgr[sy1:sy2, sx1:sx2]
#     if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
#         crop = cv2.copyMakeBorder(
#             crop,
#             top=pad_top, bottom=pad_bottom,
#             left=pad_left, right=pad_right,
#             borderType=cv2.BORDER_CONSTANT,
#             value=(0, 0, 0),
#         )

#     return crop, rx1, ry1

def nearest_gaze(gaze_buf, t_frame):
    """Return gaze (nx, ny) in gaze_buf with timestamp closest to t_frame."""
    if not gaze_buf:
        return None
    # gaze_buf: list of (t, nx, ny)
    best = min(gaze_buf, key=lambda x: abs(x[0] - t_frame))
    return (best[1], best[2])

# def box_iou_xyxy(a, b):
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b

#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)

#     iw = max(0.0, ix2 - ix1)
#     ih = max(0.0, iy2 - iy1)
#     inter = iw * ih

#     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
#     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
#     union = area_a + area_b - inter

#     return 0.0 if union <= 0 else inter / union
# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--req-port", type=int, default=50020)

    ap.add_argument("--model", type=str, default="yolo26n-seg.pt")
    ap.add_argument("--roi", type=int, default=256, help="square ROI size")
    ap.add_argument("--imgsz", type=int, default=256, help="ultralytics imgsz")
    ap.add_argument("--gaze-radius", type=int, default=80, help="fallback radius in ROI pixels (bbox-only case)")
    ap.add_argument("--center-th", type=float, default=80.0, help="fallback center-distance threshold in full-frame pixels")

    ap.add_argument("--conf", type=float, default=0.25)
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

    # --- [MOD 1] 최신 메시지 우선: 딜레이 방지 세팅 ---
    sub.setsockopt(zmq.RCVTIMEO, 5)     # 5ms timeout
    sub.setsockopt(zmq.CONFLATE, 1)     # 최신 1개만 유지(프레임 지연 제거 핵심)
    sub.setsockopt(zmq.RCVHWM, 1)       # 수신 큐 1로 더 강제

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
        out_path = f"seg_roi_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(args.out_fps), (ow, oh))
        print("[VID] saving:", out_path)

    # -----------------------------
    # State: gaze buffer
    # -----------------------------
    gaze_buf = []  # list of (t, nx, ny)
    last_gaze = None  # (nx, ny)

    # -----------------------------
    # FPS logging
    # -----------------------------
    t_fps0 = time.time()
    n_frames = 0

    print("[RUN] press 'q' to quit")
    while True:
        # --- [MOD 2] drain loop: frame/world는 최신 1개만 처리 ---
        latest_frame_msg = None
        drained = 0
        now_t = None

        while drained < args.max_drain:
            try:
                topic, msg = recv_topic_payload(sub)
            except zmq.Again:
                break  # timeout -> stop draining
            drained += 1

            if topic.startswith("gaze.3d.01"):
                # --- [MOD 3] gaze buffer timestamped ---
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

                # prune
                t_now = float(ts)
                gaze_buf = [(t, x, y) for (t, x, y) in gaze_buf if (t_now - t) <= args.gaze_buf_sec]

            elif topic == "frame.world":
                latest_frame_msg = msg
                now_t = msg.get("timestamp", None)

        if latest_frame_msg is None:
            continue  # 아직 프레임 없음

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

        # --- [MOD 4] frame timestamp에 가장 가까운 gaze 사용 (없으면 last / center) ---
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
            # NOTE: Pupil norm_pos는 (0,0)=좌하단 계열인 경우가 많아서 y flip 적용 (너 기존 코드 방식)
            gy = int((1.0 - ny) * H)
            gx = clamp(gx, 0, W - 1)
            gy = clamp(gy, 0, H - 1)

        # # ROI crop around gaze
        # roi_img, x0, y0 = crop_with_pad(frame, gx, gy, roi)

        # # 여기 주목 << ROI 좌표계에서 시선좌표를 다시 받는 이유가 뭐지?
        # # gaze in ROI coords
        # gxr = gx - x0
        # gyr = gy - y0

        # # YOLO seg inference on ROI
        # try:
        #     res = model(
        #         roi_img,
        #         imgsz=args.imgsz,
        #         conf=args.conf,
        #         iou=args.iou,
        #         half=args.half,
        #         device=args.device,
        #         verbose=False,
        #     )[0]
        # except Exception as e:
        #     print("[ERR] inference:", e)
        #     res = None

        # # Prepare visualization frame
        # vis = frame.copy()

        # # draw gaze marker + ROI rectangle
        # cv2.circle(vis, (gx, gy), 7, (0, 0, 255), -1)
        # cv2.rectangle(vis, (x0, y0), (x0 + roi, y0 + roi), (255, 0, 0), 2)

        # best_i = None
        # best_score = -1e18
        # masks = None

        # if res is not None and res.masks is not None and res.masks.data is not None:
        #     masks = res.masks.data.detach().cpu().numpy()

        # if res is not None and res.boxes is not None and len(res.boxes) > 0:
        #     R2 = float(args.gaze_radius) ** 2

        #     for i, b in enumerate(res.boxes):
        #         x1r, y1r, x2r, y2r = b.xyxy[0].detach().cpu().numpy()
        #         cx = 0.5 * (x1r + x2r)
        #         cy = 0.5 * (y1r + y2r)

        #         inside = (x1r <= gxr <= x2r) and (y1r <= gyr <= y2r)
        #         d2 = (cx - gxr) ** 2 + (cy - gyr) ** 2

        #         mask_hit = False
        #         if masks is not None and i < masks.shape[0]:
        #             if 0 <= int(gyr) < masks.shape[1] and 0 <= int(gxr) < masks.shape[2]:
        #                 mask_hit = (masks[i, int(gyr), int(gxr)] > 0.5)

        #         # 우선순위:
        #         # 1) gaze 점이 mask 안에 있으면 최고
        #         # 2) 아니면 gaze가 bbox 안에 있는 후보
        #         # 3) 아니면 gaze 반경 내에서 중심이 가장 가까운 후보
        #         if mask_hit:
        #             score = 1e9 - d2
        #         elif inside:
        #             score = 1e6 - d2
        #         elif d2 <= R2:
        #             score = -d2
        #         else:
        #             continue

        #         if score > best_score:
        #             best_score = score
        #             best_i = i

        # if res is not None and best_i is not None:
        #     # 같은 인덱스 mask
        #     if masks is not None and best_i < masks.shape[0]:
        #         overlay_mask(vis, masks[best_i] > 0.5, x0, y0, alpha=0.40)

        #     # 같은 인덱스 bbox
        #     if res.boxes is not None and best_i < len(res.boxes):
        #         b = res.boxes[best_i]
        #         xyxy = b.xyxy[0].detach().cpu().numpy()
        #         cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
        #         conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0

        #         x1r, y1r, x2r, y2r = xyxy
        #         x1 = int(x1r + x0)
        #         y1 = int(y1r + y0)
        #         x2 = int(x2r + x0)
        #         y2 = int(y2r + y0)

        #         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        #         if isinstance(names, dict):
        #             cls_name = names.get(cls, str(cls))
        #         else:
        #             cls_name = names[cls] if 0 <= cls < len(names) else str(cls)

        #         draw_label_box(vis, x1, y1, f"{cls_name} {conf:.2f}")
        
        
                # --------------------------------------------------
        # Full-frame YOLO-seg inference
        # --------------------------------------------------
        try:
            res = model(
                frame,
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

        # Prepare visualization frame
        vis = frame.copy()

        # draw gaze marker only
        cv2.circle(vis, (gx, gy), 7, (0, 0, 255), -1)

        best_i = None
        best_score = -1e18
        masks = None

        if res is not None and res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.detach().cpu().numpy()

        center_th2 = float(args.center_th) ** 2

        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            for i, b in enumerate(res.boxes):
                x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                inside = (x1 <= gx <= x2) and (y1 <= gy <= y2)
                d2 = (cx - gx) ** 2 + (cy - gy) ** 2

                mask_hit = False
                mask_area = None
                if masks is not None and i < masks.shape[0]:
                    mask_i = masks[i]

                    # mask가 frame 크기와 다르면 resize 후 gaze hit 판정
                    if mask_i.shape[:2] != frame.shape[:2]:
                        mask_i = cv2.resize(
                            mask_i.astype(np.float32),
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )

                    mask_bool_i = mask_i > 0.5
                    mask_area = float(mask_bool_i.sum())

                    min_mask_area = 300
                    if mask_area < min_mask_area:
                        continue

                    # gaze 점이 실제 mask 안에 있는지 판정
                    if 0 <= int(gy) < frame.shape[0] and 0 <= int(gx) < frame.shape[1]:
                        mask_hit = (mask_i[int(gy), int(gx)] > 0.5)

                else:
                    mask_area = float((x2 - x1) * (y2 - y1))  # fallback

                conf_i = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0


                # 우선순위:
                # 1) gaze 점이 full-frame mask 안에 있으면 최고
                # 2) 아니면 gaze가 full-frame bbox 안에 있는 후보
                # 3) 아니면 center_th 이내에서만 중심이 가장 가까운 후보
                if mask_hit:
                # 작은 물체 우선: area가 작을수록 score 증가
                    score = 1e9 - 0.01 * mask_area - d2 + 10.0 * conf_i
                elif inside:
                    score = 1e6 - 0.001 * mask_area - d2 + 5.0 * conf_i
                elif d2 <= center_th2:
                    score = -d2 - 0.0005 * mask_area + conf_i
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_i = i

        if res is not None and best_i is not None:
            # full-frame mask overlay
            if masks is not None and best_i < masks.shape[0]:
                mask = masks[best_i]

                # mask 해상도가 frame과 다르면 frame 크기로 맞춤
                if mask.shape[:2] != vis.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (vis.shape[1], vis.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )

                mask_bool = mask > 0.5

                color_layer = vis.copy()
                color_layer[mask_bool] = (0, 255, 255)
                vis = cv2.addWeighted(color_layer, 0.40, vis, 0.60, 0)

            # full-frame bbox
            if res.boxes is not None and best_i < len(res.boxes):
                b = res.boxes[best_i]
                xyxy = b.xyxy[0].detach().cpu().numpy()
                cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0

                x1, y1, x2, y2 = map(int, xyxy.tolist())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if isinstance(names, dict):
                    cls_name = names.get(cls, str(cls))
                else:
                    cls_name = names[cls] if 0 <= cls < len(names) else str(cls)

                draw_label_box(vis, x1, y1, f"{cls_name} {conf:.2f}")

        # FPS log
        n_frames += 1
        if time.time() - t_fps0 >= 1.0:
            fps = n_frames / (time.time() - t_fps0)
            print(f"[FPS] {fps:.1f} | hit={int(best_i is not None)} | FULL imgsz={args.imgsz} center_th={args.center_th}")   
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