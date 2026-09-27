#!/usr/bin/env python3
"""
extract_frames.py
------------------
Pre-extracts 64 directional WebP frames (~5.625 degrees apart) plus a
center/neutral frame from a character-turn video, for use with a
cursor-tracking canvas renderer that NEVER seeks or plays the source
video in the browser.

WHY THIS EXISTS
The source MP4 has a single keyframe (typical of AI-generated video), so
seeking `video.currentTime` in-browser causes severe lag/freezing. Instead
we do all the expensive frame-accurate work once, offline, with OpenCV,
and ship plain images to the browser.

HOW THE SOURCE VIDEO IS STRUCTURED (this file)
192 frames @ 24fps (8s), 1920x1080. The head sweeps a full compass loop
and then swings back through the top to settle on a neutral, camera-facing
pose at the very end. The 8 compass extremes were located by:
  1. Building a fixed-size skin-tone mask inside a static head ROI
     (the body/shoulders never move -- only the head rotates -- so a
     fixed ROI is valid for the whole clip).
  2. Tracking the mask's centroid per frame -> a cheap gaze/turn proxy
     (nose parallax shifts the centroid as the head yaws/pitches).
  3. Using frame-to-frame centroid velocity to find "hold" plateaus,
     then confirming each candidate visually.

Found keyframes (frame index : compass direction):
    0   UP-LEFT
    18  UP
    30  UP-RIGHT
    62  RIGHT
    76  DOWN-RIGHT
    86  DOWN
    106 DOWN-LEFT
    130 LEFT
    148 UP-LEFT  (loop closure, same pose as frame 0)
    186 CENTER   (neutral, looking at camera)

Re-running on a DIFFERENT source video: use --analyze first to dump a
velocity/centroid CSV and a contact sheet, eyeball the 8 holds + neutral
frame, then pass them with --keyframes.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults verified for Character_tracking_cursor_animation_1080p.mp4
# ---------------------------------------------------------------------------
DEFAULT_KEYFRAMES = {
    "UL": 0,
    "U": 18,
    "UR": 30,
    "R": 62,
    "DR": 76,
    "D": 86,
    "DL": 106,
    "L": 130,
    "UL_LOOP": 148,   # closes the loop back to UL, used only for interpolation math
}
DEFAULT_CENTER_FRAME = 186

# Fixed crop (x0, y0, x1, y1) in the ORIGINAL 1920x1080 frame. Confirmed to
# contain the character across every rotation extreme (body/shoulders are
# static; only hair silhouette and head bbox move, and were measured across
# all 8 directions before picking these bounds).
DEFAULT_CROP = (150, 0, 1830, 1080)
DEFAULT_OUT_WIDTH = 760

# Compass angle (degrees, screen space: 0=RIGHT, 90=DOWN, 180=LEFT, 270=UP)
# assigned to each keyframe, unwrapped so the sequence is monotonic across
# the loop. This order matches ascending source-frame order (clockwise).
KEYFRAME_ANGLES_UNWRAPPED = [225, 270, 315, 360, 405, 450, 495, 540, 585]
KEYFRAME_ORDER = ["UL", "U", "UR", "R", "DR", "D", "DL", "L", "UL_LOOP"]


def build_angle_to_frame_mapper(keyframes: dict):
    frames = [keyframes[name] for name in KEYFRAME_ORDER]
    angles = KEYFRAME_ANGLES_UNWRAPPED

    def map_angle_to_frame(theta_deg: float) -> float:
        unwrapped = theta_deg if theta_deg >= 225 else theta_deg + 360
        for i in range(len(angles) - 1):
            a0, a1 = angles[i], angles[i + 1]
            if a0 <= unwrapped <= a1:
                f0, f1 = frames[i], frames[i + 1]
                t = (unwrapped - a0) / (a1 - a0)
                return f0 + t * (f1 - f0)
        return frames[0]

    return map_angle_to_frame


def extract(video_path, out_dir, keyframes, center_frame, crop, out_width):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            sys.exit(f"Could not read frame {idx}")
        return frame

    x0, y0, x1, y1 = crop
    out_h = round(out_width * (y1 - y0) / (x1 - x0))

    def crop_resize(frame):
        c = frame[y0:y1, x0:x1]
        return cv2.resize(c, (out_width, out_h), interpolation=cv2.INTER_LANCZOS4)

    mapper = build_angle_to_frame_mapper(keyframes)

    manifest = {
        "frameWidth": out_width,
        "frameHeight": out_h,
        "frameCount": 64,
        "anglePerFrame": 360 / 64,
        "frames": [],
    }

    print(f"Source: {video_path} ({frame_count} frames)")
    print(f"Output: {out_dir}  ({out_width}x{out_h} per frame)")

    for i in range(64):
        theta = i * 360 / 64
        frac = mapper(theta)
        src_frame_idx = int(round(frac))
        img = crop_resize(read_frame(src_frame_idx))
        out_path = os.path.join(out_dir, f"frame_{i:02d}.webp")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_WEBP_QUALITY, 90])
        manifest["frames"].append({"index": i, "angle": theta, "sourceFrame": src_frame_idx})
        print(f"  frame_{i:02d}.webp  <- source frame {src_frame_idx:3d}  (angle {theta:6.2f} deg)")

    center_img = crop_resize(read_frame(center_frame))
    cv2.imwrite(os.path.join(out_dir, "center.webp"), center_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    print(f"  center.webp <- source frame {center_frame}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    cap.release()
    print("Done.")


def analyze(video_path, out_dir):
    """Dump a per-frame skin-centroid velocity CSV + contact sheet so a NEW
    source video's 8 compass holds + neutral frame can be located by eye."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ref_idx = n - 5
    gray = cv2.cvtColor(frames[ref_idx], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        sys.exit("Could not auto-detect a reference face near the end of the clip; "
                 "pick a neutral frame manually and adjust the ROI.")
    fx, fy, fw, fh = faces[0]
    pad = int(fw * 0.6)
    x0, y0 = max(0, fx - pad), max(0, fy - pad)
    x1, y1 = fx + fw + pad, fy + fh + int(fh * 0.9)

    ref_roi = frames[ref_idx][y0:y1, x0:x1]
    rh, rw = ref_roi.shape[:2]
    patch1 = ref_roi[int(rh * 0.44):int(rh * 0.48), int(rw * 0.18):int(rw * 0.24)].reshape(-1, 3).astype(np.float64)
    patch2 = ref_roi[int(rh * 0.44):int(rh * 0.48), int(rw * 0.76):int(rw * 0.82)].reshape(-1, 3).astype(np.float64)
    skin_mean = np.concatenate([patch1, patch2]).mean(axis=0)
    bg_mean = frames[ref_idx][5:40, 5:40].reshape(-1, 3).astype(np.float64).mean(axis=0)

    csv_rows = ["frame,cx,cy,velocity"]
    prev = None
    for i, frame in enumerate(frames):
        roi = frame[y0:y1, x0:x1].astype(np.float64)
        d_skin = np.linalg.norm(roi - skin_mean, axis=2)
        d_bg = np.linalg.norm(roi - bg_mean, axis=2)
        mask = ((d_skin < 45) & (d_bg > 40)).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            csv_rows.append(f"{i},,,")
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        vel = 0.0 if prev is None else float(np.hypot(cx - prev[0], cy - prev[1]))
        prev = (cx, cy)
        csv_rows.append(f"{i},{cx:.2f},{cy:.2f},{vel:.2f}")

    with open(os.path.join(out_dir, "centroid_velocity.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    # contact sheet every 4th frame, cropped to the same ROI, for visual QA
    step = 4
    thumbs = []
    for i in range(0, len(frames), step):
        c = frames[i][y0:y1, x0:x1]
        c = cv2.resize(c, (180, int(180 * (y1 - y0) / (x1 - x0))))
        cv2.putText(c, str(i), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        thumbs.append(c)
    cols = 8
    rows = (len(thumbs) + cols - 1) // cols
    th, tw = thumbs[0].shape[:2]
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(os.path.join(out_dir, "contact_sheet.png"), sheet)
    print(f"Wrote {out_dir}/centroid_velocity.csv and {out_dir}/contact_sheet.png")
    print("Look for low-velocity plateaus in the CSV, confirm each against the contact sheet,")
    print("then re-run with --keyframes UL=.. U=.. UR=.. R=.. DR=.. D=.. DL=.. L=.. UL_LOOP=.. --center ..")


def parse_keyframes(pairs):
    kf = dict(DEFAULT_KEYFRAMES)
    for pair in pairs or []:
        k, v = pair.split("=")
        kf[k] = int(v)
    return kf


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Path to the source MP4")
    p.add_argument("--out", default="public/frames", help="Output directory (default: public/frames)")
    p.add_argument("--analyze", action="store_true",
                    help="Analysis mode: dump centroid_velocity.csv + contact_sheet.png instead of extracting")
    p.add_argument("--keyframes", nargs="*", default=None,
                    help="Override e.g. UL=0 U=18 UR=30 R=62 DR=76 D=86 DL=106 L=130 UL_LOOP=148")
    p.add_argument("--center", type=int, default=DEFAULT_CENTER_FRAME, help="Neutral/center source frame index")
    p.add_argument("--crop", nargs=4, type=int, default=list(DEFAULT_CROP), metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--width", type=int, default=DEFAULT_OUT_WIDTH, help="Output frame width in px (aspect preserved)")
    args = p.parse_args()

    if args.analyze:
        analyze(args.video, args.out)
    else:
        kf = parse_keyframes(args.keyframes)
        extract(args.video, args.out, kf, args.center, tuple(args.crop), args.width)
