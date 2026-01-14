#!/usr/bin/env python3
"""
Standalone script to test license plate keypoint alignment.

This script:
1. Runs inference using DeepStream to detect plates and extract keypoints
2. Saves coordinates to a JSON file
3. Then separately reads the video with OpenCV and warps plates

Usage:
    python3 entry/debug_plate_warp.py data/videos/plate2.mp4
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_plate(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective warp plate region to rectangle."""
    points = order_points(points.astype(np.float32))

    width_a = np.linalg.norm(points[0] - points[1])
    width_b = np.linalg.norm(points[2] - points[3])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(points[0] - points[3])
    height_b = np.linalg.norm(points[1] - points[2])
    max_height = int(max(height_a, height_b))

    if max_width < 10 or max_height < 10:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))

    return warped


def main():
    parser = argparse.ArgumentParser(description="Test plate keypoint alignment")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--keypoints", help="JSON file with keypoints (optional)")
    parser.add_argument("--output-dir", default="data/license_plate/output",
                        help="Output directory for warped plates")
    args = parser.parse_args()

    # Hardcoded keypoints from DeepStream detection (from debug output)
    # Format: frame_number -> list of plates with keypoints
    # These are the actual keypoints extracted from plate2.mp4
    detected_plates = {
        # Frame ~45: Plate id=0
        45: [
            {
                "id": 0,
                "keypoints": [
                    [924.0, 216.4],  # TL
                    [1004.2, 226.1], # TR
                    [1005.0, 271.9], # BR
                    [924.8, 262.1]   # BL
                ]
            }
        ],
        # Frame ~150: Plate id=1
        150: [
            {
                "id": 1,
                "keypoints": [
                    [925.2, 217.5],  # TL
                    [983.8, 226.1],  # TR
                    [983.8, 268.5],  # BR
                    [926.2, 260.2]   # BL
                ]
            }
        ]
    }

    # Load keypoints from JSON if provided
    if args.keypoints and os.path.exists(args.keypoints):
        with open(args.keypoints) as f:
            detected_plates = json.load(f)
        print(f"[INFO] Loaded keypoints from {args.keypoints}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process frames with detections
    saved_count = 0
    for frame_num in sorted(detected_plates.keys()):
        # Seek to frame
        frame_idx = int(frame_num)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"[WARN] Cannot read frame {frame_idx}")
            continue

        # Process each plate in this frame
        for plate in detected_plates[frame_num]:
            plate_id = plate["id"]
            keypoints = np.array(plate["keypoints"], dtype=np.float32)

            print(f"[INFO] Frame {frame_idx}: Plate {plate_id}")
            print(f"       Keypoints: TL({keypoints[0][0]:.0f},{keypoints[0][1]:.0f}) "
                  f"TR({keypoints[1][0]:.0f},{keypoints[1][1]:.0f}) "
                  f"BR({keypoints[2][0]:.0f},{keypoints[2][1]:.0f}) "
                  f"BL({keypoints[3][0]:.0f},{keypoints[3][1]:.0f})")

            # Warp plate
            warped = warp_plate(frame, keypoints)
            if warped is not None:
                output_path = os.path.join(args.output_dir, f"plate_{plate_id:04d}.jpg")
                cv2.imwrite(output_path, warped)
                print(f"       Saved: {output_path} ({warped.shape[1]}x{warped.shape[0]})")
                saved_count += 1

                # Also save frame with keypoints drawn
                frame_viz = frame.copy()
                for i, pt in enumerate(keypoints):
                    cv2.circle(frame_viz, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
                # Draw polygon
                pts = keypoints.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_viz, [pts], True, (0, 255, 0), 2)

                frame_output = os.path.join(args.output_dir, f"frame_{plate_id:04d}.jpg")
                cv2.imwrite(frame_output, frame_viz)
                print(f"       Frame: {frame_output}")
            else:
                print(f"       [WARN] Failed to warp plate")

    cap.release()
    print(f"\n[DONE] Saved {saved_count} aligned plate images to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
