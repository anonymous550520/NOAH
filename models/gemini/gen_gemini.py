import os
import cv2
import base64
import argparse
import numpy as np
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from google import genai

def resize_image_max_side(img, max_side=512):
    """Resize keeping aspect so that max(width, height) <= max_side."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    ratio = max_side / float(m)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = img.resize((nw, nh), Image.LANCZOS)
    return resized


def sample_video_frames_base64(video_path, num_frames, max_side=512, jpeg_quality=85, debug_save_dir=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Clamp request
    if num_frames <= 0:
        num_frames = 1
    if num_frames > total_frames:
        num_frames = total_frames

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Prepare debug directory if requested
    # debug_save_dir="debug_frames/"
    if debug_save_dir:
        try:
            os.makedirs(debug_save_dir, exist_ok=True)
        except Exception as e:
            print(f"[warn] could not create debug dir {debug_save_dir}: {e}")
            debug_save_dir = None

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = resize_image_max_side(img, max_side=max_side)

        if debug_save_dir:
            # Save a copy for inspection (filename encodes original frame index)
            debug_path = os.path.join(debug_save_dir, f"frame_{idx:06d}.jpg")
            try:
                img.save(debug_path, format="JPEG", quality=jpeg_quality)
            except Exception as e:
                print(f"[warn] failed to save debug frame {debug_path}: {e}")

        frames.append(img)

    cap.release()
    return frames


def build_multimodal_messages(query, frames):
    content_blocks = []
    for img in frames:
        content_blocks.append(img)
    return content_blocks + [query]


def load_model(model_name):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    return {"client": client, "model_name": model_name}

def _check_finish_reason(response, candidate=None):
    """Centralized finish / block reason checking.

    - If block_reason present -> raise.
    - If candidate provided and finish_reason not OK -> raise.
    - Accepts MAX_TOKENS as OK (partial but usable output).
    """
    # 1. Prompt-level block (no candidates case)
    pf = getattr(response, "prompt_feedback", None)
    block_reason = getattr(pf, "block_reason", None)
    if block_reason:
        br_name = getattr(block_reason, "value", None) or getattr(block_reason, "name", None) or str(block_reason)
        raise RuntimeError(f"Gemini blocked generation (block_reason={br_name})")

    if candidate is None:
        return  # Nothing further to validate

    FINISH_REASON_OK = {"STOP", "FINISH_REASON_UNSPECIFIED", "MAX_TOKENS"}
    fr = getattr(candidate, "finish_reason", None)
    if fr is None:
        return
    name = getattr(fr, "value", None) or getattr(fr, "name", None) or str(fr)
    if name in FINISH_REASON_OK:
        return
    raise RuntimeError(f"Gemini response abnormal termination: finish_reason={name}")

def generate(model_info, video_path, query, max_new_tokens=1024, num_frames=128, frame_max_side=512, jpeg_quality=85):

    frames = sample_video_frames_base64(
        video_path, num_frames, max_side=frame_max_side, jpeg_quality=jpeg_quality
    )
    if not frames:
        raise RuntimeError("No frames sampled from video")

    messages = build_multimodal_messages(query, frames)

    client = model_info["client"]
    model_name = model_info["model_name"]
    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )

    candidates = getattr(response, "candidates", None) or []
    primary = candidates[0] if candidates else None
    _check_finish_reason(response, primary)

    return response.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gemini-2.0-flash", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=128)

    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.", num_frames=args.num_frames)

    print(caption)

if __name__ == "__main__":
    main()