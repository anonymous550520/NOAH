import os
import cv2
import base64
import argparse
import numpy as np
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI

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

    if debug_save_dir:
        try:
            os.makedirs(debug_save_dir, exist_ok=True)
        except Exception as e:
            print(f"[warn] could not create debug dir {debug_save_dir}: {e}")
            debug_save_dir = None

    base64_frames: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = resize_image_max_side(img, max_side=max_side)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)

        if debug_save_dir:
            # Save a copy for inspection (filename encodes original frame index)
            debug_path = os.path.join(debug_save_dir, f"frame_{idx:06d}.jpg")
            try:
                img.save(debug_path, format="JPEG", quality=jpeg_quality)
            except Exception as e:
                print(f"[warn] failed to save debug frame {debug_path}: {e}")

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        base64_frames.append(b64)

    cap.release()
    return base64_frames


def build_multimodal_messages(query: str, base64_images: List[str]) -> List[Dict[str, Any]]:
    content_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": query},
    ]
    for b64 in base64_images:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return [
        {
            "role": "user",
            "content": content_blocks,
        }
    ]


def load_model(model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    return {"client": client, "model_name": model_name}

def _check_finish_reason(response):
    FINISH_REASON_OK = {"stop", "length"}
    name = getattr(response, "finish_reason", None)
    if name is None:
        print("[debug] finish_reason=None (not provided)")
        return
    if name in FINISH_REASON_OK:
        print(f"[debug] finish_reason={name} (OK)")
        return

    msg = f"finish_reason={name} stopped"
    raise RuntimeError(f"Gpt response abnormal termination: {msg}")

def generate(model_info, video_path, query, max_new_tokens=1024, num_frames=128, frame_max_side=512, jpeg_quality=85):

    base64_images = sample_video_frames_base64(
        video_path, num_frames, max_side=frame_max_side, jpeg_quality=jpeg_quality
    )
    if not base64_images:
        raise RuntimeError("No frames sampled from video")

    messages = build_multimodal_messages(query, base64_images)

    client = model_info["client"]
    model_name = model_info["model_name"]
    if model_name == "gpt-5":
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            reasoning_effort="minimal"
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_new_tokens
        )

    _check_finish_reason(response.choices[0])

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt-4o", type=str)
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