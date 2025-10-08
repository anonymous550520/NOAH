import argparse
import torch
from transformers import AutoConfig, AutoModel
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu 

MAX_NUM_FRAMES=16

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def load_model(model_name):
    if model_name == "mplug_owl3_7b":
        model_path = "mPLUG/mPLUG-Owl3-7B-241101"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)

    return {"model": model, "processor": processor, "tokenizer": tokenizer}


def generate(model_info, video_path, query, max_new_tokens=1024):
    model = model_info["model"]
    processor = model_info["processor"]
    tokenizer = model_info["tokenizer"]

    messages = [
        {"role": "user", "content": f"""<|video|> {query}"""},
        {"role": "assistant", "content": ""}
    ]

    videos = [video_path]

    video_frames = [encode_video(_) for _ in videos]
    inputs = processor(messages, images=None, videos=video_frames)

    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':max_new_tokens,
        'decode_text':True,
    })

    return model.generate(**inputs)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="mplug_owl3_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()