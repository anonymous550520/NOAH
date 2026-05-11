import argparse
import torch
from PIL import Image
import requests
import numpy as np
import av
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_model(model_name):
    if model_name == "video_llava_7b":
        model_id = "LanguageBind/Video-LLaVA-7B-hf"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")
    processor = VideoLlavaProcessor.from_pretrained(model_id)
    return {"model": model, "processor": processor}

def generate(model_info, video_path, query, max_new_tokens=1024):
    model = model_info["model"]
    processor = model_info["processor"]

    prompt = f"USER: <video>{query} ASSISTANT:"
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames


    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)

    inputs = processor(text=prompt, videos=clip, return_tensors="pt")

    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to("cuda", non_blocking=True)


    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        generate_ids = model.generate(
            **inputs,
            do_sample=False, temperature=0.0,  
            max_new_tokens=max_new_tokens
        )

    # generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
    else:
        response = response.strip()
    
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="video_llava_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()