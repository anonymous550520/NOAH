import torch
import argparse
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def load_model(model_name):
    if model_name == "llava_onevision_7b":
        pretrained_model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(pretrained_model_name, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
    return {"model": model, "processor": processor}

def generate(model_info, video_path, query, max_new_tokens=1024):
    model = model_info["model"]
    processor = model_info["processor"]

    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": query},
                ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        num_frames=8,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

    if "assistant" in result:
        response = result.split("assistant")[-1].strip()
    else:
        response = result.strip()

    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="llava_onevision_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()