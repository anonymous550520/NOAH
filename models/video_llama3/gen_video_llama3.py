import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import argparse


def load_model(model_name):
    if model_name == "video_llama3_7b":
        pretrained_model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"
    elif model_name == "video_llama3_2b":
        pretrained_model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(pretrained_model_name, trust_remote_code=True)
    return {"model": model, "processor": processor}

def generate(model_info, video_path, query, max_new_tokens=1024):

    model = model_info["model"]
    processor = model_info["processor"]

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                {"type": "text", "text": query},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="video_llama3_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()