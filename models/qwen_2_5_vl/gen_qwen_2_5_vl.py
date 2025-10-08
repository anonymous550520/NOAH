import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_model(model_name):
    if model_name == "qwen_2_5_vl_3b":
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif model_name == "qwen_2_5_vl_7b":
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif model_name == "qwen_2_5_vl_32b":
        model_path = "Qwen/Qwen2.5-VL-32B-Instruct"        
    elif model_name == "qwen_2_5_vl_72b":
        model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)

    return {"model": model, "processor": processor}

def generate(model_info, video_path, query, max_new_tokens=1024):
    model = model_info["model"]
    processor = model_info["processor"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                    "max_frames": 128,
                },
                {"type": "text", "text": query},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=fps
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="qwen_2_5_vl_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()