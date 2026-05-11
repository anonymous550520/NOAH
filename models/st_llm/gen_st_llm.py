import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ST-LLM'))

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *

CONVERSATION_CFG_PATH = os.path.join(os.path.dirname(__file__), 'ST-LLM/config/instructblipbase_stllm_conversation.yaml')
QA_CFG_PATH = os.path.join(os.path.dirname(__file__), 'ST-LLM/config/instructblipbase_stllm_qa.yaml')

# run `huggingface-cli download farewellthree/ST_LLM_weight`
CONVERSATION_CKPT_PATH = 'path/to/STLLM_conversation_weight'
QA_CKPT_PATH = 'path/to/STLLM_qa_weight'

def load_model(model_name):
    if model_name == 'st_llm_conversation':
        args_cfg_path = CONVERSATION_CFG_PATH
        args_ckpt_path = CONVERSATION_CKPT_PATH
    elif model_name == 'st_llm_qa':
        args_cfg_path = QA_CFG_PATH
        args_ckpt_path = QA_CKPT_PATH
    else:
        raise ValueError(f"Model {model_name} not found")

    parser = argparse.ArgumentParser(description="Model-specific argument parser")
    parser.add_argument("--gpu-id", type=int, default=0)
    args, _ = parser.parse_known_args()

    args.cfg_path = args_cfg_path
    args.ckpt_path = args_ckpt_path
    args.options = None

    cfg = Config(args)

    ckpt_path = args.ckpt_path
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = ckpt_path
    model_config.llama_model = ckpt_path
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.to(torch.float16)
    CONV_VISION = CONV_instructblip_Vicuna0
    
    chat = Chat(model, device='cuda:{}'.format(args.gpu_id))

    return {"chat": chat, "conv_template": CONV_VISION}

def generate(model_info, video_path, query, max_new_tokens=1024):
    chat = model_info["chat"]
    conv_template = model_info["conv_template"]
        
    chat_state = conv_template.copy()
    img_list = []

    chat.upload_video(video_path, chat_state, img_list, 64, text=query)
    chat.ask("###Human: " + query + " ###Assistant: ", chat_state)
    llm_message = chat.answer(conv=chat_state,
                img_list=img_list,
                num_beams=5,
                do_sample=False,
                temperature=1,
                max_new_tokens=max_new_tokens,
                max_length=2048)[0]

    return llm_message.split('</s>')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="st_llm_conversation", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()





