import importlib
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

MODEL_MODULES = {
    "blip3_video_4b_32tokens": "models.blip3_video_4b_32tokens.gen_blip3_video_4b_32tokens",
    "blip3_video_4b_128tokens": "models.blip3_video_4b_32tokens.gen_blip3_video_4b_32tokens",
    "video_llama3_2b": "models.video_llama3.gen_video_llama3",
    "video_llama3_7b": "models.video_llama3.gen_video_llama3",
    "st_llm_conversation": "models.st_llm.gen_st_llm",
    "st_llm_qa":"models.st_llm.gen_st_llm",
    "longva_7b": "models.longva.gen_longva",
    "video_llava_7b": "models.video_llava.gen_video_llava",
    "llava_next_video_7b": "models.llava_next_video.gen_llava_next_video",
    "llava_onevision_7b": "models.llava_onevision.gen_llava_onevision",
    "mplug_owl3_7b": "models.mplug_owl3.gen_mplug_owl3",
    "qwen_2_5_vl_3b": "models.qwen_2_5_vl.gen_qwen_2_5_vl",
    "qwen_2_5_vl_7b": "models.qwen_2_5_vl.gen_qwen_2_5_vl",
    "qwen_2_5_vl_32b": "models.qwen_2_5_vl.gen_qwen_2_5_vl",
    "qwen_2_5_vl_72b": "models.qwen_2_5_vl.gen_qwen_2_5_vl",
    "gpt-4o-2024-08-06": "models.openai.gen_openai",
    "gemini-2.5-flash": "models.gemini.gen_gemini",   
}

def get_model_module(model_name):
    if model_name not in MODEL_MODULES:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return importlib.import_module(MODEL_MODULES[model_name])
