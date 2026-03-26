import argparse
import json
import os
import torch
import math
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from vllm import LLM, SamplingParams

def load_image(image_path: str, min_pixels: int, max_pixels: int):
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def vllm_eval_demo(question: str, image_path: str, sampling_params: SamplingParams):
    image = load_image(image_path, configs["min_pixels"], configs["max_pixels"])
    prompt_text = f"<|im_start|>system\n{configs['system_prompt']}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
    prompts = [
        {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        }
    ]
    outputs = llm.generate(prompts, sampling_params)
    token = len(outputs[0].outputs[0].token_ids)
    return outputs[0].outputs[0].text.strip()

if __name__ == "__main__":
    model_path = "/path/to/model"

    configs = {
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 8192,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 2048,
        "repetition_penalty": 1.0,
        "min_pixels": 262144,
        "max_pixels": 1000000,
        "system_prompt": "You are a helpful assistant that can answer questions and help with tasks.",
    }

    print(f"Initializing model from {model_path}")

    llm = LLM(
        model_path,
        tensor_parallel_size=configs["tensor_parallel_size"],
        dtype=torch.bfloat16,
        gpu_memory_utilization=configs["gpu_memory_utilization"],
        max_model_len=configs["max_model_len"],
        disable_log_stats=False,
    )
        
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=configs["temperature"],
        top_p=configs["top_p"],
        max_tokens=configs["max_tokens"],
        repetition_penalty=configs["repetition_penalty"],
    )

    demo_image_path = "/path/to/demo_image.jpg"
    demo_question = "Describe the image simply, and output english and chinese version."
    outputs = vllm_eval_demo(demo_question, demo_image_path, sampling_params)
    print(outputs)