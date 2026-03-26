import requests
from PIL import Image
import time
import json
import os
import argparse
from tqdm import tqdm
from loguru import logger
import shutil

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

def init_model(model_path):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def process_one_step(question, image_path, model, processor):
    """
    Process one step with retry mechanism
    """
    if '<image>' in question:
        question = question.replace('<image>', '')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    raw_image = Image.open(image_path)
    inputs = processor(
        images=raw_image,
        text=input_text,
        return_tensors='pt'
    ).to(0, torch.float16)
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
    end_time = time.time()
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    reason_token = len(generated_ids_trimmed[0]) if generated_ids_trimmed[0] != None else 0
    # import pdb; pdb.set_trace()
    reason_time = end_time - start_time
    return output_text[0], reason_time, reason_token
            

def process_with_retry(question, image_path, model, processor, max_retries=100, retry_delay=1):
    """
    Process one step with retry mechanism
    """
    attempt = 0
    while attempt < max_retries:
        try:
            response, reason_time, reason_token = process_one_step(question, image_path, model, processor)
            return response, reason_time, reason_token
            
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {e}")
            return f"ERROR: Image file not found - {str(e)}", 0, 0
            
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay += 0.5
                logger.info(f"Delay increased to {retry_delay} seconds before retrying...")
            else:
                logger.error(f"Max retries reached. Skipping.")
                return f"ERROR: Max retries reached - {str(e)}", 0, 0

def llava_onevison_demo(model_path):
    question = "Answer the following question based on the image."
    image_path = "/path/to/demo_image.png"
    model, processor = init_model(model_path)
    response, reason_time, reason_token = process_one_step(question, image_path, model, processor)
    # import pdb; pdb.set_trace()
    return response, reason_time, reason_token

def think360_eval(model_path, json_path, image_dir, save_dir, device_id):
    """
    Process think360 benchmark data sequentially.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    model, processor = init_model(model_path)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    logger.info(f"Processing {len(json_data)} items")
    
    os.makedirs(save_dir, exist_ok=True)
    
    output_filename = f"think360_{os.path.basename(model_path)}_prediction.json"
    output_filepath = os.path.join(save_dir, output_filename)
    
    results = []
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Found existing result file with {len(results)} items")
        except Exception as e:
            logger.error(f"Error loading existing result file: {e}")
            results = []
    
    for i, item in enumerate(tqdm(json_data, desc=f"Processing (GPU {device_id})")):
        if i < len(results) and results[i].get('prediction') and not results[i]['prediction'].startswith('ERROR:'):
            continue
        
        new_item = item.copy()
        question = item.get("question")
        image_path = os.path.join(image_dir, item.get("image_path").split("/")[-1])
        
        output_text, reason_time, reason_token = process_with_retry(question, image_path, model, processor)
        new_item['prediction'] = output_text
        new_item['reason_time'] = reason_time
        new_item['reason_tokens'] = reason_token
        
        if i < len(results):
            results[i] = new_item
        else:
            results.append(new_item)
        
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving result file: {e}")
    
    logger.info(f"Processing completed. Results saved to: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA-OneVision evaluation")
    parser.add_argument('--model_path', type=str, default='/path/to/model', help="Path to the model")
    parser.add_argument('--json_path', type=str, default='/path/to/data.json', help="Path to the input JSON file.")
    parser.add_argument('--image_dir', type=str, default='/path/to/images', help="Path to the images directory.")
    parser.add_argument('--save_dir', type=str, default='/path/to/results', help="Directory to store prediction results.")
    parser.add_argument('--device_id', type=int, default=0, help="GPU device ID to use")
    
    args = parser.parse_args()
    
    think360_eval(
        args.model_path,
        args.json_path,
        args.image_dir,
        args.save_dir,
        args.device_id,
    )
