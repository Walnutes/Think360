from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig, PytorchEngineConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm
import os
import time
import argparse
from loguru import logger

def lmdeploy_inference(pipe, gen_config, image_path, question):
    if image_path is not None:
        image = load_image(image_path)
        input_data = (question, image)
    else:
        image = None
        input_data = question
    time_start = time.time()
    response = pipe(input_data, gen_config=gen_config)
    time_end = time.time()
    reason_time = time_end - time_start
    reason_token = response.generate_token_len

    return response.text, reason_time, reason_token

def think360_eval(model_path, json_path, image_dir, output_path_dir):

    model_name = model_path.split('/')[-1]
    if 'InternVL2_5' in model_path:
        gen_config = GenerationConfig(temperature=0.7, max_new_tokens=8192)
        pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(session_len=16384), trust_remote_code=True)
    elif 'InternVL3' in model_path:
        gen_config = GenerationConfig(temperature=0.6, max_new_tokens=16384)
        pipe = pipeline(model_path, backend_config=PytorchEngineConfig(session_len=32768, tp=1))
    else:   
        gen_config = GenerationConfig(temperature=0.7, max_new_tokens=16384)
        pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(session_len=32768, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    logger.info(f"Processing {len(json_data)} items")
    
    os.makedirs(output_path_dir, exist_ok=True)
    output_filename = f"{model_name}_prediction.json"
    output_filepath = os.path.join(output_path_dir, output_filename)
    
    results = []
    
    # process data
    for i, item in enumerate(tqdm(json_data, desc=f"{model_name} Prediction:")):
        new_item = item.copy()
        question = item.get("question")
        image_path = os.path.join(image_dir, item.get("image_path").split("/")[-1])
        
        output_text, reason_time, reason_token = lmdeploy_inference(pipe, gen_config, image_path, question)
        new_item['prediction'] = output_text
        new_item['reason_time'] = reason_time
        new_item['reason_tokens'] = reason_token
        
        # update or add result
        if i < len(results):
            results[i] = new_item
        else:
            results.append(new_item)
        
        # save in real time
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processing completed. Results saved to: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InternVL evaluation")
    parser.add_argument('--model_path', type=str, default='/path/to/model', help="Path to the model")
    parser.add_argument('--json_path', type=str, default='/path/to/data.json', help="Path to the input JSON file (e.g., data.json).")
    parser.add_argument('--image_dir', type=str, default='/path/to/images', help="Path to the images directory.")
    parser.add_argument('--output_path_dir', type=str, default='/path/to/results', help="Path to the output directory.")
    args = parser.parse_args()
    
    think360_eval(args.model_path, args.json_path, args.image_dir, args.output_path_dir)
