import torch
import os
import json
import time
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from loguru import logger

def init_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor
    
def inference_one_step(model, processor, image_path, question):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}",
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        end_time = time.time()
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_tokens = len(generated_ids_trimmed[0]) if generated_ids_trimmed[0] !=None else 0
        inference_time = round(end_time - start_time, 4)
        
        return output_text[0], inference_time, output_tokens
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        return f"ERROR: {str(e)}", 0, 0

def process_item(item, index, model, processor, result_queue, image_dir):
    question = item.get("question")
    image_path = os.path.join(image_dir, item.get("image_path").split("/")[-1])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    prediction, inference_time, output_tokens = inference_one_step(model, processor, image_path, question)
    
    item['prediction'] = prediction
    item['reason_time'] = inference_time
    item['reason_tokens'] = output_tokens
    result_queue.put((index, item))

def run_prediction(model_path, model_name, json_path, image_dir, output_path_dir):
    
    model, processor = init_model(model_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    logger.info(f"Loaded {len(json_data)} items")

    result_queue = Queue()
    output_path = os.path.join(output_path_dir, f"{model_name}_prediction.json")
    if output_path_dir:
        os.makedirs(output_path_dir, exist_ok=True)

    # Initialize results list
    results = []
    
    pbar = tqdm(total=len(json_data), desc=f"{model_name} Prediction:")
    
    for idx, item in enumerate(json_data):
        process_item(item, idx, model, processor, result_queue, image_dir)
        index, result_item = result_queue.get()
        
        # Ensure results list is long enough
        while len(results) <= index:
            results.append(None)
        results[index] = result_item
        pbar.update(1)
        
        # Save results immediately after each item is processed
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    pbar.close()
    logger.info(f"Prediction finished. Results saved to: {output_path}")
    return output_path

def qwen_trans_demo(model_path, demo_image_path, demo_question):
    model, processor = init_model(model_path)
    output_text, inference_time, output_tokens = inference_one_step(model, processor, demo_image_path, demo_question)
    print(output_text)
    print(inference_time)
    print(output_tokens)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Qwen series model evaluation on Think360.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--json_path', type=str, required=True, help="Path to the input JSON file (e.g., data.json).")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the images directory.")
    parser.add_argument('--output_path_dir', type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    model_name = args.model_path.split("/")[-1]
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Model name: {model_name}")

    prediction_file = run_prediction(
        model_path=args.model_path,
        model_name=model_name,
        json_path=args.json_path,
        image_dir=args.image_dir,
        output_path_dir=args.output_path_dir,
    )
    logger.info(f"Evaluation completed. Results saved to: {prediction_file}")