import json
import time
import base64
import os
import argparse
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from PIL import Image
import io

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def inference_one_step(question, base64_image, client, model, model_max_tokens):
    start_time = time.time()
    payload = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            },
        ],
        max_tokens=model_max_tokens,
        temperature=0.7
    )
    end_time = time.time()
    reason_time = round(end_time - start_time, 4)
    reason_tokens = payload.usage.completion_tokens
    response = payload.choices[0].message.content
    return response, reason_time, reason_tokens

def run_prediction(client, model, model_max_tokens):
    json_path = "/path/to/data.json"
    image_dir = "/path/to/images_dir"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    logger.info(f"Loaded {len(json_data)} items")

    model_name = model.split('/')[-1]
    output_path_dir = '/path/to/results_dir'
    output_path = os.path.join(output_path_dir, f"{model_name}_prediction.json")
    
    if output_path_dir:
        os.makedirs(output_path_dir, exist_ok=True)

    results = []
    
    for i, item in enumerate(tqdm(json_data, desc=f"Processing {model}")):
        new_item = item.copy()
        question = item.get("question")
        image_path = os.path.join(image_dir, item.get("image_path").split("/")[-1])
        
        base64_image = encode_image(image_path)
        prediction, reason_time, reason_tokens = inference_one_step(question, base64_image, client, model, model_max_tokens)
        
        new_item['prediction'] = prediction
        new_item['reason_time'] = reason_time
        new_item['reason_tokens'] = reason_tokens

        results.append(new_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Prediction finished. Results saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation on a JSON dataset.")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help="The model name to use for inference.")
    parser.add_argument('--model_max_tokens', type=int, default=1024, help="The maximum number of tokens for the model.")
    args = parser.parse_args()

    client = OpenAI(
        api_key="API_KEY", 
        base_url="BASE_URL",
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    
    prediction_file = run_prediction(
        client,
        model=args.model,
        model_max_tokens=args.model_max_tokens
    )