import os
import json
import re
import time
import argparse
import base64
import io
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
from PIL import Image
from prompt import ToT_EXTRACTION_SYSTEM_PROMPT, build_tot_extraction_user_prompt
from openai import OpenAI
from tot.utils import fix_tree_structure

def encode_image(image_path):
    if image_path.lower().endswith('.png'):
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='JPEG', quality=95)
            return base64.b64encode(byte_stream.getvalue()).decode('utf-8')
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')


def build_message(system_prompt, user_prompt, image_path=None):
    if image_path is not None:
        base64_image = encode_image(image_path)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_response_with_retry(client, system_prompt, user_prompt, model='gpt-4o',
                            model_max_tokens=None, timeout=60, max_retries=5,
                            initial_delay=1, item_id=None, image_path=None):
    retries = 0
    delay = initial_delay
    message = build_message(system_prompt, user_prompt, image_path)

    while retries <= max_retries:
        try:
            start_time = time.time()
            payload = client.chat.completions.create(
                model=model,
                messages=message,
                max_tokens=model_max_tokens,
                timeout=timeout,  # Set timeout in seconds
                temperature=0.7 # default
                # temperature=0 # greedy decoding
            )
            tot_extract_time = round(time.time() - start_time, 4)
            tot_extract_tokens = payload.usage.completion_tokens
            tot_extract_response = payload.choices[0].message.content

            tree_data = parse_tree_response(tot_extract_response)
            is_valid, msg = validate_tree_structure(tree_data)

            if is_valid:
                return tree_data, tot_extract_time, tot_extract_tokens, None

            if "Parent depth must be less than child" in msg:
                # Validation failed with parent depth issue - try to fix
                logger.warning(f"Tree validation failed for item {item_id}: {msg}. Attempting fix...")
                tree_data_fixed = fix_tree_structure(tree_data)
                # Validate again after fix
                is_valid_after_fix, msg_after_fix = validate_tree_structure(tree_data_fixed)
                if is_valid_after_fix:
                    logger.info(f"Tree structure fixed successfully for item {item_id}")
                    return tree_data_fixed, tot_extract_time, tot_extract_tokens, "VALIDATION_FAILED_FIXED"
                else:
                    logger.warning(f"Tree fix failed for item {item_id}: {msg_after_fix}")
                    return tree_data, tot_extract_time, tot_extract_tokens, f"VALIDATION_FAILED_FIXED_FAILED: {msg_after_fix}"

            if retries >= max_retries:
                return tree_data, tot_extract_time, tot_extract_tokens, f"VALIDATION_FAILED_MAX_RETRIES: {msg}"
            logger.warning(f"Validation error: {msg} for item {item_id} (retry {retries}/{max_retries})")
            retries += 1
            continue

        except Exception as e:
            if retries >= max_retries:
                tot_extract_time = round(time.time() - start_time, 4)
                logger.error(f"Max retries reached for item {item_id}: {e}")
                empty_tree = {'nodes': [], 'metadata': {'max_depth': 0, 'total_nodes': 0}}
                return empty_tree, tot_extract_time, 0, str(e)
            logger.warning(f"Error: {e}, waiting {delay}s (retry {retries}/{max_retries})...")
            time.sleep(delay)
            delay += 0.1
            retries += 1


def parse_tree_response(response_text: str):
    """
    Parse the text response from API into tree structure.
    Expected format uses markers like [NODE:id], [PARENT:...], etc.
    Depth is inferred from NODE ID (first digit).
    """
    nodes = []
    current_node = {}
    current_field = None
    current_value = []
    
    lines = response_text.strip().split('\n')
    
    def save_field():
        """Save accumulated field value"""
        nonlocal current_field, current_value
        if current_field and current_value:
            value = ' '.join(current_value).strip()
            
            if current_field == 'parent':
                current_node['parent_id'] = value if value.lower() != 'none' else None
            elif current_field == 'original':
                current_node['original_text'] = value
            elif current_field == 'type':
                current_node['step_type'] = value
        
        current_field = None
        current_value = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for field markers
        if line_stripped.startswith('[NODE:'):
            # Save previous node if exists
            save_field()
            if current_node:
                nodes.append(current_node.copy())
            
            # Start new node
            match = re.search(r'\[NODE:(.*?)\]', line_stripped)
            if match:
                node_id = match.group(1).strip()
                # Infer depth from node ID (first digit before the dot)
                try:
                    depth = int(node_id.split('.')[0])
                except (ValueError, IndexError):
                    depth = 1  # Default to 1 if parsing fails
                current_node = {'id': node_id, 'depth': depth}
        
        
        elif line_stripped.startswith('[PARENT:'):
            save_field()
            match = re.search(r'\[PARENT:(.*?)\]', line_stripped)
            if match:
                current_field = 'parent'
                current_value = [match.group(1)]
        
        elif line_stripped.startswith('[ORIGINAL:'):
            save_field()
            match = re.search(r'\[ORIGINAL:(.*?)\]', line_stripped)
            if match:
                current_field = 'original'
                current_value = [match.group(1)]
        
        elif line_stripped.startswith('[TYPE:'):
            save_field()
            match = re.search(r'\[TYPE:(.*?)\]', line_stripped)
            if match:
                current_field = 'type'
                current_value = [match.group(1)]
        
        elif current_field and line_stripped:
            # Continue accumulating multi-line content
            current_value.append(line_stripped)
    
    # Save last field and node
    save_field()
    if current_node:
        nodes.append(current_node)
    
    # Calculate metadata
    max_depth = max((n.get('depth', 1) for n in nodes), default=0)
    return {
        'nodes': nodes,
        'metadata': {'max_depth': max_depth, 'total_nodes': len(nodes)},
    }


def validate_tree_structure(tree_data: dict):
    """Validate the tree structure for consistency"""
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return False, "No nodes in tree"
    
    # Check ID format and uniqueness
    ids = set()
    for node in nodes:
        node_id = node["id"]
        if node_id in ids:
            return False, f"Duplicate node ID: {node_id}"
        ids.add(node_id)
        
        # Validate ID format
        parts = node_id.split(".")
        if len(parts) != 2:
            return False, f"Invalid ID format: {node_id}, expected {{depth}}.{{sequence}}"
        try:
            depth = int(parts[0])
            if depth != node["depth"]:
                return False, f"ID depth {parts[0]} doesn't match node depth {node['depth']}"
        except ValueError:
            return False, f"Invalid ID format: {node_id}"
    
    # Check parent references
    for node in nodes:
        if node["parent_id"] is not None:
            if node["parent_id"] not in ids:
                return False, f"Node {node['id']} references non-existent parent {node['parent_id']}"
            # Parent should have smaller depth
            parent = next(n for n in nodes if n["id"] == node["parent_id"])
            if parent["depth"] != node["depth"]-1:
                return False, f"Parent depth must be less than child depth for node {node['id']}"
    
    return True, "Valid"

def extract_tree_from_item(item, model, client):
    """Extract reasoning tree from a single data item"""
    question = item.get('question', '')
    answer = item.get('answer', '')
    solution = item.get('solution', '') or item.get('prediction', '')
    image_path = item.get('image_path', None).split("/")[-1]
    image_dir = "/path/to/images"
    image_path = os.path.join(image_dir, image_path)

    user_prompt = build_tot_extraction_user_prompt(question, answer, solution)
    tree_data, extract_time, extract_tokens, error = get_response_with_retry(
        client, ToT_EXTRACTION_SYSTEM_PROMPT, user_prompt,
        model, item_id=item.get('id'), image_path=image_path,
    )

    result = {'tree': tree_data, 'time': extract_time, 'tokens': extract_tokens}
    if error:
        result['tot_error'] = error
    return result


def save_json(file_path: str, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reasoning tree structure from solutions.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input JSON file")
    parser.add_argument('--model', type=str, default='gpt-4o', help="Model to use for extraction")
    args = parser.parse_args()

    client = OpenAI(
        api_key="API_KEY",
        base_url="API_URL",
    )

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} items")
    results = []

    for item in tqdm(data, desc="Extracting Trees"):
        try:
            tree_result = extract_tree_from_item(item, args.model, client)
            result_item = item.copy()
            result_item['tree_extraction'] = tree_result
            results.append(result_item)
        except Exception as e:
            logger.error(f"Error processing item {item.get('id')}: {e}")
            result_item = item.copy()
            result_item['tree_extraction'] = {
                'tree': {'nodes': [], 'metadata': {'max_depth': 0, 'total_nodes': 0}},
                'time': 0, 'tokens': 0, 'tot_error': str(e),
            }
            results.append(result_item)

    total_ok = sum(1 for r in results if 'tot_error' not in r.get('tree_extraction', {}))
    total_fixed = sum(1 for r in results if r.get('tree_extraction', {}).get('tot_error', '').startswith('VALIDATION_FAILED_FIXED'))
    total_err = len(results) - total_ok - total_fixed
    logger.info(f"Statistics: {total_ok} succeeded, {total_fixed} fixed, {total_err} failed (total {len(results)})")

    output_file = args.input_file.replace('/prediction/', '/prediction_tot/').replace('_prediction.json', '_prediction_tot.json')
    save_json(output_file, results)
    logger.info(f"Results saved to: {output_file}")
