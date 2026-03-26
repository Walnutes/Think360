import os
import json
import time
import argparse
import base64
import io
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
from PIL import Image
from prompt import ToT_JUDGEMENT_SYSTEM_PROMPT, build_tot_judgement_user_prompt
from openai import OpenAI
from tree_metric import calculate_tot_width, calculate_tot_depth
from tree_extraction import validate_tree_structure


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


def get_judgement_with_retry(client, system_prompt, user_prompt, model='gpt-4o-mini',
                             model_max_tokens=None, timeout=60, max_retries=3,
                             initial_delay=1, item_id=None, node_id=None, image_path=None):
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
                timeout=timeout,
                temperature=0,
            )
            jtime = round(time.time() - start_time, 4)
            jtokens = payload.usage.completion_tokens
            response = payload.choices[0].message.content.strip()

            if response.lower() in ['true', 'yes', '1']:
                return True, jtime, jtokens, None
            elif response.lower() in ['false', 'no', '0']:
                return False, jtime, jtokens, None
            else:
                if retries >= max_retries:
                    logger.error(f"Failed to parse judgement for item {item_id}, node {node_id}: {response}")
                    return None, jtime, jtokens, f"PARSE_ERROR: {response}"
                logger.warning(f"Unexpected response for item {item_id}, node {node_id} (retry {retries}): {response}")
                retries += 1
                time.sleep(delay)
                delay += 0.1
                continue

        except Exception as e:
            if retries >= max_retries:
                jtime = round(time.time() - start_time, 4)
                logger.error(f"Max retries reached for item {item_id}, node {node_id}: {e}")
                return None, jtime, 0, str(e)
            logger.warning(f"Error: {str(e)}, waiting {delay}s and start {model}'s No.{retries} retry...")
            time.sleep(delay)
            delay += 0.1
            retries += 1


def build_parent_chain(node_id: str, nodes_dict: dict):
    """Build a chain of parent nodes from root to the immediate parent."""
    parent_chain = []
    current_node = nodes_dict.get(node_id)
    
    if not current_node:
        return parent_chain
    
    parent_id = current_node.get('parent_id')
    
    while parent_id is not None:
        parent_node = nodes_dict.get(parent_id)
        if parent_node:
            parent_chain.insert(0, parent_node)  # Insert at beginning to maintain order
            parent_id = parent_node.get('parent_id')
        else:
            break
    
    return parent_chain


def judge_tree_nodes(item: dict, model: str, client: OpenAI, model_max_tokens: int = None):
    """Judge all nodes in a tree structure for a single item."""
    question = item.get('question', '')
    answer = item.get('answer', '')
    item_id = item.get('id')

    image_path = None
    image_path_raw = item.get('image_path')
    if image_path_raw:
        image_filename = image_path_raw.split("/")[-1]
        image_dir = "/path/to/images"
        image_path = os.path.join(image_dir, image_filename)
    # Get tree extraction
    tree_extraction = item.get('tree_extraction', {})
    tree_data = tree_extraction.get('tree', {})
    nodes = tree_data.get('nodes', [])
    
    if not nodes:
        logger.warning(f"No nodes found for item {item_id}")
        return {
            'tree': tree_data,
            'judgement_stats': {
                'total_nodes': 0,
                'judged_nodes': 0,
                'correct_nodes': 0,
                'incorrect_nodes': 0,
                'error_nodes': 0
            },
            'time': 0,
            'tokens': 0
        }
    
    # Build nodes dictionary for easy lookup
    nodes_dict = {node['id']: node for node in nodes}
    
    # Find the final node (highest depth, last in sequence at that depth)
    max_depth = max(node.get('depth', 0) for node in nodes)
    # Sort by sequence number to get the last one
    final_nodes = sorted(
        [n for n in nodes if n.get('depth') == max_depth],
        key=lambda n: int(n['id'].split('.')[1]),
    )
    final_node_id = final_nodes[-1]['id'] if final_nodes else None
    
    # Judge each node
    total_time = 0
    total_tokens = 0
    correct_count = 0
    incorrect_count = 0
    error_count = 0
    
    for node in nodes:
        node_id = node['id']
        
        # Build parent chain
        parent_chain = build_parent_chain(node_id, nodes_dict)

        # Build judgement prompt
        user_prompt = build_tot_judgement_user_prompt(
            question=question,
            reference_answer=answer,
            current_node=node,
            parent_nodes=parent_chain,
            # Check if this is the final node
            is_final_node=(node_id == final_node_id),
        )
        
        # Get judgement
        judgement, jtime, jtokens, jerror = get_judgement_with_retry(
            client=client,
            system_prompt=ToT_JUDGEMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            model_max_tokens=model_max_tokens,
            item_id=item_id,
            node_id=node_id,
            image_path=image_path
        )
        
        total_time += jtime
        total_tokens += jtokens
        
        # Update node with judgement
        node['judgement'] = judgement
        if jerror:
            node['judgement_error'] = jerror
            error_count += 1
        elif judgement:
            correct_count += 1
        else:
            incorrect_count += 1

    # Update tree data with judged nodes
    tree_data['nodes'] = nodes
    
    # Prepare result
    return {
        'tree': tree_data,
        'judgement_stats': {
            'total_nodes': len(nodes),
            'judged_nodes': correct_count + incorrect_count,
            'correct_nodes': correct_count,
            'incorrect_nodes': incorrect_count,
            'error_nodes': error_count
        },
        'time': total_time,
        'tokens': total_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge reasoning tree nodes.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input JSON with tree extractions")
    parser.add_argument('--model', type=str, default='gpt-4o', help="Model to use for judgement")
    args = parser.parse_args()

    client = OpenAI(
        api_key="API_KEY",
        base_url="API_URL",
    )
    model_max_tokens = 4096

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} items")
    results = []

    for item in tqdm(data, desc="Judging Tree Nodes"):
        try:
            result_item = item.copy()
            result_item['tree_judgement'] = judge_tree_nodes(item, args.model, client, model_max_tokens)
            results.append(result_item)
        except Exception as e:
            logger.error(f"Error processing item (id={item.get('id')}): {str(e)}")
            result_item = item.copy()
            result_item['tree_judgement'] = {
                'tree': {'nodes': [], 'metadata': {'max_depth': 0, 'total_nodes': 0}},
                'judgement_stats': {
                    'total_nodes': 0,
                    'judged_nodes': 0,
                    'correct_nodes': 0,
                    'incorrect_nodes': 0,
                    'error_nodes': 0
                },
                'time': 0,
                'tokens': 0,
                'error': str(e)
            }
            results.append(result_item)

    # --- Calculate tree metrics ---
    valid_width_count = 0
    valid_depth_count = 0
    total_width_sum = 0.0
    total_depth_sum = 0.0
    failed_items = {}  # {error_reason: [item_ids]}
    for item in tqdm(results, desc="Calculating tree metrics"):
        item_id = item.get('id')
        tree_judgement = item.get('tree_judgement', {})
        tree_data = tree_judgement.get('tree', {})

        is_valid, msg = validate_tree_structure(tree_data)
        if not is_valid:
            if 'Parent depth must be less than child depth for node' in msg:
                msg = 'Parent depth must be less than child depth'
            error_reason = f"Invalid tree structure: {msg}"
            failed_items.setdefault(error_reason, []).append(item_id)
            tree_judgement['tot_width'] = None
            tree_judgement['tot_depth'] = None
            continue

        try:
            tot_width = calculate_tot_width(tree_data)
            tot_depth = calculate_tot_depth(tree_data)
            if tot_width is not None:
                valid_width_count += 1
                total_width_sum += tot_width
            if tot_depth is not None:
                valid_depth_count += 1
                total_depth_sum += tot_depth
            tree_judgement['tot_width'] = tot_width
            tree_judgement['tot_depth'] = tot_depth
        except Exception as e:
            error_reason = f"Calculation error: {type(e).__name__}: {str(e)}"
            failed_items.setdefault(error_reason, []).append(item_id)
            tree_judgement['tot_width'] = None
            tree_judgement['tot_depth'] = None

    # --- Save results ---
    # Re-save results with metrics
    output_file = args.input_file.replace('_prediction_tot.json', '_accuracy_tot.json').replace('/prediction_tot/', '/judgement_tot/')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Results saved to: {output_file}")

    # --- Statistics ---
    # Calculate overall statistics (excluding null items)
    total_judged = sum(1 for r in results if 'tree_judgement' in r)
    total_nodes = sum(r.get('tree_judgement', {}).get('judgement_stats', {}).get('total_nodes', 0) for r in results)
    total_correct = sum(r.get('tree_judgement', {}).get('judgement_stats', {}).get('correct_nodes', 0) for r in results)
    total_incorrect = sum(r.get('tree_judgement', {}).get('judgement_stats', {}).get('incorrect_nodes', 0) for r in results)
    total_errors = sum(r.get('tree_judgement', {}).get('judgement_stats', {}).get('error_nodes', 0) for r in results)

    avg_tot_width = total_width_sum / valid_width_count if valid_width_count > 0 else None
    avg_tot_depth = total_depth_sum / valid_depth_count if valid_depth_count > 0 else None
    total_failed_count = sum(len(ids) for ids in failed_items.values())

    statistics = {
        "model": args.model,
        "input_file": args.input_file,
        "output_file": output_file,
        "total_items": len(results),
        "judgement_stats": {
            "items_with_judgements": total_judged,
            "total_nodes": total_nodes,
            "correct_nodes": total_correct,
            "incorrect_nodes": total_incorrect,
            "error_nodes": total_errors,
            "node_accuracy": total_correct / total_nodes if total_nodes > 0 else None
        },
        "tree_metrics": {
            "valid_width_calculations": valid_width_count,
            "valid_depth_calculations": valid_depth_count,
            "avg_tot_width": avg_tot_width,
            "avg_tot_depth": avg_tot_depth,
            "failed_items_count": total_failed_count,
            "failed_items_by_reason": failed_items
        }
    }
    # Determine statistics output file path
    stat_file = output_file.replace('/judgement_tot/', '/statistics_tot/').replace('_accuracy_tot.json', '_stat_tot.json')
    # Save statistics
    os.makedirs(os.path.dirname(stat_file), exist_ok=True)
    with open(stat_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=4)
    logger.info(f"Statistics saved to: {stat_file}")
