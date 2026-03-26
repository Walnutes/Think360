import json
import os
from collections import defaultdict
from tqdm import tqdm
from tree_extraction import validate_tree_structure


def calculate_tot_width(tree_data):
    """
    Calculate tot_width metric: average accuracy of child node groups (size >= 1)
    
    Returns:
        float or None: Average accuracy of all groups, or None if no valid groups
    """
    nodes = tree_data.get('nodes', [])
    if not nodes:
        return None
    
    # Build parent -> children mapping
    children_map = defaultdict(list)
    for node in nodes:
        parent_id = node.get('parent_id')
        if parent_id is not None:
            children_map[parent_id].append(node)
    
    # Calculate accuracy for each group with size >= 1
    group_accuracies = []
    for parent_id, children in children_map.items():
        if len(children) >= 1:
            # Calculate accuracy for this group
            correct = sum(1 for child in children if child.get('judgement', False))
            accuracy = correct / len(children)
            group_accuracies.append(accuracy)
    
    if not group_accuracies:
        return None
    
    return sum(group_accuracies) / len(group_accuracies)


def calculate_tot_depth(tree_data):
    """
    Calculate tot_depth metric: average accuracy along deepest paths
    
    Returns:
        float or None: Average accuracy of all deepest paths, or None if no valid paths
    """
    nodes = tree_data.get('nodes', [])
    if not nodes:
        return None
    
    # Find max depth
    max_depth = max(node.get('depth', 1) for node in nodes)
    
    # Find all nodes at max depth
    deepest_nodes = [node for node in nodes if node.get('depth') == max_depth]
    
    if not deepest_nodes:
        return None
    
    # Build node id -> node mapping for quick lookup
    node_map = {node['id']: node for node in nodes}
    
    # For each deepest node, trace back to root and calculate path accuracy
    path_accuracies = []
    
    for leaf_node in deepest_nodes:
        path_nodes = []
        current = leaf_node
        
        # Trace back to root
        while current is not None:
            path_nodes.append(current)
            parent_id = current.get('parent_id')
            if parent_id is None:
                break
            current = node_map.get(parent_id)
            
            # Prevent infinite loop
            if current and current['id'] in [n['id'] for n in path_nodes[:-1]]:
                break
        
        # Calculate accuracy for this path
        if path_nodes:
            correct = sum(1 for node in path_nodes if node.get('judgement', False))
            accuracy = correct / len(path_nodes)
            path_accuracies.append(accuracy)
    
    if not path_accuracies:
        return None
    
    return sum(path_accuracies) / len(path_accuracies)
    # return min(path_accuracies)


def calculate_tree_metrics(input_file, output_file=None):
    """
    Calculate tot_width and tot_depth metrics for all items in the input file
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to input file)
    """
    if output_file is None:
        output_file = input_file
    
    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Separate metrics from items if exists
    items = []
    for item in data:
        if 'metrics' in item and len(item) == 1:
            # Skip existing metrics
            continue
        else:
            items.append(item)
    
    print(f"Total items: {len(items)}")
    
    # Track statistics
    failed_items = {}  # {error_reason: [item_ids]}
    valid_width_count = 0
    valid_depth_count = 0
    total_width_sum = 0.0
    total_depth_sum = 0.0
    
    # Process each item with progress bar
    print(f"Processing all items...")
    
    for item in tqdm(items, desc="Calculating tree metrics"):
        item_id = item.get('id')
        tree_extraction = item.get('tree_extraction', {})
        tree_data = tree_extraction.get('tree', {})
        
        # Initialize metrics
        tot_width = None
        tot_depth = None
        
        # Validate tree structure
        is_valid, msg = validate_tree_structure(tree_data)
        
        if not is_valid:
            if 'Parent depth must be less than child depth for node' in msg:
                msg =  'Parent depth must be less than child depth'
            error_reason = f"Invalid tree structure: {msg}"
            if error_reason not in failed_items:
                failed_items[error_reason] = []
            failed_items[error_reason].append(item_id)
            # Still add None values
            tree_extraction['tot_width'] = None
            tree_extraction['tot_depth'] = None
            continue
        
        # Calculate metrics
        try:
            tot_width = calculate_tot_width(tree_data)
            tot_depth = calculate_tot_depth(tree_data)
            
            # Update statistics
            if tot_width is not None:
                valid_width_count += 1
                total_width_sum += tot_width
            if tot_depth is not None:
                valid_depth_count += 1
                total_depth_sum += tot_depth
            
        except Exception as e:
            error_reason = f"Calculation error: {type(e).__name__}: {str(e)}"
            if error_reason not in failed_items:
                failed_items[error_reason] = []
            failed_items[error_reason].append(item_id)
        
        # Add metrics to tree_extraction
        tree_extraction['tot_width'] = tot_width
        tree_extraction['tot_depth'] = tot_depth
    
    # Calculate overall metrics
    avg_tot_width = total_width_sum / valid_width_count if valid_width_count > 0 else None
    avg_tot_depth = total_depth_sum / valid_depth_count if valid_depth_count > 0 else None
    
    # Calculate total failed items count
    total_failed_count = sum(len(ids) for ids in failed_items.values())
    
    # Create summary metrics
    summary_metrics = {
        "metrics": {
            "total_items": len(items),
            "valid_width_calculations": valid_width_count,
            "valid_depth_calculations": valid_depth_count,
            "avg_tot_width": avg_tot_width,
            "avg_tot_depth": avg_tot_depth,
            "failed_items_count": total_failed_count,
            "failed_items_by_reason": failed_items
        }
    }
    
    # Combine items and metrics
    output_data = items + [summary_metrics]
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate tree metrics from judgement results.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input JSON file with judgements")
    args = parser.parse_args()
    calculate_tree_metrics(args.input_file)
