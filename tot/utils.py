import json

def visualize_tree(tree_data: dict):
    """Generate a text visualization of the tree"""
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return "Empty tree"
    
    # Build children map
    children_map = {}
    for node in nodes:
        parent_id = node["parent_id"]
        if parent_id not in children_map:
            children_map[parent_id] = []
        children_map[parent_id].append(node)
    
    # Sort children by ID
    for parent_id in children_map:
        children_map[parent_id].sort(key=lambda x: x["id"])
    
    lines = []
    lines.append("=" * 80)
    lines.append("REASONING TREE VISUALIZATION")
    lines.append("=" * 80)
    
    def print_node(node, indent=0):
        prefix = "  " * indent
        step_type = f" [{node.get('step_type', 'N/A')}]" if node.get("step_type") else ""
        lines.append(f"{prefix}├─ [{node['id']}]{step_type}")
        original_text = node.get('original_text', '')
        lines.append(f"{prefix}│  {original_text[:300]}..." if len(original_text) > 300 else f"{prefix}│  {original_text}")
        
        # Print children
        if node["id"] in children_map:
            for child in children_map[node["id"]]:
                print_node(child, indent + 1)
    
    # Print from root nodes (depth 1, no parent)
    root_nodes = [n for n in nodes if n["parent_id"] is None]
    root_nodes.sort(key=lambda x: x["id"])
    
    for root in root_nodes:
        print_node(root)
    
    lines.append("=" * 80)
    metadata = tree_data.get("metadata", {})
    lines.append(f"Max Depth: {metadata.get('max_depth', 'N/A')}")
    lines.append(f"Total Nodes: {metadata.get('total_nodes', 'N/A')}")
    lines.append("=" * 80)
    print("\n".join(lines))


def fix_tree_structure(tree_data: dict):
    """
    Fix tree structure by correcting node depths and IDs based on parent-child relationships.
    Processes nodes layer by layer from top to bottom, maintaining sequence order.
    """
    from collections import deque
    
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return tree_data
    
    # Build parent-child map and find roots
    children_map = {}  # parent_id -> list of child nodes
    roots = []
    
    for node in nodes:
        parent_id = node.get("parent_id")
        if parent_id is None:
            roots.append(node)
        else:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node)
    
    # BFS to assign correct depths
    queue = deque()
    
    # Initialize roots with depth 1
    for root in roots:
        root["correct_depth"] = 1
        queue.append(root)
    
    while queue:
        current = queue.popleft()
        current_id = current["id"]
        current_depth = current["correct_depth"]
        
        # Process children
        if current_id in children_map:
            for child in children_map[current_id]:
                child["correct_depth"] = current_depth + 1
                queue.append(child)
    
    # Add original index to each node to preserve order
    for idx, node in enumerate(nodes):
        node["_original_index"] = idx
    
    # Group nodes by correct depth
    depth_groups = {}
    for node in nodes:
        depth = node.get("correct_depth", node["depth"])
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node)
    
    # Assign new IDs layer by layer, maintaining original sequence order
    id_mapping = {}
    
    for depth in sorted(depth_groups.keys()):
        nodes_at_depth = depth_groups[depth]
        # Sort by original index in the list to maintain order
        nodes_at_depth.sort(key=lambda n: n["_original_index"])
        
        for seq, node in enumerate(nodes_at_depth, start=1):
            old_id = node["id"]
            new_id = f"{depth}.{seq}"
            id_mapping[old_id] = new_id
    
    # Apply new IDs and update parent references
    for node in nodes:
        old_id = node["id"]
        node["id"] = id_mapping[old_id]
        node["depth"] = node["correct_depth"]
        
        if node.get("parent_id") and node["parent_id"] in id_mapping:
            node["parent_id"] = id_mapping[node["parent_id"]]
        
        # Clean up temporary fields
        if "correct_depth" in node:
            del node["correct_depth"]
        if "_original_index" in node:
            del node["_original_index"]
    
    # Sort nodes by depth and sequence for better readability in JSON
    def sort_key(node):
        node_id = node["id"]
        depth, seq = map(int, node_id.split('.'))
        return (depth, seq)
    
    nodes.sort(key=sort_key)
    
    # Recalculate metadata
    if nodes:
        max_depth = max(n.get('depth', 1) for n in nodes)
        tree_data['metadata'] = {
            'max_depth': max_depth,
            'total_nodes': len(nodes),
        }
    
    return tree_data