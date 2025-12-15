import random

class RandomAllocation:
    """Baseline 1: Random assignment"""
    def allocate(self, scenario):
        allocations = []
        nodes = scenario.get('nodes', [])
        if not nodes: return []
        
        for task in scenario.get('tasks', []):
            node = random.choice(nodes)
            allocations.append({'task_id': task['id'], 'node_id': node['id']})
        return allocations

class RoundRobin:
    """Baseline 2: Round-robin load balancing"""
    def allocate(self, scenario):
        allocations = []
        nodes = scenario.get('nodes', [])
        if not nodes: return []
        
        for i, task in enumerate(scenario.get('tasks', [])):
            node = nodes[i % len(nodes)]
            allocations.append({'task_id': task['id'], 'node_id': node['id']})
        return allocations

class GreedyFirstFit:
    """Baseline 3: First-fit greedy"""
    def allocate(self, scenario):
        allocations = []
        nodes = scenario.get('nodes', [])
        if not nodes: return []
        
        # Track usage to simulate "current state" during allocation
        node_usage = {n['id']: {'cpu': 0, 'mem': 0} for n in nodes}
        
        for task in scenario.get('tasks', []):
            placed = False
            for node in nodes:
                nid = node['id']
                if (node_usage[nid]['cpu'] + task['cpu'] <= node['cpu_capacity']) and \
                   (node_usage[nid]['mem'] + task['memory'] <= node['memory_capacity']):
                    
                    allocations.append({'task_id': task['id'], 'node_id': nid})
                    node_usage[nid]['cpu'] += task['cpu']
                    node_usage[nid]['mem'] += task['memory']
                    placed = True
                    break
            
            # If not placed, we skip (or could assign to random/best effort). 
            # For strict baseline, we leave unassigned = failure.
        return allocations

class BestFit:
    """Baseline 4: Best-fit bin packing (Minimizes leftover space)"""
    def allocate(self, scenario):
        allocations = []
        nodes = scenario.get('nodes', [])
        if not nodes: return []
        
        node_usage = {n['id']: {'cpu': 0, 'mem': 0} for n in nodes}
        
        for task in scenario.get('tasks', []):
            best_node = None
            min_waste = float('inf')
            
            for node in nodes:
                nid = node['id']
                curr_cpu = node_usage[nid]['cpu']
                curr_mem = node_usage[nid]['mem']
                
                # Check feasibility
                if (curr_cpu + task['cpu'] <= node['cpu_capacity']) and \
                   (curr_mem + task['memory'] <= node['memory_capacity']):
                    
                    # Calculate waste (unused CPU after placement)
                    # Simple heuristic: minimize remaining CPU
                    remaining_cpu = node['cpu_capacity'] - (curr_cpu + task['cpu'])
                    
                    if remaining_cpu < min_waste:
                        min_waste = remaining_cpu
                        best_node = node
            
            if best_node:
                nid = best_node['id']
                allocations.append({'task_id': task['id'], 'node_id': nid})
                node_usage[nid]['cpu'] += task['cpu']
                node_usage[nid]['mem'] += task['memory']
        
        return allocations
