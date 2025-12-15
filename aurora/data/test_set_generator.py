"""Generate diverse, stratified test set"""

import numpy as np
import json
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from aurora import config

class TestSetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_comprehensive_test_set(self) -> dict:
        """
        Generate 100+ test scenarios across difficulty levels
        Stratified by:
        - Difficulty (easy, medium, hard)
        - Workload type (web, batch, ml, mixed)
        - Problem size (small, medium, large)
        """
        
        test_set = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        # Easy scenarios (30): Small, well-balanced
        for i in range(30):
            scenario = self._generate_easy_scenario(i)
            test_set['easy'].append(scenario)
        
        # Medium scenarios (40): Moderate size, some constraints
        for i in range(40):
            scenario = self._generate_medium_scenario(i)
            test_set['medium'].append(scenario)
        
        # Hard scenarios (30): Large, tight constraints
        for i in range(30):
            scenario = self._generate_hard_scenario(i)
            test_set['hard'].append(scenario)
        
        return test_set
    
    def _generate_easy_scenario(self, idx: int) -> dict:
        """Easy: 5-15 tasks, 2-5 nodes, plenty of capacity"""
        
        num_tasks = self.rng.randint(5, 16)
        num_nodes = self.rng.randint(2, 6)
        
        # Tasks: small, uniform
        tasks = []
        for t in range(num_tasks):
            tasks.append({
                'id': f'easy_{idx}_t{t}',
                'cpu': self.rng.uniform(0.5, 2.0),
                'memory': self.rng.uniform(1, 4),
                'duration': self.rng.uniform(10, 100),
                'arrival_time': float(t * 5),  # Spread out, verify it is float
                'sla_latency_ms': 200,
                'priority': int(self.rng.choice([1, 2, 3]))
            })
        
        # Nodes: ample capacity (2x demand)
        total_cpu_demand = sum(t['cpu'] for t in tasks)
        total_mem_demand = sum(t['memory'] for t in tasks)
        
        # Avoid division by zero if accidental 0 nodes (range is 2,6 so ok)
        nodes = []
        for n in range(num_nodes):
            nodes.append({
                'id': f'easy_{idx}_n{n}',
                'cpu_capacity': (total_cpu_demand * 2) / num_nodes,
                'memory_capacity': (total_mem_demand * 2) / num_nodes,
                'cost_per_hour': 0.1,
                'energy_efficiency': 1.0
            })
        
        return {
            'id': f'easy_scenario_{idx}',
            'difficulty': 1,
            'category': 'easy',
            'tasks': tasks,
            'nodes': nodes
        }
    
    def _generate_medium_scenario(self, idx: int) -> dict:
        """Medium: 20-50 tasks, 5-15 nodes, moderate constraints"""
        
        workload_type = self.rng.choice(['web', 'batch', 'mixed'])
        
        # Simplified implementations compared to doc to handle method signature
        num_tasks = self.rng.randint(20, 51)
        num_nodes = self.rng.randint(5, 16)
        
        tasks = []
        for t in range(num_tasks):
            tasks.append({
                'id': f'medium_{idx}_t{t}',
                'cpu': self.rng.uniform(1.0, 4.0),
                'memory': self.rng.uniform(2.0, 8.0),
                'duration': self.rng.uniform(50, 200),
                'arrival_time': float(t * 2),
                'sla_latency_ms': 150,
                'priority': int(self.rng.choice([1, 2, 3, 4]))
            })
            
        nodes = []
        total_cpu = sum(t['cpu'] for t in tasks) * 1.5 # 1.5x capacity
        total_mem = sum(t['memory'] for t in tasks) * 1.5
        
        for n in range(num_nodes):
            nodes.append({
                'id': f'medium_{idx}_n{n}',
                'cpu_capacity': total_cpu / num_nodes,
                'memory_capacity': total_mem / num_nodes,
                'cost_per_hour': 0.2,
                'energy_efficiency': 1.0
            })
            
        return {
            'id': f'medium_scenario_{idx}',
            'difficulty': 3,
            'category': 'medium',
            'workload_type': workload_type,
            'tasks': tasks,
            'nodes': nodes
        }
    
    def _generate_hard_scenario(self, idx: int) -> dict:
        """Hard: 50-100 tasks, 15-30 nodes, tight constraints"""
        
        num_tasks = self.rng.randint(50, 101)
        num_nodes = self.rng.randint(15, 31)
        
        # Tasks: heterogeneous, log-normal distribution
        tasks = []
        for t in range(num_tasks):
            size_factor = self.rng.lognormal(0, 1.0)
            
            tasks.append({
                'id': f'hard_{idx}_t{t}',
                'cpu': np.clip(0.5 * size_factor, 0.1, 32),
                'memory': np.clip(1 * size_factor * 2, 0.5, 128),
                'duration': self.rng.lognormal(3, 1),  # Mean ~20s, variance high
                'arrival_time': self._generate_bursty_arrival(t),
                'sla_latency_ms': self.rng.uniform(50, 150),
                'priority': int(self.rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1]))
            })
        
        # Nodes: tight capacity (1.2x demand)
        # Heterogeneous node types
        node_types = [
            {'cpu': 4, 'mem': 8, 'prob': 0.4},
            {'cpu': 8, 'mem': 16, 'prob': 0.3},
            {'cpu': 16, 'mem': 32, 'prob': 0.2},
            {'cpu': 32, 'mem': 64, 'prob': 0.1}
        ]
        
        nodes = []
        for n in range(num_nodes):
            node_type = self.rng.choice(
                node_types,
                p=[nt['prob'] for nt in node_types]
            )
            
            nodes.append({
                'id': f'hard_{idx}_n{n}',
                'cpu_capacity': node_type['cpu'],
                'memory_capacity': node_type['mem'],
                'cost_per_hour': 0.01 * node_type['cpu'],
                'energy_efficiency': self.rng.uniform(0.8, 1.2)
            })
        
        return {
            'id': f'hard_scenario_{idx}',
            'difficulty': 5,
            'category': 'hard',
            'tasks': tasks,
            'nodes': nodes
        }
    
    def _generate_bursty_arrival(self, task_idx: int) -> float:
        """Generate bursty arrival pattern"""
        # Create bursts every 50 tasks
        burst_id = task_idx // 50
        within_burst = task_idx % 50
        
        burst_start = burst_id * 100  # Bursts separated by 100s
        arrival = burst_start + within_burst * self.rng.exponential(0.5)
        
        return float(arrival)
    
    def save_test_set(self, test_set: dict, output_path: str):
        """Save test set to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native types for JSON serialization
        def default(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, 'w') as f:
            json.dump(test_set, f, indent=2, default=default)
        
        # Save statistics
        stats = {
            'total_scenarios': sum(len(scenarios) for scenarios in test_set.values()),
            'by_difficulty': {
                diff: len(scenarios)
                for diff, scenarios in test_set.items()
            }
        }
        
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved {stats['total_scenarios']} test scenarios to {output_path}")

# Generate test set
if __name__ == '__main__':
    generator = TestSetGenerator(seed=42)
    test_set = generator.generate_comprehensive_test_set()
    
    # Use config for path if available, else default
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'test_set.json')
    output_path = os.path.normpath(output_path)
    
    generator.save_test_set(test_set, output_path)
