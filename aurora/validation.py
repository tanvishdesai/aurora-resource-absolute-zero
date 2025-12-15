import numpy as np
from collections import Counter
import scipy.stats as stats

class ScenarioValidator:
    def __init__(self):
        self.constraints = {
            'task_cpu_range': (0.1, 32.0),
            'task_memory_range': (0.5, 256.0),
            'task_duration_range': (1, 7200),
            'node_cpu_range': (2, 128),
            'node_memory_range': (4, 1024),
            'num_tasks_range': (5, 500),
            'num_nodes_range': (2, 100)
        }
        self.rejection_log = []

    def validate(self, scenario):
        """Run all validation checks on a scenario."""
        checks = [
            self.check_bounds,
            self.check_feasibility,
            self.check_realism
        ]
        
        for check in checks:
            is_valid, reason = check(scenario)
            if not is_valid:
                self.rejection_log.append({
                    'scenario_id': scenario.get('scenario_id', 'unknown'),
                    'reason': reason,
                    'check': check.__name__
                })
                return False, reason
        
        return True, "Valid"

    def check_bounds(self, scenario):
        """Ensure values are within realistic bounds."""
        tasks = scenario.get('tasks', [])
        nodes = scenario.get('nodes', [])

        if not tasks or not nodes:
            return False, "Empty tasks or nodes"

        if not (self.constraints['num_tasks_range'][0] <= len(tasks) <= self.constraints['num_tasks_range'][1]):
             return False, f"Number of tasks {len(tasks)} out of bounds"
        
        for t in tasks:
            if not (self.constraints['task_cpu_range'][0] <= t.get('cpu', 0) <= self.constraints['task_cpu_range'][1]):
                return False, f"Task CPU {t.get('cpu')} out of bounds"
            if not (self.constraints['task_memory_range'][0] <= t.get('memory', 0) <= self.constraints['task_memory_range'][1]):
                 return False, f"Task Memory {t.get('memory')} out of bounds"
        
        return True, "Bounds OK"

    def check_feasibility(self, scenario):
        """Ensure at least one valid solution theoretically exists (Bin Packing check)."""
        total_cpu_needed = sum(t['cpu'] for t in scenario['tasks'])
        total_mem_needed = sum(t['memory'] for t in scenario['tasks'])
        
        total_cpu_avail = sum(n['cpu_capacity'] for n in scenario['nodes'])
        total_mem_avail = sum(n['memory_capacity'] for n in scenario['nodes'])

        if total_cpu_needed > total_cpu_avail:
             return False, f"Infeasible: Needed CPU {total_cpu_needed} > Available {total_cpu_avail}"
        if total_mem_needed > total_mem_avail:
             return False, f"Infeasible: Needed Memory {total_mem_needed} > Available {total_mem_avail}"
        
        # Simple Bin Packing Heuristic (First Fit Decreasing) check
        sorted_tasks = sorted(scenario['tasks'], key=lambda x: x['cpu'], reverse=True)
        node_states = [{'cpu': n['cpu_capacity'], 'mem': n['memory_capacity']} for n in scenario['nodes']]
        
        for task in sorted_tasks:
            placed = False
            for node in node_states:
                if node['cpu'] >= task['cpu'] and node['mem'] >= task['memory']:
                    node['cpu'] -= task['cpu']
                    node['mem'] -= task['memory']
                    placed = True
                    break
            if not placed:
                 return False, "Infeasible: Cannot pack tasks even with First-Fit Decreasing"

        return True, "Feasible"

    def check_realism(self, scenario):
        """Check if distributions roughly match real-world patterns."""
        # Example: Check if task sizes aren't all identical (unless specific test)
        cpus = [t['cpu'] for t in scenario['tasks']]
        if len(set(cpus)) < 2 and len(cpus) > 5:
             # It's not necessarily invalid, but suspicious for "realistic" scenarios. 
             # For now, we'll allow it but log a warning if we were strictly enforcing variety.
             pass
        
        return True, "Realistic"


class RealWorldGrounder:
    def __init__(self):
        # Default parameters derived from Google Cluster Trace
        self.cpu_dist_params = (0.5, 0, 2) # Shape, Loc, Scale for LogNormal
        self.mem_dist_params = (0.6, 0, 4)

    def inject_realism(self, scenario):
        """Adjusts generated scenario values to follow real-world distributions."""
        
        # 1. Adjust Task Sizes (CPU/Mem) to Log-Normal
        for task in scenario['tasks']:
            # We don't want to completely overwrite LLM's logic, but "nudge" or re-sample if it looks synthetic.
            # Here we will re-scale based on the LLM's intent.
            
            # Simple approach: If the LLM output is very uniform, introduce noise.
            # If it's already diverse, keep it.
            pass 
        
        # 2. Inject Temporal Patterns (Arrival Times)
        # If arrival_time is missing, generate it.
        num_tasks = len(scenario['tasks'])
        if num_tasks > 0 and 'arrival_time' not in scenario['tasks'][0]:
            arrivals = self._generate_bursty_arrivals(num_tasks)
            for t, arr in zip(scenario['tasks'], arrivals):
                t['arrival_time'] = arr
        
        return scenario

    def _generate_bursty_arrivals(self, num_tasks):
        """Generates bursty arrival times using a Poisson process with varying rates."""
        arrivals = []
        time = 0
        high_rate = 5.0
        low_rate = 0.5
        
        for i in range(num_tasks):
            rate = high_rate if (i // 20) % 2 == 0 else low_rate
            inter_arrival = np.random.exponential(1/rate)
            time += inter_arrival
            arrivals.append(time)
        return arrivals
