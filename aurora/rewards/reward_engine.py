import numpy as np

class MultiObjectiveRewardEngine:
    def __init__(self, objectives=['latency', 'cost', 'energy', 'sla']):
        self.objectives = objectives
        # Default weights if not provided by scenario
        self.default_weights = {
            'latency': 0.4,
            'cost': 0.2,
            'energy': 0.1,
            'sla': 0.3
        }

    def compute_reward(self, execution_results, scenario):
        """
        Computes a scalar reward and individual objective scores.
        
        Args:
            execution_results (dict): Output from Simulator.
            scenario (dict): Original scenario (for targets/weights).
            
        Returns:
            tuple: (scalar_reward, scores_dict)
        """
        
        # 1. Normalize Metrics to [0, 1] scores where 1 is best
        
        # Latency Score: 1 / (1 + latency/target)
        # Assuming a baseline target of e.g., 50ms per task or scenario specific
        target_latency = float(len(scenario.get('tasks', [])) * 50) # Very rough baseline
        if target_latency == 0: target_latency = 1.0
        
        latency_val = execution_results.get('avg_latency', 0) * len(scenario.get('tasks', []))
        latency_score = 1.0 / (1.0 + (latency_val / target_latency))

        # Cost Score: 1 / (1 + cost)
        cost_val = execution_results.get('total_cost', 0)
        cost_score = 1.0 / (1.0 + cost_val)
        
        # Energy Score: 1 / (1 + energy/1000)
        energy_val = execution_results.get('total_energy', 0)
        energy_score = 1.0 / (1.0 + (energy_val / 1000.0))
        
        # SLA Score: 1 - violation_rate
        sla_rate = execution_results.get('sla_violation_rate', 0)
        sla_score = 1.0 - sla_rate

        scores = {
            'latency': latency_score,
            'cost': cost_score,
            'energy': energy_score,
            'sla': sla_score
        }

        # 2. Weighted Sum
        weights = scenario.get('objective_weights', self.default_weights)
        
        total_reward = sum(scores[k] * weights.get(k, 0.25) for k in self.objectives)
        
        # Clip to [0, 1] just in case
        total_reward = max(0.0, min(1.0, total_reward))
        
        return total_reward, scores
