import gymnasium as gym
from gymnasium import spaces
import numpy as np
from aurora.simulators.simulator import UnifiedResourceSimulator
from aurora.rewards.reward_engine import MultiObjectiveRewardEngine

class AuroraGymEnv(gym.Env):
    """
    Gymnasium environment for Resource Allocation.
    Observation: [Task_CPU, Task_Mem, Task_Dur, Task_SLA] + [Node_1_State... Node_N_State]
    Action: Discrete(Num_Nodes) (Assign current task to Node X)
    """
    
    def __init__(self, scenarios=None):
        super(AuroraGymEnv, self).__init__()
        
        self.simulator = UnifiedResourceSimulator()
        self.reward_engine = MultiObjectiveRewardEngine()
        self.scenarios = scenarios if scenarios else []
        self.current_scenario = None
        self.current_task_idx = 0
        self.node_states = [] # Track dynamic state [cpu_used, mem_used]
        
        # Define Space Dimensions (Fixed for simplicity, though real system is dynamic)
        # Assuming Validation ensures max 100 tasks, 20 nodes for RL training context
        self.max_nodes = 20
        self.node_feat_dim = 4 # Capacity CPU, Capacity Mem, Used CPU, Used Mem
        self.task_feat_dim = 4 # CPU, Mem, Dur, SLA
        
        # total obs = task (4) + nodes (max_nodes * 4)
        obs_dim = self.task_feat_dim + (self.max_nodes * self.node_feat_dim)
        
        self.observation_space = spaces.Box(low=0, high=10000, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_nodes)

    def set_scenarios(self, scenarios):
        self.scenarios = scenarios

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.scenarios:
            # Fallback simple scenario if none provided
            self.current_scenario = {
                "tasks": [{"id": "t1", "cpu": 1, "memory": 1, "duration": 10, "sla_latency_ms": 100}], 
                "nodes": [{"id": "n1", "cpu_capacity": 10, "memory_capacity": 10}]
            }
        else:
            # Pick random scenario
            idx = np.random.randint(0, len(self.scenarios))
            self.current_scenario = self.scenarios[idx]
        
        self.current_task_idx = 0
        
        # Initialize Node States (Used amount)
        self.num_nodes = len(self.current_scenario['nodes'])
        # Dynamic state: [cpu_cap, mem_cap, cpu_used, mem_used]
        self.node_states = []
        for n in self.current_scenario['nodes']:
            self.node_states.append([n['cpu_capacity'], n['memory_capacity'], 0.0, 0.0])
            
        return self._get_obs(), {}

    def step(self, action):
        # Action is Node Index
        # Map to actual node
        if action >= self.num_nodes:
            # Invalid action (selecting non-existent node), heavy penalty
            reward = -1.0
            done = True # Or continue with penalty? Let's end to teach validity.
            return self._get_obs(), reward, done, False, {"error": "Invalid Node Selection"}

        task = self.current_scenario['tasks'][self.current_task_idx]
        node_idx = action
        
        # Simulate placement check
        node_state = self.node_states[node_idx]
        node_cap_cpu, node_cap_mem, node_used_cpu, node_used_mem = node_state
        
        # Very simple internal simulation for step-wise reward
        # Real simulation happens at end of episode in AURORA flow, but RL needs step feedback.
        
        violation = False
        if (node_used_cpu + task['cpu'] > node_cap_cpu) or (node_used_mem + task['memory'] > node_cap_mem):
            reward = -0.5 # Penalty for overflow
            violation = True
        else:
            # Valid placement
            self.node_states[node_idx][2] += task['cpu']
            self.node_states[node_idx][3] += task['memory']
            reward = 0.1 # Small positive for successful placement
        
        # Check SLA (Simplified proxy)
        # In real sim, we calc latency. Here we assume load correlates to latency.
        utilization = self.node_states[node_idx][2] / self.node_states[node_idx][0]
        if utilization > 0.9:
            reward -= 0.1 # Penalty for near-saturation
            
        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.current_scenario['tasks'])
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Construct vector
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 1. Task Features
        if self.current_task_idx < len(self.current_scenario['tasks']):
            t = self.current_scenario['tasks'][self.current_task_idx]
            obs[0:4] = [t.get('cpu',0), t.get('memory',0), t.get('duration',0), t.get('sla_latency_ms',0)]
            
        # 2. Node Features
        # Fill up to max_nodes
        for i in range(min(self.num_nodes, self.max_nodes)):
            obs[4 + i*4 : 4 + (i+1)*4] = self.node_states[i]
            
        return obs
