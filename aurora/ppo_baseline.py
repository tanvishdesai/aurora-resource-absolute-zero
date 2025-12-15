import os
import time
from aurora.envs.gym_env import AuroraGymEnv
from aurora.data_loader import RealWorldDataManager

# Try importing SB3, handle if missing
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  stable-baselines3 not installed. PPO baseline will not work.")
    SB3_AVAILABLE = False

class PPOAllocationAgent:
    def __init__(self, model_path="ppo_aurora_model"):
        self.model_path = model_path
        self.model = None
        if SB3_AVAILABLE and os.path.exists(f"{model_path}.zip"):
            self.model = PPO.load(model_path)
            print(f"✅ PPO Model loaded from {model_path}")

    def train(self, total_timesteps=10000):
        if not SB3_AVAILABLE:
            print("❌ Cannot train: stable-baselines3 missing.")
            return

        print("🚀 Starting PPO Training...")
        # Load training data
        dm = RealWorldDataManager()
        train_scenarios = dm.get_training_scenarios(source='synthetic', num_scenarios=50)
        
        env = AuroraGymEnv(scenarios=train_scenarios)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        self.model = model
        print("✅ PPO Training Complete & Saved.")

    def allocate(self, scenario):
        """
        Baseline interface compatibility.
        Returns list of allocations.
        """
        policy = self.generate_policy(scenario)
        return policy.get("allocations", [])

    def generate_policy(self, scenario):
        """
        Mimic the DefenderAgent interface.
        """
        if not self.model:
            return self._fallback_policy(scenario)
            
        # Create temp env just for inference state tracking
        env = AuroraGymEnv(scenarios=[scenario])
        obs, _ = env.reset()
        done = False
        allocations = []
        
        task_idx = 0
        tasks = scenario.get('tasks', [])
        nodes = scenario.get('nodes', [])
        
        while not done and task_idx < len(tasks):
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Map action (int) to node_id
            node_id = nodes[action % len(nodes)]['id']
            allocations.append({
                "task_id": tasks[task_idx]['id'],
                "node_id": node_id
            })
            
            obs, reward, done, truncated, info = env.step(action)
            task_idx += 1
            
        return {
            "policy_id": "ppo_baseline",
            "reasoning": "Determined by PPO Policy Network",
            "allocations": allocations
        }

    def _fallback_policy(self, scenario):
        return {
            "policy_id": "ppo_failed",
            "reasoning": "PPO model not loaded",
            "allocations": []
        }

if __name__ == "__main__":
    # Standalone training trigger
    agent = PPOAllocationAgent()
    agent.train()
