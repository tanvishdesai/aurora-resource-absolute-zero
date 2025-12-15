"""Hyperparameter tuning for RL baselines"""

import json
import os
import sys

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from stable_baselines3 import PPO
from aurora.envs.gym_env import AuroraGymEnv
from aurora.simulators.simulator import UnifiedResourceSimulator

# Suppress gym warnings
import warnings
warnings.filterwarnings("ignore")

def optimize_ppo(trial, scenarios):
    """Optuna objective for PPO"""
    
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    
    # Create environment
    # Use subset of scenarios for tuning speed
    tuning_scenarios = scenarios[:10]
    
    # We need a custom env that resets over a list of scenarios
    env = AuroraGymEnv(scenarios=tuning_scenarios)
    
    # Create model
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=0
        )
        
        # Train (shortened for demo/tuning speed, should be 50k+ in real run)
        model.learn(total_timesteps=10000)
        
        # Evaluate
        eval_env = AuroraGymEnv(scenarios=tuning_scenarios)
        
        mean_reward = 0
        n_eval_episodes = 5
        
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward
                
                if truncated:
                    break
            
            mean_reward += episode_reward
        
        mean_reward /= n_eval_episodes
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')
    
    return mean_reward

def run_hyperparameter_search(scenarios, n_trials=10, output_path=None):
    """Run Optuna hyperparameter search"""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'config', 'best_ppo_params.json')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: optimize_ppo(trial, scenarios),
        n_trials=n_trials
    )
    
    print(f"\n✅ Best hyperparameters found:")
    print(study.best_params)
    print(f"Best reward: {study.best_value:.3f}")
    
    # Save best params
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params

# Run tuning
if __name__ == '__main__':
    # Load test set
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_set.json')
    if not os.path.exists(test_set_path):
        print(f"⚠️ Test set not found at {test_set_path}, generating...")
        from aurora.data.test_set_generator import TestSetGenerator
        gen = TestSetGenerator()
        test_set = gen.generate_comprehensive_test_set()
    else:
        with open(test_set_path, 'r') as f:
            test_set = json.load(f)
            
    # Flatten scenarios
    all_scenarios = []
    for cat in test_set:
         all_scenarios.extend(test_set[cat])

    # Run search
    # Only run 2 trials to verify it works, preventing long wait
    run_hyperparameter_search(all_scenarios, n_trials=2)
