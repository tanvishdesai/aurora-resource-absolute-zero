import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from aurora import config

class Visualizer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(config.LOG_DIR, "plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_learning_curve(self, history):
        """Plot reward over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(history['iteration'], history['reward'], label='Average Reward')
        plt.title('AURORA Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "learning_curve.png"))
        plt.close()

    def plot_benchmark_comparison(self, benchmark_results):
        """Plot comparison of AURORA vs Baselines."""
        # Convert list of dicts to DataFrame for easier plotting
        # Structure: row=scenario, col=algo_metric
        
        algorithms = ['Random', 'RoundRobin', 'GreedyFirstFit', 'BestFit', 'AURORA']
        metrics = ['avg_latency', 'sla_violation_rate', 'total_cost']
        
        # Aggregate data
        agg_data = {algo: {m: [] for m in metrics} for algo in algorithms}
        
        for res in benchmark_results:
            for algo in algorithms:
                if algo in res:
                    for m in metrics:
                        agg_data[algo][m].append(res[algo].get(m, 0))
        
        # Calculate means
        means = {algo: {m: np.mean(vals) for m, vals in metrics_data.items()} 
                 for algo, metrics_data in agg_data.items()}
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, m in enumerate(metrics):
            ax = axes[i]
            values = [means[algo][m] for algo in algorithms]
            ax.bar(algorithms, values, color=['gray']*4 + ['blue'])
            ax.set_title(f'Mean {m}')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "benchmark_comparison.png"))
        plt.close()
