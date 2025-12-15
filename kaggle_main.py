
import sys
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to python path to ensure aurora modules are found
sys.path.append(os.getcwd())

from aurora import config
from aurora.llm_wrapper import QwenHandler
from aurora.controller import MetaEvolutionaryController

def main():
    parser = argparse.ArgumentParser(description="AURORA System - Kaggle Execution with Qwen")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH, help="Path to the Qwen model directory")
    parser.add_argument("--iterations", type=int, default=10, help="Number of self-play iterations")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"], help="Mode: train or evaluate")
    parser.add_argument("--data_source", type=str, default="synthetic", choices=["synthetic", "google", "azure"], help="Data source for evaluation")
    
    args = parser.parse_args()

    print(f"🚀 Starting AURORA in {args.mode.upper()} mode using Qwen model at {args.model_path}")
    
    # 1. Initialize LLM Handler
    try:
        llm_handler = QwenHandler(model_name_or_path=args.model_path)
        print("✅ Qwen Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Failed to load Qwen Model: {e}")
        return

    # 2. Initialize Controller with injected LLM Handler
    controller = MetaEvolutionaryController(llm_handler=llm_handler)
    
    if args.mode == "train":
        print(f"Starting Training for {args.iterations} iterations...")
        
        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1} [Difficulty: {controller.curriculum.get_current_level_name()}] ---")
            controller.run_iteration(i)
            
        print("\nTraining Complete.")
        print(f"Total Policies Collected: {len(controller.policy_db.policies)}")
        
    elif args.mode == "evaluate":
        print(f"Starting Evaluation on {args.data_source} data...")
        
        from aurora.evaluation import AURORAValidator
        from aurora.data_loader import RealWorldDataManager
        from aurora.visualization import Visualizer
        
        # 1. Load Data
        data_manager = RealWorldDataManager()
        scenarios = data_manager.get_training_scenarios(source=args.data_source, num_scenarios=5)
        
        if not scenarios:
            print("❌ No scenarios found for evaluation.")
            return

        # 2. Run Benchmark
        validator = AURORAValidator(controller)
        results = validator.run_benchmark(scenarios)
        
        # 3. Print Summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        algos = [k for k in results[0].keys() if k != 'scenario_id']
        avg_metrics = {algo: {} for algo in algos}
        metric_keys = ['avg_latency', 'sla_violation_rate', 'total_cost']
        
        for algo in algos:
            for m in metric_keys:
                vals = [r[algo].get(m, 0) for r in results]
                avg_metrics[algo][m] = sum(vals) / len(vals)
                
        header = f"{'Algorithm':<20} | {'Latency':<10} | {'SLA Viol%':<10} | {'Cost':<10}"
        print(header)
        print("-" * len(header))
        
        for algo in algos:
            row = f"{algo:<20} | {avg_metrics[algo]['avg_latency']:<10.2f} | {avg_metrics[algo]['sla_violation_rate']*100:<10.1f} | {avg_metrics[algo]['total_cost']:<10.2f}"
            print(row)

        # 4. Generate Plots
        print("\n📊 Generating Plots...")
        viz = Visualizer()
        viz.plot_benchmark_comparison(results)
        print(f"✅ Plots saved to {viz.output_dir}")

if __name__ == "__main__":
    main()
