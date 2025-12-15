"""Explainability Evaluation Tool"""

import numpy as np
import json
import os
import sys

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class ExplainabilityEvaluator:
    def __init__(self):
        self.metrics = ['clarity', 'correctness', 'completeness']
        
    def generate_evaluation_samples(self, n=50):
        """
        Sample policies and their explanations for human review.
        In a real run, this would query PolicyDatabase.
        Here we generate mock samples for demonstration.
        """
        samples = []
        for i in range(n):
            samples.append({
                'id': f'policy_{i}',
                'scenario_summary': f'Scenario {i}: Web workload with high traffic.',
                'policy_summary': f'Allocate web tasks to Node A, batch to Node B.',
                'explanation': f'Node A has high bandwidth suitable for web tasks. Node B is optimized for compute. This allocation minimizes latency.',
                'generated_at': '2025-12-15T10:00:00'
            })
        return samples

    def mock_human_ratings(self, samples, num_raters=3):
        """
        Simulate human ratings for testing the pipeline.
        Ratings are 1-5 scale.
        """
        ratings = {}
        for s in samples:
            s_id = s['id']
            ratings[s_id] = []
            
            # True quality of this sample (hidden)
            true_quality = np.random.normal(3.5, 0.8)
            
            for _ in range(num_raters):
                rater_scores = {}
                for m in self.metrics:
                    # Rater variation
                    score = np.random.normal(true_quality, 0.5)
                    score = np.clip(round(score), 1, 5)
                    rater_scores[m] = int(score)
                ratings[s_id].append(rater_scores)
                
        return ratings

    def evaluate(self, ratings):
        """
        Compute agreement and average scores.
        """
        results = {}
        
        # 1. Average Scores
        for m in self.metrics:
            all_scores = []
            for s_id in ratings:
                for r in ratings[s_id]:
                    all_scores.append(r[m])
            
            results[f'avg_{m}'] = np.mean(all_scores)
            results[f'std_{m}'] = np.std(all_scores)
            
        # 2. Inter-rater Reliability (Simple Percent Agreement for demo)
        # For rigorous research, use Fleiss' Kappa
        agreements = []
        for s_id, rater_list in ratings.items():
            # Check if all raters gave same score (strict) or within 1 point (relaxed)
            # Using relaxed agreement on 'overall' score (avg of metrics)
            
            avg_rater_scores = [np.mean(list(r.values())) for r in rater_list]
            diff = max(avg_rater_scores) - min(avg_rater_scores)
            if diff <= 1.0:
                agreements.append(1)
            else:
                agreements.append(0)
                
        results['inter_rater_agreement_relaxed'] = np.mean(agreements)
        
        return results

    def save_evaluation_kit(self, samples, output_file=None):
        """Save samples for distribution to human raters"""
        if output_file is None:
            # Go up two levels from aurora/evaluation/ -> aurora/ -> root/ -> results/
            output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'explainability_samples.json')
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"✅ Saved {len(samples)} samples to {output_file} for review.")

if __name__ == '__main__':
    evaluator = ExplainabilityEvaluator()
    
    print("1. Generating samples...")
    samples = evaluator.generate_evaluation_samples(50)
    
    print("2. Saving evaluation kit...")
    evaluator.save_evaluation_kit(samples)
    
    print("3. Simulating human ratings...")
    ratings = evaluator.mock_human_ratings(samples)
    
    print("4. Computing metrics...")
    results = evaluator.evaluate(ratings)
    
    print("\n📈 Explainability Evaluation Results (Simulated):")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
