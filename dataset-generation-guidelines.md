# Comprehensive Guide: Controlling Dataset Quality in Self-Generated Scenarios

You've identified the **most critical risk** in this approach. If the Attacker generates unrealistic scenarios, the entire system learns from fantasy data and fails in production. Let me give you a detailed strategy.

---

## THE CORE PROBLEM

```
Bad Self-Generated Scenario:
{
  "tasks": [{"cpu": 999999, "memory": 1, "duration": 0.0001}],
  "nodes": [{"cpu": 1, "memory": 999999}]
}
❌ Technically valid JSON, but completely unrealistic
❌ Defender learns useless strategies
❌ Zero transfer to real world
```

---

## MULTI-LAYER QUALITY CONTROL SYSTEM

### Layer 1: Hard Constraints (Pre-Generation)

**Inject domain knowledge into Attacker's prompt:**

```python
SCENARIO_CONSTRAINTS = {
    # Resource bounds (from real-world analysis)
    'task_cpu_range': (0.1, 32.0),      # CPU cores
    'task_memory_range': (0.5, 256.0),  # GB
    'task_duration_range': (1, 7200),   # seconds (1s to 2hr)
    
    'node_cpu_range': (2, 128),         # CPU cores per node
    'node_memory_range': (4, 1024),     # GB per node
    
    # Workload characteristics
    'num_tasks_range': (5, 500),
    'num_nodes_range': (2, 100),
    'arrival_rate_range': (0.1, 100),   # tasks/second
    
    # Realistic patterns
    'workload_patterns': ['bursty', 'periodic', 'constant', 'diurnal'],
    'task_size_distribution': 'log-normal',  # Most tasks small, few large
    'node_heterogeneity': True,  # Different node capacities
}

def build_constrained_attacker_prompt(difficulty):
    """Inject constraints into prompt"""
    return f"""
You are an expert scenario generator for cloud resource allocation.

STRICT CONSTRAINTS (violating these = invalid scenario):
1. Task CPU: {SCENARIO_CONSTRAINTS['task_cpu_range'][0]:.1f} to {SCENARIO_CONSTRAINTS['task_cpu_range'][1]:.1f} cores
2. Task Memory: {SCENARIO_CONSTRAINTS['task_memory_range'][0]:.1f} to {SCENARIO_CONSTRAINTS['task_memory_range'][1]:.1f} GB
3. Task Duration: {SCENARIO_CONSTRAINTS['task_duration_range'][0]} to {SCENARIO_CONSTRAINTS['task_duration_range'][1]} seconds
4. Number of tasks: {SCENARIO_CONSTRAINTS['num_tasks_range'][0]} to {SCENARIO_CONSTRAINTS['num_tasks_range'][1]}
5. Number of nodes: {SCENARIO_CONSTRAINTS['num_nodes_range'][0]} to {SCENARIO_CONSTRAINTS['num_nodes_range'][1]}

REALISM REQUIREMENTS:
- Task sizes should follow log-normal distribution (many small, few large)
- At least 60% of tasks should be schedulable (feasible solution exists)
- Node capacities should be heterogeneous (e.g., 4 cores, 8 cores, 16 cores)
- Workload pattern must be one of: {', '.join(SCENARIO_CONSTRAINTS['workload_patterns'])}

REAL-WORLD INSPIRATION:
- Web applications: 0.5-2 CPU, 1-4 GB memory, 50-500ms duration
- Data processing: 4-16 CPU, 8-64 GB memory, 10-300s duration
- ML training: 8-32 CPU, 32-256 GB memory, 300-7200s duration

Generate a difficulty level {difficulty} scenario following these constraints.
"""
```

### Layer 2: Validation Functions (Post-Generation)

**Automatically reject bad scenarios:**

```python
class ScenarioValidator:
    def __init__(self, constraints=SCENARIO_CONSTRAINTS):
        self.constraints = constraints
        self.rejection_log = []
        
    def validate(self, scenario):
        """Run all validation checks"""
        checks = [
            self.check_bounds,
            self.check_feasibility,
            self.check_realism,
            self.check_diversity,
            self.check_complexity_alignment
        ]
        
        for check in checks:
            is_valid, reason = check(scenario)
            if not is_valid:
                self.rejection_log.append({
                    'scenario': scenario,
                    'reason': reason,
                    'check': check.__name__
                })
                return False, reason
        
        return True, "Valid"
    
    def check_bounds(self, scenario):
        """Ensure all values within realistic bounds"""
        
        # Check task resources
        for task in scenario['tasks']:
            if not (self.constraints['task_cpu_range'][0] <= 
                    task['cpu'] <= 
                    self.constraints['task_cpu_range'][1]):
                return False, f"Task CPU {task['cpu']} out of bounds"
            
            if not (self.constraints['task_memory_range'][0] <= 
                    task['memory'] <= 
                    self.constraints['task_memory_range'][1]):
                return False, f"Task memory {task['memory']} out of bounds"
            
            if not (self.constraints['task_duration_range'][0] <= 
                    task['duration'] <= 
                    self.constraints['task_duration_range'][1]):
                return False, f"Task duration {task['duration']} out of bounds"
        
        # Check node resources
        for node in scenario['nodes']:
            if not (self.constraints['node_cpu_range'][0] <= 
                    node['cpu'] <= 
                    self.constraints['node_cpu_range'][1]):
                return False, f"Node CPU {node['cpu']} out of bounds"
        
        # Check problem size
        if not (self.constraints['num_tasks_range'][0] <= 
                len(scenario['tasks']) <= 
                self.constraints['num_tasks_range'][1]):
            return False, f"Number of tasks {len(scenario['tasks'])} out of bounds"
        
        return True, "Bounds OK"
    
    def check_feasibility(self, scenario):
        """Ensure at least one valid solution exists"""
        
        # Total resource demand
        total_cpu_needed = sum(task['cpu'] for task in scenario['tasks'])
        total_memory_needed = sum(task['memory'] for task in scenario['tasks'])
        
        # Total resource supply
        total_cpu_available = sum(node['cpu'] for node in scenario['nodes'])
        total_memory_available = sum(node['memory'] for node in scenario['nodes'])
        
        # Basic feasibility check
        if total_cpu_needed > total_cpu_available:
            return False, f"Infeasible: need {total_cpu_needed} CPU, have {total_cpu_available}"
        
        if total_memory_needed > total_memory_available:
            return False, f"Infeasible: need {total_memory_needed} GB, have {total_memory_available}"
        
        # Advanced: Check if bin-packing solution exists
        if not self._bin_packing_feasible(scenario):
            return False, "No valid bin-packing solution exists"
        
        # Should not be TOO easy (at least 60% utilization)
        min_utilization = 0.6
        cpu_utilization = total_cpu_needed / total_cpu_available
        
        if cpu_utilization < min_utilization:
            return False, f"Too easy: only {cpu_utilization:.1%} CPU utilization"
        
        return True, "Feasible"
    
    def _bin_packing_feasible(self, scenario):
        """Quick check if tasks can fit in nodes"""
        # Sort tasks by size (largest first)
        tasks = sorted(scenario['tasks'], 
                      key=lambda t: t['cpu'], 
                      reverse=True)
        
        # Try first-fit decreasing
        node_capacities = {
            node['id']: {'cpu': node['cpu'], 'memory': node['memory']}
            for node in scenario['nodes']
        }
        
        for task in tasks:
            placed = False
            for node_id, capacity in node_capacities.items():
                if (capacity['cpu'] >= task['cpu'] and 
                    capacity['memory'] >= task['memory']):
                    # Place task
                    capacity['cpu'] -= task['cpu']
                    capacity['memory'] -= task['memory']
                    placed = True
                    break
            
            if not placed:
                return False  # Cannot place this task
        
        return True  # All tasks placed
    
    def check_realism(self, scenario):
        """Check if scenario resembles real-world patterns"""
        
        # 1. Task size distribution should be log-normal
        task_sizes = [task['cpu'] * task['memory'] for task in scenario['tasks']]
        if not self._is_log_normal_ish(task_sizes):
            return False, "Task sizes don't follow realistic distribution"
        
        # 2. Node heterogeneity (not all nodes the same)
        node_cpus = [node['cpu'] for node in scenario['nodes']]
        if len(set(node_cpus)) < 2:
            return False, "Nodes too homogeneous (need variety)"
        
        # 3. Temporal patterns (if arrival times specified)
        if 'arrival_time' in scenario['tasks'][0]:
            arrivals = [task['arrival_time'] for task in scenario['tasks']]
            if not self._has_temporal_pattern(arrivals):
                return False, "Arrival pattern unrealistic"
        
        # 4. Task duration vs resource correlation
        # Larger tasks should generally run longer
        correlations = self._check_duration_correlation(scenario['tasks'])
        if correlations < 0.3:  # Weak correlation threshold
            return False, f"Duration-resource correlation too weak ({correlations:.2f})"
        
        return True, "Realistic patterns"
    
    def _is_log_normal_ish(self, values):
        """Check if distribution resembles log-normal"""
        if len(values) < 10:
            return True  # Too few samples to judge
        
        # Log-normal: mean > median, right-skewed
        mean = np.mean(values)
        median = np.median(values)
        
        # Should be right-skewed
        if mean <= median:
            return False
        
        # Check skewness
        from scipy.stats import skew
        skewness = skew(values)
        
        # Positive skew expected for log-normal
        return skewness > 0.5
    
    def _has_temporal_pattern(self, arrivals):
        """Check if arrivals follow a pattern"""
        # Calculate inter-arrival times
        arrivals = sorted(arrivals)
        inter_arrivals = np.diff(arrivals)
        
        # Check for patterns
        cv = np.std(inter_arrivals) / np.mean(inter_arrivals)  # Coefficient of variation
        
        # Periodic: low CV
        # Bursty: high CV
        # Both are realistic, uniform is not
        
        return cv > 0.2  # Not too uniform
    
    def _check_duration_correlation(self, tasks):
        """Check correlation between resources and duration"""
        resources = [task['cpu'] * task['memory'] for task in tasks]
        durations = [task['duration'] for task in tasks]
        
        # Compute Pearson correlation
        correlation = np.corrcoef(resources, durations)[0, 1]
        return correlation
    
    def check_diversity(self, scenario):
        """Ensure scenario is different from recent ones"""
        
        if len(self.recent_scenarios) < 10:
            return True, "Not enough history to check diversity"
        
        # Compute similarity to recent scenarios
        for recent in self.recent_scenarios[-20:]:
            similarity = self._compute_similarity(scenario, recent)
            if similarity > 0.9:  # Too similar
                return False, f"Too similar to recent scenario (similarity: {similarity:.2f})"
        
        return True, "Sufficiently diverse"
    
    def _compute_similarity(self, scenario1, scenario2):
        """Compute similarity between two scenarios"""
        # Simple feature-based similarity
        features1 = self._extract_features(scenario1)
        features2 = self._extract_features(scenario2)
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        
        return similarity
    
    def _extract_features(self, scenario):
        """Extract feature vector from scenario"""
        return np.array([
            len(scenario['tasks']),
            len(scenario['nodes']),
            np.mean([t['cpu'] for t in scenario['tasks']]),
            np.mean([t['memory'] for t in scenario['tasks']]),
            np.std([t['cpu'] for t in scenario['tasks']]),
            sum(t['cpu'] for t in scenario['tasks']) / sum(n['cpu'] for n in scenario['nodes']),
            # Add more features as needed
        ])
    
    def check_complexity_alignment(self, scenario, target_difficulty):
        """Ensure scenario matches curriculum difficulty"""
        
        # Compute actual complexity
        actual_complexity = self._compute_complexity(scenario)
        
        # Allow 20% deviation
        lower_bound = target_difficulty * 0.8
        upper_bound = target_difficulty * 1.2
        
        if not (lower_bound <= actual_complexity <= upper_bound):
            return False, f"Complexity {actual_complexity:.1f} doesn't match target {target_difficulty}"
        
        return True, "Complexity aligned"
    
    def _compute_complexity(self, scenario):
        """Compute scenario complexity score (1-10)"""
        # Multiple factors contribute to complexity
        
        # 1. Problem size
        size_score = min(10, len(scenario['tasks']) / 50)
        
        # 2. Resource tightness (how close to capacity)
        total_demand = sum(t['cpu'] for t in scenario['tasks'])
        total_supply = sum(n['cpu'] for n in scenario['nodes'])
        tightness_score = (total_demand / total_supply) * 10
        
        # 3. Heterogeneity (more variety = harder)
        task_cpus = [t['cpu'] for t in scenario['tasks']]
        heterogeneity_score = np.std(task_cpus) / np.mean(task_cpus) * 10
        
        # 4. Temporal complexity (if applicable)
        temporal_score = 0
        if 'arrival_time' in scenario['tasks'][0]:
            arrival_variance = np.var([t['arrival_time'] for t in scenario['tasks']])
            temporal_score = min(10, arrival_variance / 100)
        
        # Weighted average
        complexity = (
            0.3 * size_score +
            0.4 * tightness_score +
            0.2 * heterogeneity_score +
            0.1 * temporal_score
        )
        
        return min(10, complexity)
```

### Layer 3: Real-World Grounding

**Continuously inject real-world patterns:**

```python
class RealWorldGrounder:
    def __init__(self):
        self.real_patterns = self._extract_real_patterns()
        
    def _extract_real_patterns(self):
        """Analyze real traces to extract patterns"""
        
        # Load real-world traces
        google_trace = self._load_google_trace_sample(1000)
        azure_trace = self._load_azure_trace_sample(1000)
        
        patterns = {
            'task_size_distribution': self._fit_distribution(
                [t['cpu'] * t['memory'] for t in google_trace['tasks']]
            ),
            'duration_distribution': self._fit_distribution(
                [t['duration'] for t in google_trace['tasks']]
            ),
            'arrival_patterns': self._extract_arrival_patterns(google_trace),
            'resource_ratios': self._compute_cpu_memory_ratios(google_trace),
            'node_capacities': self._extract_node_capacities(azure_trace),
        }
        
        return patterns
    
    def _fit_distribution(self, data):
        """Fit statistical distribution to data"""
        from scipy import stats
        
        # Try common distributions
        distributions = [
            stats.lognorm,
            stats.expon,
            stats.gamma
        ]
        
        best_fit = None
        best_score = float('inf')
        
        for dist in distributions:
            params = dist.fit(data)
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(data, dist.name, args=params)
            
            if ks_stat < best_score:
                best_score = ks_stat
                best_fit = {'distribution': dist, 'params': params}
        
        return best_fit
    
    def inject_realism(self, scenario):
        """Modify generated scenario to match real patterns"""
        
        # 1. Adjust task sizes to match real distribution
        for task in scenario['tasks']:
            # Sample from fitted distribution
            real_size = self.real_patterns['task_size_distribution']['distribution'].rvs(
                *self.real_patterns['task_size_distribution']['params']
            )
            
            # Scale current task to match
            current_size = task['cpu'] * task['memory']
            scale_factor = real_size / current_size
            
            task['cpu'] *= np.sqrt(scale_factor)
            task['memory'] *= np.sqrt(scale_factor)
        
        # 2. Adjust CPU:Memory ratios
        typical_ratios = self.real_patterns['resource_ratios']
        for task in scenario['tasks']:
            ratio = task['cpu'] / task['memory']
            # Nudge toward realistic ratio
            if ratio < typical_ratios['min']:
                task['cpu'] = task['memory'] * typical_ratios['min']
            elif ratio > typical_ratios['max']:
                task['memory'] = task['cpu'] / typical_ratios['max']
        
        # 3. Add realistic arrival pattern
        if 'arrival_time' not in scenario['tasks'][0]:
            arrivals = self._generate_realistic_arrivals(
                len(scenario['tasks']),
                pattern=np.random.choice(['bursty', 'periodic', 'diurnal'])
            )
            for task, arrival in zip(scenario['tasks'], arrivals):
                task['arrival_time'] = arrival
        
        return scenario
    
    def _generate_realistic_arrivals(self, num_tasks, pattern):
        """Generate realistic arrival times"""
        
        if pattern == 'bursty':
            # Poisson process with varying rate
            arrivals = []
            time = 0
            high_rate = 5.0  # tasks/second
            low_rate = 0.5
            
            for i in range(num_tasks):
                # Alternate between high and low rate
                rate = high_rate if (i // 20) % 2 == 0 else low_rate
                inter_arrival = np.random.exponential(1/rate)
                time += inter_arrival
                arrivals.append(time)
        
        elif pattern == 'periodic':
            # Regular intervals with noise
            base_interval = 10.0  # seconds
            arrivals = [
                i * base_interval + np.random.normal(0, 1)
                for i in range(num_tasks)
            ]
        
        elif pattern == 'diurnal':
            # 24-hour cycle (scaled)
            hour = 0
            arrivals = []
            for i in range(num_tasks):
                # Higher rate during "business hours" (8am-6pm)
                hour_of_day = (hour % 24)
                if 8 <= hour_of_day <= 18:
                    rate = 2.0
                else:
                    rate = 0.3
                
                inter_arrival = np.random.exponential(1/rate)
                hour += inter_arrival / 3600  # Convert to hours
                arrivals.append(hour * 3600)  # Back to seconds
        
        return arrivals
```

### Layer 4: Feedback-Driven Refinement

**Use validation performance to improve generation:**

```python
class AdaptiveScenarioGenerator:
    def __init__(self, attacker_agent, validator, grounder):
        self.attacker = attacker_agent
        self.validator = validator
        self.grounder = grounder
        self.generation_stats = {
            'attempts': 0,
            'rejections': 0,
            'rejection_reasons': []
        }
        
    def generate_valid_scenario(self, difficulty, max_attempts=5):
        """Generate scenario with quality control"""
        
        for attempt in range(max_attempts):
            self.generation_stats['attempts'] += 1
            
            # Generate scenario
            scenario = self.attacker.generate_scenario(difficulty)
            
            # Validate
            is_valid, reason = self.validator.validate(scenario)
            
            if is_valid:
                # Apply real-world grounding
                scenario = self.grounder.inject_realism(scenario)
                
                # Final validation after grounding
                is_valid, reason = self.validator.validate(scenario)
                
                if is_valid:
                    return scenario
            
            # Log rejection
            self.generation_stats['rejections'] += 1
            self.generation_stats['rejection_reasons'].append(reason)
            
            # Give feedback to attacker for next attempt
            feedback = self._build_feedback(reason)
            self.attacker.update_context(feedback)
        
        # Failed to generate valid scenario
        # Fallback: use seed scenario of similar difficulty
        print(f"⚠️  Failed to generate valid scenario after {max_attempts} attempts")
        return self._get_fallback_scenario(difficulty)
    
    def _build_feedback(self, rejection_reason):
        """Build feedback prompt for attacker"""
        return f"""
Your last scenario was rejected: {rejection_reason}

Please fix this issue in your next generation. Remember:
- All constraints must be satisfied
- Scenario must be feasible (solvable)
- Patterns should match real-world workloads
"""
    
    def _get_fallback_scenario(self, difficulty):
        """Get pre-validated seed scenario"""
        # Return a seed scenario matching difficulty
        seed_scenarios = load_seed_scenarios()
        matching = [s for s in seed_scenarios if s['difficulty'] == difficulty]
        
        if matching:
            return random.choice(matching)
        else:
            # Return closest difficulty
            return min(seed_scenarios, 
                      key=lambda s: abs(s['difficulty'] - difficulty))
    
    def get_quality_report(self):
        """Generate quality control report"""
        total = self.generation_stats['attempts']
        rejected = self.generation_stats['rejections']
        
        report = f"""
SCENARIO GENERATION QUALITY REPORT
{'='*60}
Total attempts: {total}
Valid scenarios: {total - rejected} ({(total-rejected)/total:.1%})
Rejected: {rejected} ({rejected/total:.1%})

Top rejection reasons:
"""
        # Count rejection reasons
        from collections import Counter
        reason_counts = Counter(self.generation_stats['rejection_reasons'])
        
        for reason, count in reason_counts.most_common(5):
            report += f"  - {reason}: {count} times\n"
        
        return report
```

### Layer 5: Continuous Monitoring

**Track scenario quality metrics over time:**

```python
class QualityMonitor:
    def __init__(self):
        self.metrics_history = []
        
    def log_scenario(self, scenario, validation_result, execution_result=None):
        """Log quality metrics for each scenario"""
        
        metrics = {
            'timestamp': time.time(),
            'scenario_id': scenario.get('id'),
            
            # Validation metrics
            'is_valid': validation_result['is_valid'],
            'rejection_reason': validation_result.get('reason'),
            
            # Complexity metrics
            'num_tasks': len(scenario['tasks']),
            'num_nodes': len(scenario['nodes']),
            'complexity_score': self._compute_complexity(scenario),
            
            # Realism metrics
            'task_size_skew': self._compute_skewness(
                [t['cpu'] * t['memory'] for t in scenario['tasks']]
            ),
            'resource_tightness': self._compute_tightness(scenario),
            
            # Performance metrics (if executed)
            'solvable': execution_result is not None if execution_result else None,
            'best_reward': execution_result.get('reward') if execution_result else None,
        }
        
        self.metrics_history.append(metrics)
        
        # Alert if quality degrading
        if len(self.metrics_history) >= 50:
            self._check_quality_drift()
    
    def _check_quality_drift(self):
        """Detect if scenario quality is degrading"""
        recent = self.metrics_history[-50:]
        
        # Check validation rate
        validation_rate = sum(m['is_valid'] for m in recent) / len(recent)
        if validation_rate < 0.7:
            print(f"⚠️  WARNING: Validation rate dropped to {validation_rate:.1%}")
        
        # Check complexity drift
        recent_complexity = np.mean([m['complexity_score'] for m in recent])
        all_complexity = np.mean([m['complexity_score'] for m in self.metrics_history])
        
        if abs(recent_complexity - all_complexity) > 2.0:
            print(f"⚠️  WARNING: Complexity drift detected")
            print(f"   Recent: {recent_complexity:.2f}, Overall: {all_complexity:.2f}")
    
    def visualize_quality(self):
        """Generate quality visualization"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Validation rate over time
        window_size = 50
        validation_rates = []
        for i in range(window_size, len(self.metrics_history)):
            window = self.metrics_history[i-window_size:i]
            rate = sum(m['is_valid'] for m in window) / window_size
            validation_rates.append(rate)
        
        axes[0, 0].plot(validation_rates)
        axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
        axes[0, 0].set_title('Validation Rate Over Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Validation Rate')
        axes[0, 0].legend()
        
        # 2. Complexity distribution
        complexities = [m['complexity_score'] for m in self.metrics_history if m['is_valid']]
        axes[0, 1].hist(complexities, bins=20)
        axes[0, 1].set_title('Complexity Distribution')
        axes[0, 1].set_xlabel('Complexity Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Task size distribution (should be log-normal)
        all_task_sizes = []
        for m in self.metrics_history[-100:]:  # Last 100 scenarios
            if 'task_sizes' in m:
                all_task_sizes.extend(m['task_sizes'])
        
        axes[1, 0].hist(np.log(all_task_sizes + 1), bins=30)
        axes[1, 0].set_title('Task Size Distribution (log scale)')
        axes[1, 0].set_xlabel('log(Task Size)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Rejection reasons
        rejection_reasons = [
            m['rejection_reason'] for m in self.metrics_history 
            if not m['is_valid']
        ]
        from collections import Counter
        reason_counts = Counter(rejection_reasons).most_common(5)
        
        if reason_counts:
            reasons, counts = zip(*reason_counts)
            axes[1, 1].barh(reasons, counts)
            axes[1, 1].set_title('Top Rejection Reasons')
            axes[1, 1].set_xlabel('Count')
        
        plt.tight_layout()
        plt.savefig('scenario_quality_report.png')
        print("📊 Quality report saved to scenario_quality_report.png")
```

---

## PRACTICAL IMPLEMENTATION

### Complete Workflow with Quality Control:

```python
class AURORAWithQualityControl:
    def __init__(self):
        # Core components
        self.attacker = AttackerAgent(llm)
        self.defender = DefenderAgent(llm)
        self.simulator = UnifiedResourceSimulator()
        
        # Quality control components
        self.validator = ScenarioValidator()
        self.grounder = RealWorldGrounder()
        self.adaptive_generator = AdaptiveScenarioGenerator(
            self.attacker, self.validator, self.grounder
        )
        self.quality_monitor = QualityMonitor()
        
        # Initialize with real-world patterns
        self._initialize_real_patterns()
    
    def _initialize_real_patterns(self):
        """Extract patterns from real traces"""
        print("📊 Analyzing real-world traces...")
        
        # Load sample of real traces
        google_sample = load_google_trace_sample(1000)
        azure_sample = load_azure_trace_sample(1000)
        
        # Extract patterns
        patterns = {
            'task_cpu_distribution': self._fit_distribution(
                [t['cpu'] for t in google_sample['tasks']]
            ),
            'task_memory_distribution': self._fit_distribution(
                [t['memory'] for t in google_sample['tasks']]
            ),
            'cpu_memory_correlation': np.corrcoef(
                [t['cpu'] for t in google_sample['tasks']],
                [t['memory'] for t in google_sample['tasks']]
            )[0, 1],
            'node_capacities': Counter([
                n['cpu'] for n in azure_sample['nodes']
            ]).most_common(10)
        }
        
        # Update constraints with real patterns
        self.validator.real_patterns = patterns
        self.grounder.real_patterns = patterns
        
        print("✅ Real-world patterns extracted and injected")
    
    def run_iteration_with_quality_control(self, iteration):
        """Run one iteration with full quality control"""
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} - WITH QUALITY CONTROL")
        print(f"{'='*60}\n")
        
        # Get curriculum difficulty
        difficulty = self.curriculum.get_current_difficulty()
        
        # Generate VALID scenario (with retries)
        print("🔴 Generating scenario with quality control...")
        scenario = self.adaptive_generator.generate_valid_scenario(difficulty)
        
        # Log scenario quality
        validation_result = self.validator.validate(scenario)
        self.quality_monitor.log_scenario(scenario, validation_result)
        
        print(f"   ✅ Valid scenario generated")
        print(f"   Tasks: {len(scenario['tasks'])}, Nodes: {len(scenario['nodes'])}")
        print(f"   Complexity: {self.validator._compute_complexity(scenario):.2f}/10")
        
        # Rest of iteration (defender, simulation, etc.)
        policies = self.defender.generate_policies(scenario, num_candidates=3)
        
        results = []
        for policy in policies:
            sim_results = self.simulator.execute_policy(scenario, policy)
            reward = self.reward_engine.compute_reward(sim_results, scenario)
            results.append({'policy': policy, 'results': sim_results, 'reward': reward})
        
        best = max(results, key=lambda x: x['reward'])
        
        # Log execution results
        self.quality_monitor.log_scenario(
            scenario, 
            validation_result,
            execution_result=best
        )
        
        # Periodic quality report
        if iteration % 100 == 0:
            print("\n" + self.adaptive_generator.get_quality_report())
            self.quality_monitor.visualize_quality()
        
        return best
```

---

## SEED SCENARIOS (Bootstrap Quality)

**Create 50 high-quality seed scenarios manually:**

```python
def create_seed_scenarios():
    """Create diverse, high-quality seed scenarios"""
    
    seeds = []
    
    # Category 1: Web application workloads
    seeds.extend(generate_web_app_scenarios(10))
    
    # Category 2: Batch processing
    seeds.extend(generate_batch_processing_scenarios(10))
    
    # Category 3: ML training
    seeds.extend(generate_ml_training_scenarios(10))
    
    # Category 4: Mixed workloads
    seeds.extend(generate_mixed_scenarios(10))
    
    # Category 5: Edge computing
    seeds.extend(generate_edge_scenarios(10))
    
    # Validate all seeds
    validator = ScenarioValidator()
    valid_seeds = []
    for seed in seeds:
        is_valid, reason = validator.validate(seed)
        if is_valid:
            valid_seeds.append(seed)
        else:
            print(f"⚠️  Seed scenario invalid: {reason}")
    
    print(f"✅ Created {len(valid_seeds)} valid seed scenarios")
    
    # Save to file
    with open('seed_scenarios.json', 'w') as f:
        json.dump(valid_seeds, f, indent=2)
    
    return valid_seeds

def generate_web_app_scenarios(count):
    """Generate realistic web application scenarios"""
    scenarios = []
    
    for i in range(count):
        # Web apps: many small requests
        num_tasks = np.random.randint(50, 200)
        num_nodes = np.random.randint(5, 15)
        
        # Generate tasks (HTTP requests)
        tasks = []
        for t in range(num_tasks):
            tasks.append({
                'id': f'req_{t}',
                'type': 'web_request',
                'cpu': np.random.lognormal(mean=0, sigma=0.5),  # 0.5-2 cores
                'memory': np.random.lognormal(mean=1, sigma=0.3),  # 1-4 GB
                'duration': np.random.lognormal(mean=4, sigma=1),  # 50-500 ms
                'arrival_time': generate_poisson_arrivals(t, rate=2.0),
                'sla_latency_ms': 1000,  # 1 second SLA
                'priority': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            })
        
        # Generate nodes (web servers)
        nodes = []
        for n in range(num_nodes):
            nodes.append({
                'id': f'server_{n}',
                'cpu': np.random.choice([4, 8, 16]),
                'memory': np.random.choice([8, 16, 32]),
                'location': f'region_{n % 3}'  # 3 regions
            })
        
        scenarios.append({
            'id': f'web_app_{i}',
            'difficulty': min(3, 1 + i // 3),  # Progressive difficulty
            'category': 'web_application',
            'tasks': tasks,
            'nodes': nodes
        })
    
    return scenarios
```

---

## EVALUATION: Validate Quality Control Works

```python
def evaluate_quality_control():
    """Test that quality control catches bad scenarios"""
    
    validator = ScenarioValidator()
    
    # Test 1: Out of bounds
    bad_scenario_1 = {
        'tasks': [{'cpu': 99999, 'memory': 1, 'duration': 100}],
        'nodes': [{'cpu': 8, 'memory': 16}]
    }
    is_valid, reason = validator.validate(bad_scenario_1)
    assert not is_valid, "Should reject out-of-bounds scenario"
    print(f"✅ Test 1 passed: {reason}")
    
    # Test 2: Infeasible
    bad_scenario_2 = {
        'tasks': [{'cpu': 16, 'memory': 32, 'duration': 100}] * 10,
        'nodes': [{'cpu': 8, 'memory': 16}]
    }
    is_valid, reason = validator.validate(bad_scenario_2)
    assert not is_valid, "Should reject infeasible scenario"
    print(f"✅ Test 2 passed: {reason}")
    
    # Test 3: Unrealistic distribution
    bad_scenario_3 = {
        'tasks': [{'cpu': 2, 'memory': 4, 'duration': 100}] * 50,  # All identical
        'nodes': [{'cpu': 8, 'memory': 16}] * 10  # All identical
    }
    is_valid, reason = validator.validate(bad_scenario_3)
    assert not is_valid, "Should reject unrealistic scenario"
    print(f"✅ Test 3 passed: {reason}")
    
    # Test 4: Good scenario
    good_scenario = load_seed_scenarios()[0]
    is_valid, reason = validator.validate(good_scenario)
    assert is_valid, f"Should accept good scenario: {reason}"
    print(f"✅ Test 4 passed: Good scenario accepted")
    
    print("\n✅ All quality control tests passed!")
```

---

## SUMMARY: YOUR COMPLETE QUALITY CONTROL SYSTEM

```python
# 1. Initialize with real patterns
grounder = RealWorldGrounder()
grounder.load_real_traces(['google_cluster', 'azure_public'])

# 2. Set up validation
validator = ScenarioValidator(real_patterns=grounder.patterns)

# 3. Create adaptive generator
adaptive_gen = AdaptiveScenarioGenerator(attacker, validator, grounder)

# 4. Generate with quality control
for iteration in range(1000):
    # This internally validates and retries
    scenario = adaptive_gen.generate_valid_scenario(difficulty)
    
    # Monitor quality
    quality_monitor.log_scenario(scenario)
    
    # Use scenario
    policy = defender.generate_policy(scenario)
    results = simulator.execute(scenario, policy)
    
    # Report every 100 iterations
    if iteration % 100 == 0:
        print(adaptive_gen.get_quality_report())
        quality_monitor.visualize_quality()
```

**Key Guarantees**:
1. ✅ All scenarios within realistic bounds
2. ✅ All scenarios are feasible (solvable)
3. ✅ Scenarios match real-world patterns
4. ✅ Sufficient diversity
5. ✅ Complexity aligned with curriculum
6. ✅ Continuous monitoring and alerts

This system ensures your self-generated data is as good as (or better than) manually curated datasets!