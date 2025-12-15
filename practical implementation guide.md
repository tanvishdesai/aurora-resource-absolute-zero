# Implementation Q&A: Practical Guide for AURORA

Let me answer all your questions with concrete, actionable details.

---

## 1. MODEL CHOICE: Qwen vs Gemini

**Short answer**: **Use Gemini 2.5 Flash through API** - it's significantly smarter and you won't need fine-tuning.

**Detailed reasoning**:

### Why Gemini 2.5 Flash > Qwen 7B/8B:

**Qwen 7B/8B limitations**:
- Will struggle with complex reasoning chains needed for policy generation
- Poor structured output (JSON) adherence
- Weaker at following multi-step instructions
- You'd likely need 100+ iterations just to get decent policies

**Gemini 2.5 Flash advantages**:
- Much better reasoning capability
- Excellent JSON mode (structured outputs)
- Faster inference (API is optimized)
- Better instruction following

**Rate limit math**:
```python
# Per iteration:
# - 5 scenarios (Attacker) = 5 requests
# - 5 scenarios × 3 policies (Defender) = 15 requests
# Total per iteration: 20 requests

# With 5000/day limit:
5000 / 20 = 250 iterations per day

# 1000 iterations = 4 days with 1 API key
# With multiple keys: 1-2 days total
```

**Recommendation**: 
```python
# Use hybrid approach if needed
class ModelManager:
    def __init__(self):
        self.gemini_keys = [
            "key1", "key2", "key3"  # Your multiple keys
        ]
        self.current_key_idx = 0
        self.daily_counts = {}
        
    def get_client(self):
        """Rotate through API keys when hitting limits"""
        key = self.gemini_keys[self.current_key_idx]
        
        if self.daily_counts.get(key, 0) >= 4800:  # Buffer
            self.current_key_idx = (self.current_key_idx + 1) % len(self.gemini_keys)
            key = self.gemini_keys[self.current_key_idx]
        
        return genai.GenerativeModel('gemini-2.5-flash', api_key=key)
```

---

## 2. DATASET REQUIREMENTS

**Important**: You **DON'T need traditional ML datasets**. The system generates its own training scenarios.

**What you DO need**:

### A. Seed Scenarios (for initial curriculum):
```python
# Create 20-30 handcrafted scenarios across difficulty levels
seed_scenarios = {
    'easy': [
        {
            'tasks': [
                {'id': 't1', 'cpu': 2, 'memory': 4, 'duration': 100},
                {'id': 't2', 'cpu': 1, 'memory': 2, 'duration': 50}
            ],
            'nodes': [
                {'id': 'n1', 'cpu': 8, 'memory': 16},
                {'id': 'n2', 'cpu': 4, 'memory': 8}
            ]
        },
        # ... 5-10 more easy scenarios
    ],
    'medium': [...],
    'hard': [...]
}
```

### B. Real-World Traces (for final testing only):

**Google Cluster Trace**:
```python
# Download from: https://github.com/google/cluster-data
# Use only for validation, NOT training

def load_google_trace(sample_size=100):
    """Load subset for testing"""
    import pandas as pd
    
    # Task events file
    df = pd.read_csv('task_events-000000000000.csv.gz', 
                     nrows=10000)  # Sample
    
    # Convert to our format
    scenarios = []
    for timestamp in df['time'].unique()[:sample_size]:
        tasks_at_time = df[df['time'] == timestamp]
        
        scenario = {
            'timestamp': timestamp,
            'tasks': [
                {
                    'cpu': row['cpu_request'],
                    'memory': row['memory_request'],
                    'duration': row['duration']
                }
                for _, row in tasks_at_time.iterrows()
            ]
        }
        scenarios.append(scenario)
    
    return scenarios
```

**Azure Public Dataset 2021**:
```python
# Download: https://github.com/Azure/AzurePublicDataset
# Use for validation only

def load_azure_trace():
    """Azure VM traces"""
    import pandas as pd
    
    df = pd.read_csv('vmtable.csv', nrows=1000)
    
    return convert_to_scenarios(df)
```

### C. Synthetic Scenario Generator (backup):
```python
def generate_synthetic_scenarios(num_scenarios=100):
    """If you can't get real traces"""
    scenarios = []
    
    for i in range(num_scenarios):
        num_tasks = np.random.randint(10, 100)
        num_nodes = np.random.randint(5, 20)
        
        tasks = []
        for t in range(num_tasks):
            tasks.append({
                'id': f't{t}',
                'cpu': np.random.uniform(0.5, 4),
                'memory': np.random.uniform(1, 16),
                'duration': np.random.uniform(10, 1000),
                'arrival_time': np.random.uniform(0, 500)
            })
        
        nodes = []
        for n in range(num_nodes):
            nodes.append({
                'id': f'n{n}',
                'cpu': np.random.choice([4, 8, 16, 32]),
                'memory': np.random.choice([8, 16, 32, 64])
            })
        
        scenarios.append({'tasks': tasks, 'nodes': nodes})
    
    return scenarios
```

**You only need ~50 seed scenarios total**. The system generates 1000s more during self-play.

---

## 3. END-TO-END WORKFLOW EXPLANATION

Let me break down exactly how one iteration works:

### Iteration Flow Diagram:

```
START ITERATION 1
│
├─> [Curriculum Manager] 
│   └─> "Start at difficulty level 1 (easy)"
│
├─> [Attacker Agent (Gemini)] 
│   ├─> Input: difficulty=1, past_scenarios=[], defender_stats=None
│   ├─> Prompt: "Generate a simple resource allocation scenario..."
│   └─> Output: scenario_1.json
│       {
│         "tasks": [
│           {"id": "t1", "cpu": 2, "memory": 4, "duration": 100},
│           {"id": "t2", "cpu": 1, "memory": 2, "duration": 50}
│         ],
│         "nodes": [
│           {"id": "n1", "cpu": 8, "memory": 16},
│           {"id": "n2", "cpu": 4, "memory": 8}
│         ]
│       }
│
├─> [Defender Agent (Gemini)] 
│   ├─> Input: scenario_1.json
│   ├─> Prompt: "Generate allocation policy for this scenario..."
│   └─> Output: policy_1.json (generates 3 candidates)
│       {
│         "allocations": [
│           {"task": "t1", "node": "n1", "reasoning": "..."},
│           {"task": "t2", "node": "n2", "reasoning": "..."}
│         ]
│       }
│
├─> [Simulator] 
│   ├─> Execute policy_1 in scenario_1
│   ├─> Track: task_completion_times, resource_usage, violations
│   └─> Output: results_1.json
│       {
│         "avg_latency": 145.3,
│         "utilization": 0.73,
│         "cost": 0.15,
│         "sla_violations": 0
│       }
│
├─> [Reward Engine]
│   ├─> Compute multi-objective reward
│   └─> defender_reward = 0.82, attacker_reward = 0.65
│
├─> [Policy Database]
│   └─> Store policy_1 with ELO rating = 1500 (initial)
│
└─> [Curriculum Check]
    └─> Success rate < 70% → Stay at level 1

ITERATION 1 COMPLETE
```

### Code Implementation:

```python
def run_one_iteration(iteration_num):
    """Complete flow for one iteration"""
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*60}\n")
    
    # 1. Get current curriculum level
    difficulty = curriculum.get_current_difficulty()
    print(f"📚 Difficulty Level: {difficulty['name']} (level {difficulty['difficulty']})")
    
    # 2. Attacker generates scenario
    print("\n🔴 Attacker generating scenario...")
    scenario = attacker_agent.generate_scenario(
        difficulty=difficulty,
        defender_stats=get_recent_defender_stats()
    )
    print(f"   Generated: {scenario['num_tasks']} tasks, {scenario['num_nodes']} nodes")
    
    # 3. Defender generates multiple policy candidates
    print("\n🔵 Defender generating policies...")
    policies = defender_agent.generate_policies(
        scenario=scenario,
        num_candidates=3
    )
    print(f"   Generated {len(policies)} policy candidates")
    
    # 4. Execute and evaluate each policy
    print("\n⚙️  Simulating policies...")
    results = []
    for i, policy in enumerate(policies):
        print(f"   Testing policy {i+1}/3...", end="")
        
        # Run simulation
        sim_results = simulator.execute_policy(scenario, policy)
        
        # Compute reward
        reward, scores = reward_engine.compute_reward(sim_results, scenario)
        
        print(f" Reward: {reward:.3f}")
        
        results.append({
            'policy': policy,
            'results': sim_results,
            'reward': reward,
            'scores': scores
        })
    
    # 5. Select best policy
    best = max(results, key=lambda x: x['reward'])
    print(f"\n🏆 Best policy reward: {best['reward']:.3f}")
    
    # 6. Update policy database
    policy_db.add_policy(
        policy=best['policy'],
        performance=best['results'],
        reward=best['reward']
    )
    
    # 7. Update curriculum if needed
    recent_rewards = get_last_n_rewards(20)
    if curriculum.should_advance(recent_rewards):
        curriculum.advance()
        print(f"\n🎓 ADVANCED to {curriculum.get_current_difficulty()['name']}!")
    
    # 8. Return summary
    return {
        'iteration': iteration_num,
        'reward': best['reward'],
        'latency': best['results']['avg_latency'],
        'utilization': best['results']['utilization'],
        'curriculum_level': curriculum.current_level
    }
```

### What Happens Over 1000 Iterations:

```
Iterations 1-100:   Learn basic allocation (single node, few tasks)
Iterations 101-300: Learn load balancing (multiple nodes)
Iterations 301-600: Learn constraint handling (SLAs, failures)
Iterations 601-900: Learn complex optimization (energy, cost)
Iterations 901-1000: Master multi-datacenter scenarios
```

---

## 4. FINE-TUNING: NOT NEEDED (and you're right about API)

**You are correct**: You cannot fine-tune Gemini through API.

**But you DON'T need to fine-tune!** Here's why:

### Traditional ML:
```
[Fixed Model] → Train on dataset → Fine-tune → Deploy
❌ Requires thousands of labeled examples
❌ Model is static after training
```

### AURORA Approach:
```
[Smart LLM] → Self-generate scenarios → Learn through prompts → Evolve
✅ Zero labeled examples needed
✅ Continuous improvement through experience
✅ Knowledge stored in Policy Database, not model weights
```

### How Learning Happens WITHOUT Fine-tuning:

**Learning Mechanism: In-Context Learning + Policy Database**

```python
class DefenderAgent:
    def generate_policy(self, scenario):
        # Retrieve similar successful policies
        similar_successes = policy_db.get_top_k_similar(scenario, k=5)
        
        # Build prompt with examples (in-context learning)
        prompt = f"""
        You are an expert resource allocator.
        
        Here are 5 similar scenarios and their successful policies:
        
        {format_examples(similar_successes)}  # This is the "learning"
        
        Now solve this new scenario:
        {scenario}
        """
        
        policy = llm.generate(prompt)
        return policy
```

**The model doesn't change, but its effective knowledge grows!**

**Visual Representation**:

```
Traditional Fine-tuning:
Knowledge stored IN model weights
│
├─> Model Size: 8GB
├─> Training: Hours/days
└─> Update: Retrain entire model

AURORA (In-Context Learning):
Knowledge stored IN policy database
│
├─> Database Size: 100MB
├─> "Training": Seconds per iteration
└─> Update: Add new entry to database
```

---

## 5. TRAINING PARADIGM: Not Traditional Epochs

You're absolutely right - this is fundamentally different!

### Traditional ML Training:

```python
# Traditional approach
for epoch in range(100):  # Fixed epochs
    for batch in dataloader:  # Fixed dataset
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss:.3f}")
# Done! Model is trained.
```

### AURORA Training:

```python
# AURORA approach
iteration = 0
while not converged():  # No fixed end point
    # Generate NEW scenario (dataset grows)
    scenario = attacker.generate()
    
    # Generate NEW policy (no backprop)
    policy = defender.generate(scenario)
    
    # Execute and learn
    results = simulator.execute(policy)
    
    # Store successful strategies
    if results.reward > threshold:
        policy_db.add(policy)
    
    # Adapt difficulty
    if success_rate > 0.7:
        curriculum.advance()
    
    iteration += 1
    
    # You decide when to stop!
    if iteration == 1000 or plateau_detected():
        break
```

### Key Differences:

| Aspect | Traditional ML | AURORA |
|--------|----------------|--------|
| **Dataset** | Fixed | Grows dynamically |
| **Epochs** | Predetermined (100) | Open-ended |
| **Stopping** | When epochs done | When performance plateaus |
| **Learning** | Gradient descent | Experience accumulation |
| **Progress** | Loss decreases | Reward increases |
| **Validation** | Held-out test set | Real-time curriculum advancement |

### How to Know When to Stop:

```python
class ConvergenceDetector:
    def __init__(self, patience=50):
        self.patience = patience
        self.best_reward = -float('inf')
        self.plateau_count = 0
        self.reward_history = []
        
    def should_stop(self, current_reward):
        self.reward_history.append(current_reward)
        
        # Check if improving
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        
        # Multiple stopping conditions
        conditions = {
            'plateau': self.plateau_count >= self.patience,
            'max_curriculum': curriculum.current_level == curriculum.max_level,
            'reward_threshold': current_reward >= 0.95,  # Near-optimal
            'diminishing_returns': self._check_diminishing_returns()
        }
        
        return any(conditions.values())
    
    def _check_diminishing_returns(self):
        """Check if last 100 iterations improved < 1%"""
        if len(self.reward_history) < 100:
            return False
        
        recent = np.mean(self.reward_history[-100:])
        previous = np.mean(self.reward_history[-200:-100])
        
        improvement = (recent - previous) / previous
        return improvement < 0.01  # Less than 1% improvement
```

### Typical Training Trajectory:

```
Iteration    Avg Reward    Curriculum    Policy DB Size    Status
-----------------------------------------------------------------
0-50         0.35          Level 1       23                Exploring
51-100       0.48          Level 1       67                Learning basics
101-200      0.62          Level 2       184               Load balancing
201-400      0.73          Level 3       412               Constraint handling
401-700      0.81          Level 4       876               Multi-objective
701-1000     0.87          Level 5       1523              Near-optimal
1000+        0.89          Level 5       2100              Plateau (STOP)
```

**You'll know you're done when**:
1. Reward plateaus for 50+ iterations
2. Reached highest curriculum level
3. Policies generalize to test scenarios
4. Diminishing returns on new iterations

---

## 6. EVALUATION METRICS

### Primary Metrics (Multi-Objective):

```python
class EvaluationMetrics:
    def compute_all(self, results, scenario):
        return {
            # Performance
            'avg_latency_ms': self._compute_latency(results),
            'p95_latency_ms': self._compute_percentile_latency(results, 95),
            'max_latency_ms': max(results.task_latencies),
            
            # Resource efficiency
            'cpu_utilization': self._compute_utilization(results, 'cpu'),
            'memory_utilization': self._compute_utilization(results, 'memory'),
            'resource_wastage': self._compute_wastage(results),
            
            # Cost
            'total_cost_usd': self._compute_cost(results),
            'cost_per_task': self._compute_cost(results) / len(results.tasks),
            
            # Energy
            'energy_kwh': self._compute_energy(results),
            'carbon_emissions_kg': self._compute_carbon(results),
            
            # Reliability
            'sla_violation_rate': self._compute_sla_violations(results, scenario),
            'task_failure_rate': self._compute_failures(results),
            
            # Quality
            'makespan': self._compute_makespan(results),  # Total completion time
            'fairness': self._compute_fairness(results)   # Gini coefficient
        }
    
    def _compute_latency(self, results):
        """Average task completion time"""
        return np.mean([
            r.completion_time - r.arrival_time 
            for r in results.task_completions
        ])
    
    def _compute_utilization(self, results, resource_type):
        """Average resource utilization"""
        total_capacity = sum(node.capacity for node in results.nodes)
        total_used = sum(node.used for node in results.nodes)
        return total_used / total_capacity
    
    def _compute_sla_violations(self, results, scenario):
        """Percentage of tasks violating SLA"""
        violations = 0
        for task in results.task_completions:
            sla = scenario['tasks'][task.id].get('sla_latency_ms')
            if sla and task.latency > sla:
                violations += 1
        return violations / len(results.task_completions)
```

### Aggregate Metrics (for Paper):

```python
def compute_aggregate_metrics(all_results):
    """Metrics to report in paper"""
    return {
        # Performance
        'mean_latency': np.mean([r['avg_latency_ms'] for r in all_results]),
        'std_latency': np.std([r['avg_latency_ms'] for r in all_results]),
        
        # Efficiency
        'mean_utilization': np.mean([r['cpu_utilization'] for r in all_results]),
        
        # Cost-effectiveness
        'mean_cost_per_task': np.mean([r['cost_per_task'] for r in all_results]),
        
        # Reliability
        'sla_success_rate': 1 - np.mean([r['sla_violation_rate'] for r in all_results]),
        
        # Pareto optimality
        'pareto_optimal_percentage': compute_pareto_percentage(all_results)
    }
```

---

## 7. BASELINES FOR COMPARISON

You must compare against these:

### A. Simple Heuristics:

```python
class RandomAllocation:
    """Baseline 1: Random assignment"""
    def allocate(self, scenario):
        allocations = []
        for task in scenario['tasks']:
            node = random.choice(scenario['nodes'])
            allocations.append({'task': task['id'], 'node': node['id']})
        return allocations

class GreedyFirstFit:
    """Baseline 2: First-fit greedy"""
    def allocate(self, scenario):
        allocations = []
        for task in scenario['tasks']:
            # Assign to first node with capacity
            for node in scenario['nodes']:
                if node.has_capacity(task):
                    allocations.append({'task': task['id'], 'node': node['id']})
                    break
        return allocations

class RoundRobin:
    """Baseline 3: Round-robin load balancing"""
    def __init__(self):
        self.next_node = 0
        
    def allocate(self, scenario):
        allocations = []
        nodes = scenario['nodes']
        for task in scenario['tasks']:
            node = nodes[self.next_node % len(nodes)]
            allocations.append({'task': task['id'], 'node': node['id']})
            self.next_node += 1
        return allocations

class BestFit:
    """Baseline 4: Best-fit bin packing"""
    def allocate(self, scenario):
        allocations = []
        for task in scenario['tasks']:
            # Find node with minimum remaining capacity that fits
            best_node = min(
                [n for n in scenario['nodes'] if n.has_capacity(task)],
                key=lambda n: n.remaining_capacity()
            )
            allocations.append({'task': task['id'], 'node': best_node['id']})
        return allocations
```

### B. Classical Optimization:

```python
class GeneticAlgorithm:
    """Baseline 5: GA optimizer"""
    def __init__(self, pop_size=100, generations=50):
        self.pop_size = pop_size
        self.generations = generations
    
    def allocate(self, scenario):
        # Initialize population
        population = self._initialize_population(scenario)
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = [self._evaluate(ind, scenario) for ind in population]
            
            # Selection
            parents = self._tournament_selection(population, fitness)
            
            # Crossover & Mutation
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            
            population = offspring
        
        # Return best individual
        best = max(population, key=lambda ind: self._evaluate(ind, scenario))
        return self._decode(best)
```

### C. Reinforcement Learning:

```python
class DQNBaseline:
    """Baseline 6: Deep Q-Network"""
    def __init__(self):
        self.model = self._build_model()
        self.replay_buffer = []
        
    def train(self, scenarios):
        """Train on same scenarios AURORA sees"""
        for episode in range(1000):
            scenario = random.choice(scenarios)
            state = self._encode_state(scenario)
            
            for task in scenario['tasks']:
                # Choose action (node assignment)
                action = self._epsilon_greedy(state)
                
                # Execute
                next_state, reward = self._step(state, action)
                
                # Store transition
                self.replay_buffer.append((state, action, reward, next_state))
                
                # Train
                if len(self.replay_buffer) > 32:
                    self._train_batch()
                
                state = next_state
    
    def allocate(self, scenario):
        """Generate allocation using trained Q-network"""
        state = self._encode_state(scenario)
        allocations = []
        
        for task in scenario['tasks']:
            action = self.model.predict(state).argmax()
            node_id = self._decode_action(action)
            allocations.append({'task': task['id'], 'node': node_id})
            state = self._update_state(state, task, node_id)
        
        return allocations
```

### D. Implement These For Fair Comparison:

```python
baselines = {
    'Random': RandomAllocation(),
    'Greedy': GreedyFirstFit(),
    'RoundRobin': RoundRobin(),
    'BestFit': BestFit(),
    'GA': GeneticAlgorithm(),
    'DQN': DQNBaseline()  # Train on same data
}
```

---

## 8. VALIDATION PROTOCOL

### Complete Evaluation Pipeline:

```python
class AURORAValidator:
    def __init__(self, aurora_system, baselines):
        self.aurora = aurora_system
        self.baselines = baselines
        
    def run_full_evaluation(self):
        """Complete validation protocol"""
        
        results = {
            'benchmark_performance': self.evaluate_benchmarks(),
            'generalization': self.evaluate_generalization(),
            'ablation': self.run_ablation_studies(),
            'scalability': self.evaluate_scalability(),
            'explainability': self.evaluate_explainability()
        }
        
        return results
    
    def evaluate_benchmarks(self):
        """Test on standard datasets"""
        print("\n" + "="*60)
        print("BENCHMARK EVALUATION")
        print("="*60)
        
        benchmarks = {
            'Google Cluster Trace': load_google_trace(100),
            'Azure Public Dataset': load_azure_trace(100),
            'Synthetic Workload': generate_synthetic_scenarios(100)
        }
        
        all_results = {}
        
        for benchmark_name, scenarios in benchmarks.items():
            print(f"\n📊 Testing on {benchmark_name}...")
            
            # Test AURORA
            aurora_results = []
            for scenario in tqdm(scenarios):
                policy = self.aurora.policy_db.get_best_policy(scenario)
                results = simulator.execute_policy(scenario, policy)
                aurora_results.append(results)
            
            # Test each baseline
            baseline_results = {}
            for baseline_name, baseline in self.baselines.items():
                baseline_results[baseline_name] = []
                for scenario in scenarios:
                    policy = baseline.allocate(scenario)
                    results = simulator.execute_policy(scenario, policy)
                    baseline_results[baseline_name].append(results)
            
            # Compute statistics
            all_results[benchmark_name] = {
                'aurora': compute_statistics(aurora_results),
                'baselines': {
                    name: compute_statistics(results)
                    for name, results in baseline_results.items()
                }
            }
        
        # Print comparison table
        self._print_comparison_table(all_results)
        
        return all_results
    
    def evaluate_generalization(self):
        """Zero-shot transfer across domains"""
        print("\n" + "="*60)
        print("GENERALIZATION EVALUATION")
        print("="*60)
        
        transfers = [
            ('cloud', 'edge'),
            ('edge', 'network'),
            ('network', 'hybrid')
        ]
        
        results = {}
        for source, target in transfers:
            print(f"\n🔄 Testing {source} → {target} transfer...")
            
            # Get policies from source domain
            source_policies = self.aurora.policy_db.get_domain_policies(source)
            
            # Test on target domain scenarios
            target_scenarios = self._get_domain_scenarios(target, count=50)
            
            transfer_results = []
            for scenario in target_scenarios:
                # Find most similar source policy
                policy = self._find_similar_policy(source_policies, scenario)
                
                # Execute in target domain
                result = simulator.execute_policy(scenario, policy)
                transfer_results.append(result)
            
            # Compare to baseline trained on target
            baseline_trained = self.baselines['DQN']
            baseline_trained.train(target_scenarios)  # Train on target
            
            baseline_results = []
            for scenario in target_scenarios:
                policy = baseline_trained.allocate(scenario)
                result = simulator.execute_policy(scenario, policy)
                baseline_results.append(result)
            
            results[f"{source}→{target}"] = {
                'aurora_transfer': compute_statistics(transfer_results),
                'baseline_retrained': compute_statistics(baseline_results),
                'transfer_efficiency': self._compute_transfer_efficiency(
                    transfer_results, baseline_results
                )
            }
        
        return results
    
    def run_ablation_studies(self):
        """Test which components matter"""
        print("\n" + "="*60)
        print("ABLATION STUDIES")
        print("="*60)
        
        test_scenarios = generate_synthetic_scenarios(100)
        
        ablations = {
            'Full AURORA': self.aurora,
            'No Attacker (random scenarios)': self._create_no_attacker_variant(),
            'No Policy DB (no memory)': self._create_no_memory_variant(),
            'No Curriculum (fixed difficulty)': self._create_no_curriculum_variant(),
            'Single Agent (no adversarial)': self._create_single_agent_variant()
        }
        
        results = {}
        for name, variant in ablations.items():
            print(f"\n🧪 Testing {name}...")
            variant_results = self._evaluate_variant(variant, test_scenarios)
            results[name] = compute_statistics(variant_results)
        
        return results
    
    def evaluate_scalability(self):
        """Test performance as problem size grows"""
        print("\n" + "="*60)
        print("SCALABILITY EVALUATION")
        print("="*60)
        
        scales = [
            {'tasks': 10, 'nodes': 3},
            {'tasks': 50, 'nodes': 10},
            {'tasks': 100, 'nodes': 20},
            {'tasks': 500, 'nodes': 50},
            {'tasks': 1000, 'nodes': 100}
        ]
        
        results = {}
        for scale in scales:
            print(f"\n📈 Testing scale: {scale['tasks']} tasks, {scale['nodes']} nodes")
            
            scenarios = generate_synthetic_scenarios(
                num_scenarios=20,
                num_tasks=scale['tasks'],
                num_nodes=scale['nodes']
            )
            
            # Measure inference time
            start = time.time()
            aurora_results = []
            for scenario in scenarios:
                policy = self.aurora.policy_db.get_best_policy(scenario)
                result = simulator.execute_policy(scenario, policy)
                aurora_results.append(result)
            aurora_time = time.time() - start
            
            # Baseline
            start = time.time()
            baseline_results = []
            for scenario in scenarios:
                policy = self.baselines['DQN'].allocate(scenario)
                result = simulator.execute_policy(scenario, policy)
                baseline_results.append(result)
            baseline_time = time.time() - start
            
            results[f"{scale['tasks']}_tasks"] = {
                'aurora_performance': compute_statistics(aurora_results),
                'aurora_time_sec': aurora_time,
                'baseline_performance': compute_statistics(baseline_results),
                'baseline_time_sec': baseline_time,
                'speedup': baseline_time / aurora_time
            }
        
        return results
    
    def evaluate_explainability(self):
        """Qualitative evaluation of explanations"""
        print("\n" + "="*60)
        print("EXPLAINABILITY EVALUATION")
        print("="*60)
        
        # Sample 20 policies with explanations
        sample_policies = random.sample(self.aurora.policy_db.policies, 20)
        
        # Extract explanations
        explanations = [
            {
                'scenario': p['scenario'],
                'policy': p['policy'],
                'reasoning': p['policy']['allocations'][0]['reasoning']
            }
            for p in sample_policies
        ]
        
        # Automated metrics
        automated_scores = {
            'completeness': self._check_completeness(explanations),
            'consistency': self._check_consistency(explanations),
            'technical_accuracy': self._check_accuracy(explanations)
        }
        
        # Human evaluation (you manually score 20 explanations)
        print("\n📝 Manual scoring required:")
        print("For each explanation, rate 1-5 on:")
        print("  - Clarity: Is it understandable?")
        print("  - Correctness: Is the reasoning sound?")
        print("  - Completeness: Does it explain all decisions?")
        
        # Save to file for manual annotation
        with open('explanations_to_score.json', 'w') as f:
            json.dump(explanations, f, indent=2)
        
        return {
            'automated': automated_scores,
            'manual': 'See explanations_to_score.json'
        }
    
    def _print_comparison_table(self, results):
        """Pretty print results"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        for benchmark, data in results.items():
            print(f"\n{benchmark}:")
            print("-" * 80)
            print(f"{'Method':<20} {'Latency↓':<15} {'Utilization↑':<15} {'Cost↓':<15} {'SLA%↑':<10}")
            print("-" * 80)
            
            # AURORA
            aurora = data['aurora']
            print(f"{'AURORA (ours)':<20} {aurora['mean_latency']:<15.2f} "
                  f"{aurora['mean_utilization']:<15.2%} "
                  f"{aurora['mean_cost']:<15.2f} "
                  f"{aurora['sla_success_rate']:<10.1%}")
            
            # Baselines
            for name, stats in data['baselines'].items():
                print(f"{name:<20} {stats['mean_latency']:<15.2f} "
                      f"{stats['mean_utilization']:<15.2%} "
                      f"{stats['mean_cost']:<15.2f} "
                      f"{stats['sla_success_rate']:<10.1%}")
```

### Statistical Significance Testing:

```python
def compute_statistical_significance(aurora_results, baseline_results):
    """T-test for significance"""
    from scipy import stats
    
    aurora_latencies = [r['avg_latency_ms'] for r in aurora_results]
    baseline_latencies = [r['avg_latency_ms'] for r in baseline_results]
    
    t_stat, p_value = stats.ttest_ind(aurora_latencies, baseline_latencies)
    
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant improvement (p < 0.05)")
    else:
        print("❌ Not statistically significant")
    
    return p_value
```

---

## FINAL IMPLEMENTATION CHECKLIST

```python
# Your complete workflow:

# 1. Setup (Week 1)
✅ Install dependencies
✅ Setup Gemini API
✅ Create 50 seed scenarios
✅ Implement dummy simulator for testing

# 2. Core Implementation (Weeks 2-4)
✅ Implement Attacker agent (Gemini)
✅ Implement Defender agent (Gemini)
✅ Implement realistic simulator
✅ Implement reward engine
✅ Implement policy database

# 3. Training (Weeks 5-6)
✅ Run 1000 self-play iterations
✅ Monitor convergence
✅ Save checkpoints

# 4. Evaluation (Weeks 7-8)
✅ Implement all baselines
✅ Run benchmark tests
✅ Compute statistical significance
✅ Ablation studies
✅ Generalization tests

# 5. Paper Writing (Weeks 9-10)
✅ Write methods section
✅ Create result visualizations
✅ Write discussion
✅ Polish and submit
```

