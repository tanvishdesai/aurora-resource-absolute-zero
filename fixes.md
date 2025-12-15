# AURORA Complete Fix Guide

**Document Purpose**: Detailed instructions to transform your current 4.5/10 implementation into publication-ready 8/10 system.

**Time Estimate**: 6-10 weeks full-time
**Priority**: Fix CRITICAL issues first (weeks 1-4), then HIGH (weeks 5-7), then MEDIUM (weeks 8-10)

---

## Table of Contents

1. [Critical Fixes (Must Do)](#critical-fixes)
   - [Fix 1: Realistic Simulator](#fix-1-realistic-simulator)
   - [Fix 2: Comprehensive Evaluation](#fix-2-comprehensive-evaluation)
   - [Fix 3: Strong Baselines](#fix-3-strong-baselines)
   - [Fix 4: Real Data Integration](#fix-4-real-data-integration)
2. [High Priority Fixes](#high-priority-fixes)
   - [Fix 5: Ablation Studies](#fix-5-ablation-studies)
   - [Fix 6: Zero-Shot Transfer](#fix-6-zero-shot-transfer)
   - [Fix 7: Quality Control Integration](#fix-7-quality-control-integration)
3. [Medium Priority Fixes](#medium-priority-fixes)
   - [Fix 8: Explainability Evaluation](#fix-8-explainability-evaluation)
   - [Fix 9: Meta-Prompt Validation](#fix-9-meta-prompt-validation)
   - [Fix 10: Scalability Testing](#fix-10-scalability-testing)

---

# CRITICAL FIXES

## Fix 1: Realistic Simulator

### **Current Problem**

**File**: `aurora/simulators/simulator.py`

**Current Code**:
```python
execution_time = (task['duration'] / speed_factor) * (1 + congestion)
# where congestion = (node_usage['cpu'] / node['cpu_capacity']) * 0.5
```

**Issues**:
- ❌ Static resource model (no queuing)
- ❌ Linear congestion model (unrealistic)
- ❌ No network latency
- ❌ No energy modeling
- ❌ No failure scenarios
- ❌ Instant allocation (no setup time)

**Impact**: All experimental results are meaningless because simulator doesn't reflect reality.

---

### **Solution**

Replace with discrete event simulation that models:
1. Task queuing when nodes are busy
2. Non-linear performance degradation under load
3. Network communication delays
4. Energy consumption based on utilization
5. Node failures and recovery

---

### **Implementation Steps**

#### Step 1.1: Create Discrete Event Simulator Base

Create new file: `aurora/simulators/event_simulator.py`

```python
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class EventType(Enum):
    TASK_ARRIVAL = "task_arrival"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    NODE_FAILURE = "node_failure"
    NODE_RECOVERY = "node_recovery"

@dataclass
class Event:
    time: float
    event_type: EventType
    task_id: Optional[str] = None
    node_id: Optional[str] = None
    
    def __lt__(self, other):
        return self.time < other.time

class DiscreteEventSimulator:
    def __init__(self, config: dict):
        self.event_queue = []
        self.current_time = 0.0
        
        # Configuration
        self.network_latency_base = config.get('network_latency_base', 1.0)  # ms
        self.network_latency_variance = config.get('network_latency_variance', 0.5)
        self.node_failure_rate = config.get('node_failure_rate', 0.001)  # per hour
        self.node_recovery_time = config.get('node_recovery_time', 300)  # seconds
        
        # State
        self.node_states = {}
        self.task_states = {}
        self.statistics = {
            'task_completions': [],
            'node_utilization_history': [],
            'energy_consumption': 0.0,
            'network_bytes_transferred': 0
        }
        
    def schedule_event(self, event: Event):
        """Add event to priority queue"""
        heapq.heappush(self.event_queue, event)
    
    def get_next_event(self) -> Optional[Event]:
        """Get next event from queue"""
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None
    
    def advance_time(self, new_time: float):
        """Advance simulation clock"""
        time_delta = new_time - self.current_time
        
        # Update energy consumption for all active nodes
        for node_id, node_state in self.node_states.items():
            if node_state['status'] == 'active':
                power = self._compute_power_consumption(node_state)
                energy = power * (time_delta / 3600)  # kWh
                self.statistics['energy_consumption'] += energy
        
        self.current_time = new_time
    
    def _compute_power_consumption(self, node_state: dict) -> float:
        """
        Compute power consumption based on utilization
        Uses realistic power model: P = P_idle + (P_max - P_idle) * utilization^2
        """
        P_idle = 100  # Watts (idle power)
        P_max = 300   # Watts (max power)
        
        cpu_util = node_state['cpu_used'] / node_state['cpu_capacity']
        
        # Quadratic relationship (based on research)
        power = P_idle + (P_max - P_idle) * (cpu_util ** 2)
        
        return power
```

---

#### Step 1.2: Implement Queuing Model

Add to `event_simulator.py`:

```python
class TaskQueue:
    """
    Priority queue for tasks waiting on a node
    Uses M/G/1 queuing model for service time estimation
    """
    def __init__(self, node_capacity: dict):
        self.queue = []
        self.node_capacity = node_capacity
        self.active_tasks = []
        
    def add_task(self, task: dict, arrival_time: float):
        """Add task to queue with priority"""
        heapq.heappush(self.queue, (
            -task['priority'],  # Negative for max-heap
            arrival_time,
            task
        ))
    
    def can_schedule(self, task: dict) -> bool:
        """Check if task can be scheduled now"""
        cpu_available = self.node_capacity['cpu'] - sum(
            t['cpu'] for t in self.active_tasks
        )
        mem_available = self.node_capacity['memory'] - sum(
            t['memory'] for t in self.active_tasks
        )
        
        return (cpu_available >= task['cpu'] and 
                mem_available >= task['memory'])
    
    def get_next_schedulable(self) -> Optional[dict]:
        """Get next task that can be scheduled"""
        for i, (priority, arrival, task) in enumerate(self.queue):
            if self.can_schedule(task):
                self.queue.pop(i)
                self.active_tasks.append(task)
                return task
        return None
    
    def compute_service_time(self, task: dict) -> float:
        """
        Compute service time using M/G/1 queuing model
        Service time increases with load
        """
        base_time = task['duration']
        
        # Current load factor
        cpu_load = sum(t['cpu'] for t in self.active_tasks) / self.node_capacity['cpu']
        
        # M/G/1 mean waiting time
        # W = (λ * S^2) / (2 * (1 - ρ))
        # Simplified: service time increases as 1/(1-ρ)
        if cpu_load < 0.95:  # Prevent division by zero
            service_time = base_time / (1 - cpu_load)
        else:
            service_time = base_time * 20  # Heavy penalty for overload
        
        # Add variance (coefficient of variation = 1 for exponential)
        variance = base_time * np.random.exponential(1.0)
        
        return service_time + variance * 0.1  # Small variance component
```

---

#### Step 1.3: Complete Realistic Simulator

Add to `event_simulator.py`:

```python
class RealisticResourceSimulator(DiscreteEventSimulator):
    def execute_policy(self, scenario: dict, policy: dict) -> dict:
        """
        Execute allocation policy using discrete event simulation
        """
        # Initialize
        self._initialize_simulation(scenario)
        
        # Schedule all task arrivals
        for task in scenario['tasks']:
            self.schedule_event(Event(
                time=task.get('arrival_time', 0),
                event_type=EventType.TASK_ARRIVAL,
                task_id=task['id']
            ))
        
        # Process events
        while True:
            event = self.get_next_event()
            if event is None:
                break
            
            self.advance_time(event.time)
            self._handle_event(event, scenario, policy)
        
        # Compute final statistics
        return self._compute_results()
    
    def _initialize_simulation(self, scenario: dict):
        """Initialize simulation state"""
        self.event_queue = []
        self.current_time = 0.0
        
        # Initialize node states
        self.node_states = {}
        for node in scenario['nodes']:
            self.node_states[node['id']] = {
                'cpu_capacity': node['cpu_capacity'],
                'memory_capacity': node['memory_capacity'],
                'cpu_used': 0.0,
                'memory_used': 0.0,
                'queue': TaskQueue({
                    'cpu': node['cpu_capacity'],
                    'memory': node['memory_capacity']
                }),
                'status': 'active',
                'energy_efficiency': node.get('energy_efficiency', 1.0),
                'cost_per_hour': node.get('cost_per_hour', 0.1)
            }
        
        # Initialize task states
        self.task_states = {}
        for task in scenario['tasks']:
            self.task_states[task['id']] = {
                'status': 'pending',
                'arrival_time': task.get('arrival_time', 0),
                'start_time': None,
                'completion_time': None,
                'assigned_node': None
            }
    
    def _handle_event(self, event: Event, scenario: dict, policy: dict):
        """Handle different event types"""
        
        if event.event_type == EventType.TASK_ARRIVAL:
            self._handle_task_arrival(event, scenario, policy)
            
        elif event.event_type == EventType.TASK_START:
            self._handle_task_start(event, scenario)
            
        elif event.event_type == EventType.TASK_COMPLETE:
            self._handle_task_completion(event, scenario)
            
        elif event.event_type == EventType.NODE_FAILURE:
            self._handle_node_failure(event)
            
        elif event.event_type == EventType.NODE_RECOVERY:
            self._handle_node_recovery(event)
    
    def _handle_task_arrival(self, event: Event, scenario: dict, policy: dict):
        """Handle task arrival"""
        task_id = event.task_id
        task = next(t for t in scenario['tasks'] if t['id'] == task_id)
        
        # Find assigned node from policy
        allocation = next(
            (a for a in policy['allocations'] if a['task_id'] == task_id),
            None
        )
        
        if allocation is None:
            # Task not allocated - mark as failed
            self.task_states[task_id]['status'] = 'failed'
            self.task_states[task_id]['completion_time'] = event.time
            return
        
        node_id = allocation['node_id']
        node_state = self.node_states[node_id]
        
        # Check if node is active
        if node_state['status'] != 'active':
            # Node failed - mark task as failed
            self.task_states[task_id]['status'] = 'failed'
            self.task_states[task_id]['completion_time'] = event.time
            return
        
        # Add task to node's queue
        node_state['queue'].add_task(task, event.time)
        self.task_states[task_id]['assigned_node'] = node_id
        
        # Try to schedule immediately
        self._try_schedule_tasks(node_id, scenario)
    
    def _try_schedule_tasks(self, node_id: str, scenario: dict):
        """Try to schedule waiting tasks on a node"""
        node_state = self.node_states[node_id]
        queue = node_state['queue']
        
        # Try to schedule next task
        task_dict = queue.get_next_schedulable()
        
        if task_dict:
            # Schedule task start
            # Add network transfer delay
            network_delay = self._compute_network_delay(task_dict, node_id)
            start_time = self.current_time + network_delay
            
            self.schedule_event(Event(
                time=start_time,
                event_type=EventType.TASK_START,
                task_id=task_dict['id'],
                node_id=node_id
            ))
    
    def _handle_task_start(self, event: Event, scenario: dict):
        """Handle task starting execution"""
        task_id = event.task_id
        node_id = event.node_id
        
        task = next(t for t in scenario['tasks'] if t['id'] == task_id)
        node_state = self.node_states[node_id]
        
        # Update node resources
        node_state['cpu_used'] += task['cpu']
        node_state['memory_used'] += task['memory']
        
        # Update task state
        self.task_states[task_id]['status'] = 'running'
        self.task_states[task_id]['start_time'] = event.time
        
        # Compute service time (with queuing effects)
        service_time = node_state['queue'].compute_service_time(task)
        
        # Schedule completion
        completion_time = event.time + service_time
        
        self.schedule_event(Event(
            time=completion_time,
            event_type=EventType.TASK_COMPLETE,
            task_id=task_id,
            node_id=node_id
        ))
    
    def _handle_task_completion(self, event: Event, scenario: dict):
        """Handle task completion"""
        task_id = event.task_id
        node_id = event.node_id
        
        task = next(t for t in scenario['tasks'] if t['id'] == task_id)
        node_state = self.node_states[node_id]
        
        # Release resources
        node_state['cpu_used'] -= task['cpu']
        node_state['memory_used'] -= task['memory']
        
        # Remove from active tasks
        node_state['queue'].active_tasks = [
            t for t in node_state['queue'].active_tasks 
            if t['id'] != task_id
        ]
        
        # Update task state
        self.task_states[task_id]['status'] = 'completed'
        self.task_states[task_id]['completion_time'] = event.time
        
        # Record completion
        latency = event.time - self.task_states[task_id]['arrival_time']
        sla_violated = latency > task.get('sla_latency_ms', float('inf'))
        
        self.statistics['task_completions'].append({
            'task_id': task_id,
            'arrival_time': self.task_states[task_id]['arrival_time'],
            'start_time': self.task_states[task_id]['start_time'],
            'completion_time': event.time,
            'latency': latency,
            'sla_violated': sla_violated,
            'node_id': node_id
        })
        
        # Try to schedule next waiting task
        self._try_schedule_tasks(node_id, scenario)
    
    def _compute_network_delay(self, task: dict, node_id: str) -> float:
        """
        Compute network transfer delay
        Delay = data_size / bandwidth + base_latency
        """
        data_size_mb = task.get('input_data_mb', 10)  # Default 10MB
        bandwidth_mbps = 1000  # 1 Gbps
        
        transfer_time = data_size_mb / bandwidth_mbps
        base_latency = np.random.normal(
            self.network_latency_base, 
            self.network_latency_variance
        )
        
        return max(0, transfer_time + base_latency / 1000)  # Convert ms to s
    
    def _compute_results(self) -> dict:
        """Compute final simulation results"""
        completions = self.statistics['task_completions']
        
        if not completions:
            return {
                'avg_latency': float('inf'),
                'sla_violations': len(self.task_states),
                'sla_violation_rate': 1.0,
                'successful_tasks': 0,
                'total_energy': self.statistics['energy_consumption'],
                'total_cost': 0,
                'avg_cpu_utilization': 0
            }
        
        # Compute metrics
        latencies = [c['latency'] for c in completions]
        sla_violations = sum(1 for c in completions if c['sla_violated'])
        
        # Cost computation
        total_cost = 0
        for node_id, node_state in self.node_states.items():
            runtime_hours = self.current_time / 3600
            total_cost += runtime_hours * node_state['cost_per_hour']
        
        # Utilization (time-averaged)
        total_capacity = sum(n['cpu_capacity'] for n in self.node_states.values())
        
        # Approximate average utilization
        total_cpu_seconds = sum(
            (c['completion_time'] - c['start_time']) * 
            next(t['cpu'] for t in self.task_states.values() if t)
            for c in completions
        )
        avg_cpu_utilization = total_cpu_seconds / (self.current_time * total_capacity)
        
        return {
            'avg_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'max_latency': np.max(latencies),
            'sla_violations': sla_violations,
            'sla_violation_rate': sla_violations / len(self.task_states),
            'successful_tasks': len(completions),
            'total_energy': self.statistics['energy_consumption'],
            'total_cost': total_cost,
            'avg_cpu_utilization': avg_cpu_utilization,
            'task_completions': completions
        }
```

---

#### Step 1.4: Update Main Simulator Interface

Modify `aurora/simulators/simulator.py`:

```python
from aurora.simulators.event_simulator import RealisticResourceSimulator

class UnifiedResourceSimulator:
    """
    Unified interface for resource simulation
    Now uses realistic discrete event simulation
    """
    def __init__(self, config: dict = None):
        self.config = config or {
            'network_latency_base': 1.0,
            'network_latency_variance': 0.5,
            'node_failure_rate': 0.0,  # Disable failures for initial testing
            'node_recovery_time': 300
        }
        self.simulator = RealisticResourceSimulator(self.config)
    
    def execute_policy(self, scenario: dict, policy: dict) -> dict:
        """Execute policy using realistic simulator"""
        return self.simulator.execute_policy(scenario, policy)
```

---

#### Step 1.5: Test New Simulator

Create `tests/test_realistic_simulator.py`:

```python
import pytest
import numpy as np
from aurora.simulators.simulator import UnifiedResourceSimulator

def test_queuing_delays():
    """Test that tasks experience queuing delays under load"""
    
    # Scenario: 10 tasks, 1 node with capacity for 2 concurrent tasks
    scenario = {
        'tasks': [
            {
                'id': f't{i}',
                'cpu': 1,
                'memory': 1,
                'duration': 10,
                'arrival_time': i * 2,  # Arrive every 2 seconds
                'sla_latency_ms': 100
            }
            for i in range(10)
        ],
        'nodes': [
            {
                'id': 'n1',
                'cpu_capacity': 2,
                'memory_capacity': 10
            }
        ]
    }
    
    policy = {
        'allocations': [
            {'task_id': f't{i}', 'node_id': 'n1'}
            for i in range(10)
        ]
    }
    
    simulator = UnifiedResourceSimulator()
    results = simulator.execute_policy(scenario, policy)
    
    # First few tasks should complete quickly
    # Later tasks should experience queuing delays
    
    completions = results['task_completions']
    
    # Sort by arrival time
    completions.sort(key=lambda x: x['arrival_time'])
    
    # Task 0: should complete near 10s (no queue)
    assert completions[0]['latency'] < 15
    
    # Task 9: should experience significant queuing
    assert completions[9]['latency'] > 30  # Waited in queue

def test_network_delays():
    """Test that network delays are added"""
    
    scenario = {
        'tasks': [{
            'id': 't1',
            'cpu': 1,
            'memory': 1,
            'duration': 5,
            'arrival_time': 0,
            'input_data_mb': 100  # 100MB transfer
        }],
        'nodes': [{
            'id': 'n1',
            'cpu_capacity': 10,
            'memory_capacity': 10
        }]
    }
    
    policy = {
        'allocations': [{'task_id': 't1', 'node_id': 'n1'}]
    }
    
    simulator = UnifiedResourceSimulator()
    results = simulator.execute_policy(scenario, policy)
    
    completion = results['task_completions'][0]
    
    # Latency should include network delay
    # 100MB / 1000 Mbps = 0.1s transfer time + base latency
    assert completion['latency'] > 5.0  # Duration + network delay

def test_energy_computation():
    """Test that energy consumption is computed"""
    
    scenario = {
        'tasks': [{
            'id': 't1',
            'cpu': 8,  # High CPU usage
            'memory': 16,
            'duration': 3600,  # 1 hour
            'arrival_time': 0
        }],
        'nodes': [{
            'id': 'n1',
            'cpu_capacity': 16,
            'memory_capacity': 32
        }]
    }
    
    policy = {
        'allocations': [{'task_id': 't1', 'node_id': 'n1'}]
    }
    
    simulator = UnifiedResourceSimulator()
    results = simulator.execute_policy(scenario, policy)
    
    # Energy should be > 0
    assert results['total_energy'] > 0
    
    # At 50% utilization, power ≈ 100 + 200 * 0.5^2 = 150W
    # 1 hour = 0.15 kWh
    assert 0.1 < results['total_energy'] < 0.25

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

### **Testing Plan**

```bash
# Step 1: Run unit tests
python -m pytest tests/test_realistic_simulator.py -v

# Step 2: Compare old vs new simulator
python scripts/compare_simulators.py

# Step 3: Validate on simple scenario
python scripts/validate_simulator.py
```

Create `scripts/compare_simulators.py`:

```python
"""Compare old simple simulator vs new realistic simulator"""

from aurora.simulators.simulator import UnifiedResourceSimulator
import matplotlib.pyplot as plt

# Simple scenario
scenario = {
    'tasks': [
        {
            'id': f't{i}',
            'cpu': 2,
            'memory': 4,
            'duration': 100,
            'arrival_time': i * 10
        }
        for i in range(20)
    ],
    'nodes': [
        {'id': f'n{i}', 'cpu_capacity': 8, 'memory_capacity': 16}
        for i in range(3)
    ]
}

policy = {
    'allocations': [
        {'task_id': f't{i}', 'node_id': f'n{i % 3}'}
        for i in range(20)
    ]
}

# New simulator
new_sim = UnifiedResourceSimulator()
new_results = new_sim.execute_policy(scenario, policy)

print("New Realistic Simulator:")
print(f"  Avg Latency: {new_results['avg_latency']:.2f}s")
print(f"  P95 Latency: {new_results['p95_latency']:.2f}s")
print(f"  Energy: {new_results['total_energy']:.3f} kWh")
print(f"  Utilization: {new_results['avg_cpu_utilization']:.2%}")

# Plot latency distribution
latencies = [c['latency'] for c in new_results['task_completions']]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(latencies, bins=20)
plt.xlabel('Latency (s)')
plt.ylabel('Count')
plt.title('Task Latency Distribution')

plt.subplot(1, 2, 2)
plt.plot(sorted(latencies))
plt.xlabel('Task (sorted)')
plt.ylabel('Latency (s)')
plt.title('Latency CDF')

plt.tight_layout()
plt.savefig('simulator_comparison.png')
print("\nPlot saved to simulator_comparison.png")
```

---

### **Acceptance Criteria**

✅ Tasks experience queuing delays when nodes are busy  
✅ Network transfer time is added to latency  
✅ Energy consumption computed based on utilization  
✅ Cost computed based on node runtime  
✅ Service time increases non-linearly with load  
✅ All tests pass

---

### **Time Estimate**: 1-2 weeks

---

## Fix 2: Comprehensive Evaluation

### **Current Problem**

**File**: `aurora/evaluation.py`

**Current Code**:
```python
scenarios = data_manager.get_training_scenarios(source=args.data_source, num_scenarios=5)
```

**Issues**:
- ❌ Only 5 test scenarios (need 100+)
- ❌ Baselines not properly tuned
- ❌ Results not reproducible (no seed setting)

**Impact**: Cannot make any scientific claims about performance.

---

### **Solution**

Implement comprehensive evaluation protocol with:
1. 100+ diverse test scenarios
2. Proper baseline tuning
3. Reproducible results

---

### **Implementation Steps**

#### Step 2.1: Create Comprehensive Test Set

Create `aurora/data/test_set_generator.py`:

```python
"""Generate diverse, stratified test set"""

import numpy as np
import json
from typing import List, Dict

class TestSetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_comprehensive_test_set(self) -> Dict[str, List[dict]]:
        """
        Generate 100+ test scenarios across difficulty levels
        Stratified by:
        - Difficulty (easy, medium, hard)
        - Workload type (web, batch, ml, mixed)
        - Problem size (small, medium, large)
        """
        
        test_set = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        # Easy scenarios (30): Small, well-balanced
        for i in range(30):
            scenario = self._generate_easy_scenario(i)
            test_set['easy'].append(scenario)
        
        # Medium scenarios (40): Moderate size, some constraints
        for i in range(40):
            scenario = self._generate_medium_scenario(i)
            test_set['medium'].append(scenario)
        
        # Hard scenarios (30): Large, tight constraints
        for i in range(30):
            scenario = self._generate_hard_scenario(i)
            test_set['hard'].append(scenario)
        
        return test_set
    
    def _generate_easy_scenario(self, idx: int) -> dict:
        """Easy: 5-15 tasks, 2-5 nodes, plenty of capacity"""
        
        num_tasks = self.rng.randint(5, 16)
        num_nodes = self.rng.randint(2, 6)
        
        # Tasks: small, uniform
        tasks = []
        for t in range(num_tasks):
            tasks.append({
                'id': f'easy_{idx}_t{t}',
                'cpu': self.rng.uniform(0.5, 2.0),
                'memory': self.rng.uniform(1, 4),
                'duration': self.rng.uniform(10, 100),
                'arrival_time': t * 5,  # Spread out
                'sla_latency_ms': 200,
                'priority': self.rng.choice([1, 2, 3])
            })
        
        # Nodes: ample capacity (2x demand)
        total_cpu_demand = sum(t['cpu'] for t in tasks)
        total_mem_demand = sum(t['memory'] for t in tasks)
        
        nodes = []
        for n in range(num_nodes):
            nodes.append({
                'id': f'easy_{idx}_n{n}',
                'cpu_capacity': (total_cpu_demand * 2) / num_nodes,
                'memory_capacity': (total_mem_demand * 2) / num_nodes,
                'cost_per_hour': 0.1,
                'energy_efficiency': 1.0
            })
        
        return {
            'id': f'easy_scenario_{idx}',
            'difficulty': 1,
            'category': 'easy',
            'tasks': tasks,
            'nodes': nodes
        }
    
    def _generate_medium_scenario(self, idx: int) -> dict:
        """Medium: 20-50 tasks, 5-15 nodes, moderate constraints"""
        
        workload_type = self.rng.choice(['web', 'batch', 'mixed'])
        
        if workload_type == 'web':
            return self._generate_web_workload(idx, difficulty='medium')
        elif workload_type == 'batch':
            return self._generate_batch_workload(idx, difficulty='medium')
        else:
            return self._generate_mixed_workload(idx, difficulty='medium')
    
    def _generate_hard_scenario(self, idx: int) -> dict:
        """Hard: 50-100 tasks, 15-30 nodes, tight constraints"""
        
        num_tasks = self.rng.randint(50, 101)
        num_nodes = self.rng.randint(15, 31)
        
        # Tasks: heterogeneous, log-normal distribution
        tasks = []
        for t in range(num_tasks):
            size_factor = self.rng.lognormal(0, 1.0)
            
            tasks.append({
                'id': f'hard_{idx}_t{t}',
                'cpu': np.clip(0.5 * size_factor, 0.1, 32),
                'memory': np.clip(1 * size_factor * 2, 0.5, 128),
                'duration': self.rng.lognormal(3, 1),  # Mean ~20s, variance high
                'arrival_time': self._generate_bursty_arrival(t, num_tasks),
                'sla_latency_ms': self.rng.uniform(50, 150),
                'priority': self.rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            })
        
        # Nodes: tight capacity (1.2x demand)
        total_cpu_demand = sum(t['cpu'] for t in tasks)
        total_mem_demand = sum(t['memory'] for t in tasks)
        
        nodes = []
        # Heterogeneous node types
        node_types = [
            {'cpu': 4, 'mem': 8, 'prob': 0.4},
            {'cpu': 8, 'mem': 16, 'prob': 0.3},
            {'cpu': 16, 'mem': 32, 'prob': 0.2},
            {'cpu': 32, 'mem': 64, 'prob': 0.1}
        ]
        
        for n in range(num_nodes):
            node_type = self.rng.choice(
                node_types,
                p=[nt['prob'] for nt in node_types]
            )
            
            nodes.append({
                'id': f'hard_{idx}_n{n}',
                'cpu_capacity': node_type['cpu'],
                'memory_capacity': node_type['mem'],
                'cost_per_hour': 0.01 * node_type['cpu'],
                'energy_efficiency': self.rng.uniform(0.8, 1.2)
            })
        
        return {
            'id': f'hard_scenario_{idx}',
            'difficulty': 5,
            'category': 'hard',
            'tasks': tasks,
            'nodes': nodes
        }
    
    def _generate_bursty_arrival(self, task_idx: int, total_tasks: int) -> float:
        """Generate bursty arrival pattern"""
        # Create bursts every 50 tasks
        burst_id = task_idx // 50
        within_burst = task_idx % 50
        
        burst_start = burst_id * 100  # Bursts separated by 100s
        arrival = burst_start + within_burst * self.rng.exponential(0.5)
        
        return arrival
    
    def _generate_web_workload(self, idx: int, difficulty: str) -> dict:
        """Web application workload"""
        # Implementation similar to above but web-specific
        # Many small tasks, bursty arrivals, low latency SLAs
        pass
    
    def _generate_batch_workload(self, idx: int, difficulty: str) -> dict:
        """Batch processing workload"""
        # Fewer large tasks, predictable arrivals, high throughput
        pass
    
    def _generate_mixed_workload(self, idx: int, difficulty: str) -> dict:
        """Mixed workload"""
        # Combination of web and batch
        pass
    
    def save_test_set(self, test_set: dict, output_path: str):
        """Save test set to file"""
        with open(output_path, 'w') as f:
            json.dump(test_set, f, indent=2)
        
        # Save statistics
        stats = {
            'total_scenarios': sum(len(scenarios) for scenarios in test_set.values()),
            'by_difficulty': {
                diff: len(scenarios)
                for diff, scenarios in test_set.items()
            }
        }
        
        with open(output_path.replace('.json', '_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved {stats['total_scenarios']} test scenarios to {output_path}")

# Generate test set
if __name__ == '__main__':
    generator = TestSetGenerator(seed=42)
    test_set = generator.generate_comprehensive_test_set()
    generator.save_test_set(test_set, 'data/test_set.json')
```

---



### **Testing Plan for Fix 2**

```bash
# Step 1: Generate test set
python -m aurora.data.test_set_generator

# Step 2: Run comprehensive evaluation
python scripts/run_comprehensive_evaluation.py


```

---

### **Acceptance Criteria for Fix 2**

✅ 100+ test scenarios generated (stratified by difficulty)  
✅ All baselines properly trained (DQN, PPO with tuned hyperparameters)  
✅ Effect sizes computed (Cohen's d)  
✅ Confidence intervals calculated  
✅ Results saved and visualized  

---

### **Time Estimate**: 2-3 weeks

---

## Fix 3: Strong Baselines

### **Current Problem**

**File**: `aurora/ppo_baseline.py`

**Current Code**:
```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Default hyperparameters
```

**Issues**:
- ❌ Default hyperparameters (not tuned)
- ❌ Only 10k timesteps (way too few)
- ❌ No hyperparameter search
- ❌ Missing recent RL methods (SAC, TD3)
- ❌ No optimization solver baseline

**Impact**: Unfair comparison - baselines performing worse than they should.

---

### **Solution**

1. Properly tune PPO/DQN hyperparameters
2. Train for sufficient timesteps (100k-1M)
3. Add SAC (continuous actions)
4. Add OR-Tools optimization solver

---

### **Implementation Steps**

#### Step 3.1: Hyperparameter Tuning

Create `aurora/baselines/tune_hyperparameters.py`:

```python
"""Hyperparameter tuning for RL baselines"""

import optuna
from stable_baselines3 import PPO, DQN
from aurora.envs.gym_env import AuroraGymEnv

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
    env = AuroraGymEnv(scenarios=scenarios)
    
    # Create model
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
    
    # Train
    model.learn(total_timesteps=50000)
    
    # Evaluate
    eval_env = AuroraGymEnv(scenarios=scenarios[:10])  # Subset for speed
    
    mean_reward = 0
    n_eval_episodes = 10
    
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
    
    return mean_reward

def run_hyperparameter_search(scenarios, n_trials=100):
    """Run Optuna hyperparameter search"""
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: optimize_ppo(trial, scenarios),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ Best hyperparameters found:")
    print(study.best_params)
    print(f"Best reward: {study.best_value:.3f}")
    
    # Save best params
    import json
    with open('best_ppo_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params

# Run tuning
if __name__ == '__main__':
    from aurora.data.test_set_generator import TestSetGenerator
    
    generator = TestSetGenerator()
    test_set = generator.generate_comprehensive_test_set()
    all_scenarios = sum(test_set.values(), [])[:50]  # Use 50 for tuning
    
    best_params = run_hyperparameter_search(all_scenarios, n_trials=50)
```

---

#### Step 3.2: Add Optimization Solver Baseline

Create `aurora/baselines/optimization_solver.py`:

```python
"""Optimization solver baseline using OR-Tools"""

from ortools.sat.python import cp_model
import numpy as np

class ORToolsBaseline:
    """
    Use Google OR-Tools CP-SAT solver for optimal allocation
    Only works for small problems (<50 tasks) due to complexity
    """
    
    def __init__(self, timeout_seconds=60):
        self.timeout = timeout_seconds
    
    def allocate(self, scenario):
        """
        Formulate as constraint programming problem and solve
        """
        
        tasks = scenario['tasks']
        nodes = scenario['nodes']
        
        # For large problems, return heuristic
        if len(tasks) > 50:
            print("⚠️ Problem too large for optimization solver, using heuristic")
            return self._heuristic_fallback(scenario)
        
        # Create model
        model = cp_model.CpModel()
        
        # Variables: x[i][j] = 1 if task i assigned to node j
        x = {}
        for i, task in enumerate(tasks):
            for j, node in enumerate(nodes):
                x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
        
        # Constraint 1: Each task assigned to exactly one node
        for i in range(len(tasks)):
            model.Add(sum(x[i, j] for j in range(len(nodes))) == 1)
        
        # Constraint 2: Node capacity constraints
        for j, node in enumerate(nodes):
            # CPU capacity
            model.Add(
                sum(x[i, j] * int(tasks[i]['cpu'] * 100)  # Scale to int
                    for i in range(len(tasks))) 
                <= int(node['cpu_capacity'] * 100)
            )
            
            # Memory capacity
            model.Add(
                sum(x[i, j] * int(tasks[i]['memory'] * 100)
                    for i in range(len(tasks)))
                <= int(node['memory_capacity'] * 100)
            )
        
        # Objective: Minimize total latency (proxy)
        # Latency increases with node load
        total_cost = []
        
        for i, task in enumerate(tasks):
            for j, node in enumerate(nodes):
                # Cost: task_duration * (1 + node_load)
                # Approximate node load as sum of task sizes
                node_load = sum(
                    x[k, j] * int(tasks[k]['cpu'] * 100)
                    for k in range(len(tasks))
                )
                
                cost = x[i, j] * int(task['duration'] * 100)
                total_cost.append(cost)
        
        model.Minimize(sum(total_cost))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            allocations = []
            for i, task in enumerate(tasks):
                for j, node in enumerate(nodes):
                    if solver.Value(x[i, j]) == 1:
                        allocations.append({
                            'task_id': task['id'],
                            'node_id': node['id']
                        })
                        break
            
            return allocations
        else:
            print(f"⚠️ Solver failed with status: {status}, using heuristic")
            return self._heuristic_fallback(scenario)
    
    def _heuristic_fallback(self, scenario):
        """Fallback to best-fit heuristic"""
        from aurora.baselines import BestFit
        baseline = BestFit()
        return baseline.allocate(scenario)
```

---

### **Acceptance Criteria for Fix 3**

✅ PPO trained with tuned hyperparameters (100k+ timesteps)  
✅ DQN trained with tuned hyperparameters  
✅ SAC implemented and trained  
✅ OR-Tools solver baseline working (for small problems)  
✅ All baselines achieve reasonable performance (not random)  
✅ Training curves saved and analyzed

---

### **Time Estimate**: 2 weeks

---

## Fix 4: Real Data Integration

### **Current Problem**

**File**: `aurora/data_loader.py`

**Current Code**:
```python
except Exception as e:
    print(f"❌ Download failed: {e}. Returning synthetic.")
    return self._generate_synthetic_google_like(num_scenarios)
```

**Issues**:
- ❌ Gives up immediately on any error
- ❌ No retry logic
- ❌ Doesn't actually parse traces properly
- ❌ Falls back to synthetic without real attempt

**Impact**: Cannot claim "real-world evaluation" without real data.

---

### **Solution**

1. Implement robust trace downloading
2. Parse traces correctly
3. Extract realistic patterns
4. Validate trace quality
5. Only fall back to synthetic as last resort

---

### **Implementation**

See `real-data-that-will-be-ingested.md` document for complete implementation. Key changes:

```python
class RobustDataLoader:
    def __init__(self):
        self.max_retries = 3
        self.backoff_factor = 2
    
    def load_google_trace(self):
        """Load with retries and validation"""
        for attempt in range(self.max_retries):
            try:
                df = self._download_and_parse()
                if self._validate_trace(df):
                    return self._process_trace(df)
                else:
                    raise ValueError("Trace validation failed")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")
```

---

### **Acceptance Criteria for Fix 4**

✅ Google Cluster Trace successfully downloaded  
✅ At least 50 real scenarios extracted  
✅ Real patterns (distributions) extracted and validated  
✅ Synthetic scenarios match real patterns statistically  
✅ No premature fallback to synthetic

---

### **Time Estimate**: 1 week

---

# HIGH PRIORITY FIXES

## Fix 5: Ablation Studies

**Implementation**: See Fix 2.3 for ablation code structure.

Create system variants:
1. Full AURORA
2. No Attacker (random scenarios)
3. No Policy DB (no memory)
4. No Curriculum (fixed difficulty)
5. No Meta-Evolution

Test each on same scenarios, compare performance.

**Time Estimate**: 1 week

---

## Fix 6: Zero-Shot Transfer

Create `aurora/evaluation/transfer_experiments.py`:

```python
def test_cloud_to_edge_transfer():
    # Train on 100 cloud scenarios
    aurora.train(cloud_scenarios)
    
    # Test on 50 edge scenarios (never seen)
    edge_results = []
    for scenario in edge_scenarios:
        policy = aurora.get_policy(scenario)  # Zero-shot
        result = edge_simulator.execute(scenario, policy)
        edge_results.append(result)
    
    # Compare to baseline trained on edge
    baseline.train(edge_scenarios)
    baseline_results = baseline.evaluate(edge_scenarios)
    
    retention = aurora_perf / baseline_perf
    print(f"Transfer retention: {retention:.1%}")
```

**Time Estimate**: 1-2 weeks

---

## Fix 7: Quality Control Integration

**Current**: Validator exists but not called in main loop.

**Fix**: Integrate validation into iteration loop:

```python
def run_iteration_with_validation(self):
    max_attempts = 5
    
    for attempt in range(max_attempts):
        scenario = self.attacker.generate_scenario()
        
        is_valid, reason = self.validator.validate(scenario)
        
        if is_valid:
            break
        else:
            print(f"Rejected: {reason}")
            self.attacker.add_feedback(reason)
    
    if not is_valid:
        scenario = self.get_seed_scenario()  # Fallback
    
    # Continue with valid scenario
    policy = self.defender.generate_policy(scenario)
    ...
```

**Time Estimate**: 3 days

---

# MEDIUM PRIORITY FIXES

## Fix 8: Explainability Evaluation

Implement human evaluation of policy explanations:

1. Sample 50 policy explanations
2. Score on clarity, correctness, completeness (1-5 scale)
3. Compute inter-rater reliability
4. Report average scores

**Time Estimate**: 1 week (including getting human ratings)

---

## Fix 9: Meta-Prompt Validation

**Current**: Prompts evolve every 10 iterations without validation.

**Fix**: Only apply evolved prompts if they improve performance:

```python
def evolve_with_validation(self, performance_history):
    new_prompt = self._generate_evolved_prompt()
    
    # Test new prompt on held-out scenarios
    test_perf = self._test_prompt(new_prompt, test_scenarios)
    baseline_perf = self._test_prompt(self.current_prompt, test_scenarios)
    
    if test_perf > baseline_perf:
        self.current_prompt = new_prompt
        print("✅ Adopted improved prompt")
    else:
        print("❌ Rejected worse prompt")
```

**Time Estimate**: 3-4 days

---

## Fix 10: Scalability Testing

Test on increasingly large problems:
- 10 tasks → 1000 tasks
- Measure inference time
- Compare to baselines

**Time Estimate**: 3-4 days

---

# SUMMARY & TIMELINE

## Critical Path (Must Do)

| Fix | Time | Can Parallelize? |
|-----|------|------------------|
| 1. Simulator | 1-2 weeks | No (foundation) |
| 2. Evaluation | 2-3 weeks | After Fix 1 |
| 3. Baselines | 2 weeks | Parallel with 2 |
| 4. Real Data | 1 week | Parallel with 2-3 |

**Total Critical Path**: 4-6 weeks if parallelized, 6-8 weeks sequential

## High Priority

| Fix | Time |
|-----|------|
| 5. Ablations | 1 week |
| 6. Transfer | 1-2 weeks |
| 7. Quality Control | 3 days |

**Additional**: 2-3 weeks

## Total Time

**Minimum (Critical only)**: 6 weeks  
**Recommended (Critical + High)**: 8-10 weeks  
**Complete (All fixes)**: 10-12 weeks

---

# RECOMMENDED APPROACH

## Week-by-Week Plan

### Weeks 1-2: Foundation
- [ ] Fix 1: Implement realistic simulator
- [ ] Test simulator thoroughly
- [ ] Validate against toy examples

### Weeks 3-4: Evaluation Infrastructure
- [ ] Fix 2.1: Generate 100+ test scenarios
- [ ] Fix 2.2: Implement statistical tests
- [ ] Fix 3.1: Tune RL hyperparameters
- [ ] Fix 4: Load real traces

### Weeks 5-6: Run Experiments
- [ ] Fix 2.3: Run comprehensive evaluation
- [ ] Train all baselines properly
- [ ] Collect results on all scenarios
- [ ] Run statistical analysis

### Weeks 7-8: Advanced Experiments
- [ ] Fix 5: Ablation studies
- [ ] Fix 6: Transfer experiments
- [ ] Fix 7: Integrate quality control
- [ ] Generate all plots and tables

### Weeks 9-10: Polish & Paper
- [ ] Fix 8: Explainability evaluation
- [ ] Fix 9-10: Meta-learning and scalability
- [ ] Write paper
- [ ] Prepare submission

---

# GETTING STARTED

## Immediate Next Steps

1. **Start with Fix 1** (Realistic Simulator)
   ```bash
   # Create branch
   git checkout -b fix/realistic-simulator
   
   # Implement event simulator
   touch aurora/simulators/event_simulator.py
   # Copy code from Fix 1
   
   # Test
   python -m pytest tests/test_realistic_simulator.py
   ```

2. **While Fix 1 is ongoing, start Fix 4** (Real Data)
   ```bash
   # Download traces
   python scripts/download_traces.py
   
   # Validate
   python scripts/validate_traces.py
   ```

3. **After Fix 1 complete, move to Fix 2** (Evaluation)

---

# SUCCESS METRICS

After completing all fixes, you should be able to claim:

✅ "Evaluated on 100+ diverse scenarios"  
✅ "Statistically significant improvement over 6 baselines (p < 0.05)"  
✅ "Validated on Google Cluster Trace and Azure datasets"  
✅ "Demonstrated zero-shot transfer with >70% retention"  
✅ "Ablation studies confirm each component contributes"  
✅ "Realistic discrete-event simulation with queuing"

This transforms your paper from **4.5/10** → **8/10** publication-ready.

---
