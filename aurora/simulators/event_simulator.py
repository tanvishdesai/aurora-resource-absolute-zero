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
            -task.get('priority', 0),  # Negative for max-heap, default 0 if missing
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
        # Iterate through the queue to find the first schedulable task
        # Note: altering heap during iteration is tricky, so we might need a better approach for production M/G/1
        # For this implementation we'll scan and pop.
        for i, (priority, arrival, task) in enumerate(self.queue):
            if self.can_schedule(task):
                self.queue.pop(i)
                heapq.heapify(self.queue) # Re-heapify after removal
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
        
        # M/G/1 mean waiting time assumption for service degradation implementation
        # Simplified: service time increases as 1/(1-ρ)
        if cpu_load < 0.95:  # Prevent division by zero
            service_time = base_time / (1 - cpu_load)
        else:
            service_time = base_time * 20  # Heavy penalty for overload
        
        # Add variance (coefficient of variation = 1 for exponential)
        variance = base_time * np.random.exponential(1.0)
        
        return service_time + variance * 0.1  # Small variance component

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
        if new_time < self.current_time:
             # This can happen if events are scheduled in the past or same time, just ignore/warn if strictly needed
             return 

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
        P_idle = 100.0  # Watts (idle power)
        P_max = 300.0   # Watts (max power)
        
        if node_state['cpu_capacity'] > 0:
            cpu_util = node_state['cpu_used'] / node_state['cpu_capacity']
        else:
            cpu_util = 0.0
        
        # Quadratic relationship (based on research)
        power = P_idle + (P_max - P_idle) * (cpu_util ** 2)
        
        return power

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
        self.statistics = {
            'task_completions': [],
            'node_utilization_history': [],
            'energy_consumption': 0.0,
            'network_bytes_transferred': 0
        }
        
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
                'cpu': task['cpu'], # Store CPU for stats
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
        
        if node_id not in self.node_states:
             # Node ID from policy doesn't exist in scenario
            self.task_states[task_id]['status'] = 'failed'
            self.task_states[task_id]['completion_time'] = event.time
            return

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
    
    def _handle_node_failure(self, event: Event):
        # Implementation placeholder
        pass

    def _handle_node_recovery(self, event: Event):
        # Implementation placeholder
        pass

    def _compute_network_delay(self, task: dict, node_id: str) -> float:
        """
        Compute network transfer delay
        Delay = data_size / bandwidth + base_latency
        """
        data_size_mb = task.get('input_data_mb', 10)  # Default 10MB
        bandwidth_mbps = 1000.0  # 1 Gbps
        
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
                'avg_cpu_utilization': 0,
                'task_completions': []
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
        if total_capacity == 0:
            total_capacity = 1 # Avoid div by zero
            
        total_cpu_seconds = 0
        for c in completions:
            task_cpu = self.task_states[c['task_id']]['cpu']
            duration = c['completion_time'] - c['start_time']
            total_cpu_seconds += duration * task_cpu

        avg_cpu_utilization = total_cpu_seconds / (self.current_time * total_capacity) if self.current_time > 0 else 0
        
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
