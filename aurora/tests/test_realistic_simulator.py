import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
                'cpu_capacity': 10,
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
    
    if not completions:
        pytest.fail("No tasks completed")

    # Task 0: should complete near 10s (no queue)
    # Allow some buffer for overhead/network simulation defaults
    print(f"Task 0 Latency: {completions[0]['latency']}")
    assert completions[0]['latency'] < 15
    
    # Task 9: should experience significant queuing
    # It arrives at t=18.
    # Previous tasks take 10s each. Capacity is 2.
    # Throughput is 2 tasks/10s = 0.2 tasks/s.
    # Arrival rate is 0.5 tasks/s (1 task every 2s).
    # System is overloaded (lambda > mu), so queue grows.
    print(f"Task 9 Latency: {completions[9]['latency']}")
    assert completions[9]['latency'] > 20  # Waited in queue significantly

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
    
    if not results['task_completions']:
        pytest.fail("Task failed to complete")

    completion = results['task_completions'][0]
    
    # Latency should include network delay
    # 100MB / 1000 Mbps = 0.1s transfer time (or 0.8s if bits vs bytes, simulator uses 1000 Mbps)
    # Let's check the code: event_simulator.py says: 
    # bandwidth_mbps = 1000.0  # 1 Gbps
    # transfer_time = data_size_mb / bandwidth_mbps
    # 100 / 1000 = 0.1 seconds.
    # base_latency = random normal(1.0, 0.5) ms -> ~0.001s
    # Total extra delay ~ 0.1s. 
    # Duration is 5s.
    # So total should be > 5.05s.
    
    print(f"Network Delay Test Latency: {completion['latency']}")
    assert completion['latency'] > 5.0

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
            'memory_capacity': 32,
            'energy_efficiency': 1.0
        }]
    }
    
    policy = {
        'allocations': [{'task_id': 't1', 'node_id': 'n1'}]
    }
    
    simulator = UnifiedResourceSimulator()
    results = simulator.execute_policy(scenario, policy)
    
    # Energy should be > 0
    assert results['total_energy'] > 0
    
    # P_idle = 100, P_max = 300
    # Util = 8/16 = 0.5
    # Power = 100 + (200) * 0.5^2 = 100 + 50 = 150 W
    # Load is 0.5, so service time doubles to 7200s (2h).
    # Energy = 150W * 2h = 300 Wh = 0.3 kWh.
    print(f"Total Energy: {results['total_energy']}")
    assert 0.2 < results['total_energy'] < 0.5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
