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
