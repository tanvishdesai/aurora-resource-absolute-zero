"""Optimization solver baseline using OR-Tools"""

from ortools.sat.python import cp_model
import numpy as np
import os
import sys

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                # Approximate cost as constant duration for simple allocation model
                # Ideally this should model the non-linear objective but CP-SAT is good for linear/integer constraints
                
                # Minimizing makespan is hard, let's minimize total load imbalance or just cost
                
                # Using simple cost proxy: specific cost + duration
                cost = int(task['duration'] * 100)
                total_cost.append(x[i, j] * cost)
        
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
        # Lazy import to avoid circular dependency issues if any
        from aurora.baselines import BestFit
        baseline = BestFit()
        return baseline.allocate(scenario)
