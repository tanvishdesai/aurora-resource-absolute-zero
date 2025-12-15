from aurora.agents.attacker import AttackerAgent
from aurora.agents.defender import DefenderAgent
from aurora.simulators.simulator import UnifiedResourceSimulator
from aurora.rewards.reward_engine import MultiObjectiveRewardEngine
from aurora.evolution.policy_db import PolicyDatabase
from aurora.evolution.curriculum import CurriculumManager
from aurora.evolution.meta_prompts import MetaPromptEvolver
from aurora.validation import ScenarioValidator

class MetaEvolutionaryController:
    def __init__(self, llm_handler):
        # Components
        self.llm_handler = llm_handler

        # 1. Initialize DB first
        self.policy_db = PolicyDatabase() 
        self.validator = ScenarioValidator()
        self.attacker = AttackerAgent(llm_handler=self.llm_handler)
        # 2. Pass DB to Defender
        self.defender = DefenderAgent(llm_handler=self.llm_handler, policy_db=self.policy_db)
        
        self.simulator = UnifiedResourceSimulator()
        self.reward_engine = MultiObjectiveRewardEngine()
        
        # State & Learning
        self.curriculum = CurriculumManager()
        self.prompt_evolver = MetaPromptEvolver(llm_handler=self.llm_handler)
        
        self.iteration_count = 0

    def run_iteration(self, iteration_idx=None):
        """
        Executes one full self-play loop.
        """
        if iteration_idx is not None:
             self.iteration_count = iteration_idx + 1
        else:
             self.iteration_count += 1
        print(f"\n--- Iteration {self.iteration_count} [Difficulty: {self.curriculum.get_current_level_name()}] ---")

        # 1. Attacker generates scenario
        difficulty = self.curriculum.get_current_difficulty()
        # Mocking defender performance for now, or use rolling average
        defender_perf = "Average" 
        
        print(">> Attacker generating scenario...")
        
        # Validation Loop (Fix 7)
        max_attempts = 5
        is_valid = False
        scenario = None
        
        for attempt in range(max_attempts):
            scenario = self.attacker.generate_scenario(difficulty, defender_perf)
            is_valid, reason = self.validator.validate(scenario)
            
            if is_valid:
                break
            else:
                print(f"⚠️ Scenario Rejected (Attempt {attempt+1}): {reason}")
                if hasattr(self.attacker, 'add_feedback'):
                    self.attacker.add_feedback(reason)
        
        if not is_valid:
            print("❌ Max validation attempts reached. Falling back to seed scenario.")
            # Fallback logic: generate a simple valid scenario manually if seed not available
            # Or use one from curriculum if possible. For now, simplest fallback:
            from aurora.data.test_set_generator import TestSetGenerator
            gen = TestSetGenerator()
            scenario = gen._generate_easy_scenario(0) # guaranteed valid
            
        print(f"Scenario: {scenario.get('description', 'No description')}")

        # 2. Defender generates policy
        print(">> Defender generating policy...")
        policy = self.defender.generate_policy(scenario)
        # print(f"Policy reasoning: {policy.get('reasoning', 'No reasoning')}")

        # 3. Simulation
        print(">> Simulating...")
        results = self.simulator.execute_policy(scenario, policy)
        
        # 4. Evaluation
        reward, scores = self.reward_engine.compute_reward(results, scenario)
        print(f"RESULTS: Reward={reward:.4f} | Latency={results.get('avg_latency',0):.2f} | SLA Violations={results.get('sla_violations',0)}")

        # 5. Evolution / Learning Update
        if reward > 0.4: # Filter for decent policies (Lowered for initial learning)
            self.policy_db.add_policy(policy, results, reward)
        
        self.curriculum.record_performance(reward)
        if self.curriculum.check_progression():
            print(">>> LEVEL UP! Curriculum Advanced. <<<")

        # 6. Meta-Prompt Evolution (Periodic)
        if self.iteration_count % 10 == 0:
            # Gather data
            history = self.curriculum.performance_history
            rejection_log = self.attacker.validation_log if hasattr(self.attacker, 'validation_log') else []
            
            new_attacker_p, new_defender_p = self.prompt_evolver.evolve_prompts(history, rejection_log)
            
            # Apply Updates (Update agent system prompts)
            if new_attacker_p and new_attacker_p != self.prompt_evolver.current_attacker_prompt:
                print("✨ APPLYING NEW ATTACKER PROMPT")
                self.prompt_evolver.current_attacker_prompt = new_attacker_p
                # Update Attacker Agent Prompt
                self.attacker.system_prompt = new_attacker_p

            if new_defender_p and new_defender_p != self.prompt_evolver.current_defender_prompt:
                print("✨ APPLYING NEW DEFENDER PROMPT")
                self.prompt_evolver.current_defender_prompt = new_defender_p
                # Update Defender Agent Prompt
                self.defender.system_prompt = new_defender_p

        return {
            "iteration": self.iteration_count,
            "difficulty": self.curriculum.get_current_level_name(),
            "reward": reward,
            "scores": scores,
            "results": results
        }
