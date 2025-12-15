import json
from aurora import config
from aurora.agents.prompts import ATTACKER_SYSTEM_PROMPT

class AttackerAgent:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        self.system_prompt = ATTACKER_SYSTEM_PROMPT

    def generate_scenario(self, difficulty_level, defender_performance):
        """
        Generates a scenario based on difficulty and past performance.
        """
        
        prompt = f"""
        Generate a resource allocation scenario.
        Difficulty Level: {difficulty_level}
        Defender Performance on last run: {defender_performance}
        
        Target complexity:
        - Number of tasks: {difficulty_level * 5}
        - Number of nodes: {max(2, difficulty_level * 2)}
        """

        try:
            response_text = self.llm_handler.generate(prompt, system_instruction=self.system_prompt)
            # Robust JSON extraction: look for first { and last }
            text = response_text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx : end_idx + 1]
            else:
                # Fallback to simple cleanup if braces not strictly found (unlikely for valid JSON)
                text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)
        except Exception as e:
            print(f"Error generating scenario: {e}")
            # Fallback for stability
            return self._fallback_scenario()

    def _fallback_scenario(self):
        return {
            "scenario_id": "fallback_001",
            "description": "Fallback scenario",
            "tasks": [{"id": "t1", "cpu": 1, "memory": 512, "duration": 10, "sla_latency_ms": 100, "priority": 1}],
            "nodes": [{"id": "n1", "cpu_capacity": 4, "memory_capacity": 2048, "energy_efficiency": 1.0, "cost_per_hour": 0.1}]
        }
