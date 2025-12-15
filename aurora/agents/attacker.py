import json
import google.generativeai as genai
from aurora import config
from aurora.agents.prompts import ATTACKER_SYSTEM_PROMPT

class AttackerAgent:
    def __init__(self, model_name=config.MODEL_NAME):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=ATTACKER_SYSTEM_PROMPT,
            generation_config=config.GENERATION_CONFIG
        )

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

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                # Clean up potential markdown formatting if the model adds it strictly for JSON
                text = response.text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
            except Exception as e:
                is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
                
                if is_rate_limit and attempt < max_retries:
                    print(f"⚠️  Rate limit hit (Attacker). keys available. Rotating key and retrying... (Attempt {attempt+1}/{max_retries})")
                    config.rotate_api_key()
                    # Re-initialize model to pick up new config/key if necessary
                    # (Note: genai.configure might adhere globally, but re-init is safer)
                    self.model = genai.GenerativeModel(
                        model_name=self.model.model_name,
                        system_instruction=self.model._system_instruction,
                        generation_config=self.model._generation_config
                    )
                    continue
                
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
