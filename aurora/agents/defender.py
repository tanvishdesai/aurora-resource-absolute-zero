import json
import google.generativeai as genai
from aurora import config
from aurora.agents.prompts import DEFENDER_SYSTEM_PROMPT

class DefenderAgent:
    def __init__(self, policy_db=None, model_name=config.MODEL_NAME):
        self.model = genai.GenerativeModel(
            model_name=model_name, # "gemini-2.0-flash-exp"
            system_instruction=DEFENDER_SYSTEM_PROMPT,
            generation_config=config.GENERATION_CONFIG
        )
        self.policy_db = policy_db # Store reference to the DB

    def generate_policy(self, scenario):
        """
        Generates an allocation policy.
        If a PolicyDB is present, it retrieves a past success example to guide the LLM.
        """
        
        # 1. RETRIEVAL: Try to find a helpful example from training
        example_context = ""
        if self.policy_db:
            # Currently retrieving the absolute best, can be improved to similarity search later
            best_policy = self.policy_db.retrieve_similar_policy(scenario)
            if best_policy and 'policy' in best_policy:
                # We strip strict JSON formatting from the example to save tokens/confusion
                example_desc = json.dumps(best_policy['policy']['allocations'])
                reasoning = best_policy['policy'].get('reasoning', 'Efficient allocation.')
                
                example_context = f"""
                Here is a similar scenario you solved successfully in the past:
                PAST STRATEGY: {reasoning}
                PAST ALLOCATION: {example_desc}
                
                Use the strategy above as a reference to solve the new scenario below.
                """

        # 2. PROMPT CONSTRUCTION
        scenario_str = json.dumps(scenario)
        
        prompt = f"""
        {example_context}
        
        NEW SCENARIO TO SOLVE:
        {scenario_str}
        
        Create an optimized allocation policy.
        """

        # 3. GENERATION
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                text = response.text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
            except Exception as e:
                is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
                
                if is_rate_limit and attempt < max_retries:
                    print(f"⚠️  Rate limit hit (Defender). Rotating key... (Attempt {attempt+1})")
                    config.rotate_api_key()
                    # Re-instantiate model with new key
                    self.model = genai.GenerativeModel(
                        model_name=self.model.model_name,
                        system_instruction=self.model._system_instruction,
                        generation_config=self.model._generation_config
                    )
                    continue

                # print(f"Error: {e}") # debug
                return self._fallback_policy(scenario)

    def _fallback_policy(self, scenario):
        # Naive First-Fit
        allocations = []
        nodes = scenario.get("nodes", [])
        for task in scenario.get("tasks", []):
            if nodes:
                # Very naive: assign all to first node (likely to fail but valid format)
                # Or round robin
                node = nodes[0]
                allocations.append({"task_id": task["id"], "node_id": node["id"]})
        return {
            "policy_id": "fallback_strategy",
            "reasoning": "Fallback First-Fit",
            "allocations": allocations
        }
