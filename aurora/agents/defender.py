import json
from aurora import config
from aurora.agents.prompts import DEFENDER_SYSTEM_PROMPT

class DefenderAgent:
    def __init__(self, llm_handler, policy_db=None):
        self.llm_handler = llm_handler
        self.policy_db = policy_db # Store reference to the DB
        self.system_prompt = DEFENDER_SYSTEM_PROMPT

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
        try:
            # Note: Max retries loop removed as rate limits are not an issue with local models
            response_text = self.llm_handler.generate(prompt, system_instruction=self.system_prompt)
            
            # Robust JSON extraction
            text = response_text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx : end_idx + 1]
            else:
                text = text.replace("```json", "").replace("```", "").strip()
                
            return json.loads(text)
        except Exception as e:
            print(f"Error (Defender): {e}")
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
