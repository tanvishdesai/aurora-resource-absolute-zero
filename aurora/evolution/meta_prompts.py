from aurora import config
from aurora.agents.prompts import ATTACKER_SYSTEM_PROMPT, DEFENDER_SYSTEM_PROMPT

class MetaPromptEvolver:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler # Use the shared handler
        self.current_attacker_prompt = ATTACKER_SYSTEM_PROMPT
        self.current_defender_prompt = DEFENDER_SYSTEM_PROMPT

    def evolve_prompts(self, performance_history, rejection_log=None):
        """
        Analyze performance and logs to evolve system prompts.
        Returns: (new_attacker_prompt, new_defender_prompt) or (None, None) if no update.
        """
        print("🧬 Running Meta-Prompt Evolution...")
        
        # 1. Analyze Defender Performance
        # If rewards are stagnating or low, update Defender
        avg_reward = 0
        if performance_history:
            avg_reward = sum(performance_history) / len(performance_history)
        
        updated_defender = self.current_defender_prompt
        if avg_reward < 0.6: # Stagnation threshold
            print("   -> Optimizing Defender Prompt (Low Reward)")
            updated_defender = self._optimize_prompt(
                role="Defender",
                current_prompt=self.current_defender_prompt,
                context=f"The agent is stuck at average reward {avg_reward:.2f}. It needs to be encouraged to explore better packing strategies (First Fit Decreasing equivalent)."
            )

        # 2. Analyze Attacker (Rejection Rate)
        # If many scenarios are rejected, Attacker needs to be corrected
        updated_attacker = self.current_attacker_prompt
        if rejection_log and len(rejection_log) > 5:
            print("   -> Optimizing Attacker Prompt (High Rejection)")
            # Summarize reasons
            reasons = [entry['reason'] for entry in rejection_log[-5:]]
            context = f"The agent is generating invalid scenarios. Common rejection reasons: {reasons}. It needs to strictly follow the constraints."
            
            updated_attacker = self._optimize_prompt(
                role="Attacker",
                current_prompt=self.current_attacker_prompt,
                context=context
            )
            
        return updated_attacker, updated_defender

    def _optimize_prompt(self, role, current_prompt, context):
        """
        Asks the LLM to rewrite the prompt based on feedback.
        """
        print(f"      Asking Meta-Architect to rewrite {role} prompt...")
        
        meta_prompt = f"""
        You are an expert AI Architect optimizing system prompts for a Multi-Agent System.
        
        ROLE: {role}
        CURRENT PROMPT:
        {current_prompt}
        
        PROBLEM CONTEXT:
        {context}
        
        TASK:
        Rewrite the CURRENT PROMPT to address the problem. 
        Keep the format and strict constraint sections intact, but improve the strategy instructions.
        Return ONLY the new prompt text.
        """
        
        try:
            response_text = self.llm_handler.generate(meta_prompt)
            new_prompt = response_text.replace("```", "").strip()
            return new_prompt
        except Exception as e:
            print(f"      ⚠️ Evolution failed: {e}")
            return current_prompt
