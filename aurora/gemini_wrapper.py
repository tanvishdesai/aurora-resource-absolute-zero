import os
import time
import logging
import google.generativeai as genai
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.key_status = {
            key: {
                "rpm_count": 0,
                "rpd_count": 0,
                "last_rpm_reset": datetime.now(),
                "retired_for_day": False,
                "cooldown_until": datetime.min
            } 
            for key in api_keys
        }
        self.rpm_limit = 10
        self.rpd_limit = 250
        self.rpm_window = 60 # seconds

    def _reset_rpm_counters(self):
        now = datetime.now()
        for key, stats in self.key_status.items():
            if (now - stats["last_rpm_reset"]).total_seconds() >= self.rpm_window:
                stats["rpm_count"] = 0
                stats["last_rpm_reset"] = now

    def _mark_key_retired_for_day(self, key):
        self.key_status[key]["retired_for_day"] = True
        logger.warning(f"⚠️ API Key {key[:8]}... has hit its DAILY limit ({self.rpd_limit}) and is being retired for the day.")

    def get_available_key(self) -> Optional[str]:
        self._reset_rpm_counters()
        
        # Try to find a key that is valid
        for key in self.api_keys:
            stats = self.key_status[key]
            
            if stats["retired_for_day"]:
                continue

            if stats["cooldown_until"] > datetime.now():
                continue
                
            if stats["rpd_count"] >= self.rpd_limit:
                self._mark_key_retired_for_day(key)
                continue
                
            if stats["rpm_count"] < self.rpm_limit:
                return key
                
        return None

    def record_usage(self, key):
        if key in self.key_status:
            self.key_status[key]["rpm_count"] += 1
            self.key_status[key]["rpd_count"] += 1

    def report_rate_limit_error(self, key):
        # If we hit a rate limit error despite our tracking, cool it down for a minute
        logger.warning(f"⚠️ Rate limit hit for key {key[:8]}... Rotating.")
        self.key_status[key]["cooldown_until"] = datetime.now() + timedelta(seconds=60)

class GeminiHandler:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        if not api_keys:
            raise ValueError("No API keys provided for GeminiHandler")
            
        self.key_manager = KeyManager(api_keys)
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        logger.info(f"Initialized GeminiHandler with {len(api_keys)} keys.")

    def generate(self, prompt: str, system_instruction: str = None, max_new_tokens: int = 2048) -> str:
        """
        Generates content using Gemini API with key rotation and retries.
        Matches the interface of QwenHandler.
        """
        retries = 0
        max_retries = len(self.key_manager.api_keys) * 2 # Allow cycling through keys twice
        
        current_instruction = system_instruction or ""
        # Create a combined prompt if system instruction is simple, 
        # but Gemini supports system_instruction on model init or generate (depending on lib version).
        # We will wrap it in user structure for simplicity or use the new API if available.
        # Check if google-generativeai supports system_instruction argument in GenerativeModel (v0.3.0+)
        
        while retries < max_retries:
            key = self.key_manager.get_available_key()
            
            if not key:
                logger.warning("All keys exhausted or rate limited. Waiting 10 seconds...")
                time.sleep(10)
                retries += 1
                continue
            
            try:
                genai.configure(api_key=key)
                # Initialize model with system instruction if possible, but keeping it simple for rotation
                # Pass system instruction as part of messages or config if needed
                
                # Simple generation for now
                model = genai.GenerativeModel(self.model_name)
                
                full_prompt = prompt
                if system_instruction:
                    full_prompt = f"System Instruction: {system_instruction}\n\nUser: {prompt}"

                self.key_manager.record_usage(key)
                
                logger.info(f"Generating with Gemini (Key: {key[:8]}...)")
                response = model.generate_content(
                    full_prompt, 
                    generation_config={**self.generation_config, "max_output_tokens": max_new_tokens}
                )
                
                return response.text

            except Exception as e:
                error_str = str(e)
                logger.error(f"Error with key {key[:8]}...: {e}")
                
                if "429" in error_str or "Resource has been exhausted" in error_str:
                    self.key_manager.report_rate_limit_error(key)
                else:
                    # Non-rate-limit error (e.g. safety, blocked), maybe don't retry locally but for now retry
                   logger.error(f"Non-retriable error or unknown: {e}")
                   # Optional: break if safety violation to avoid loop
                
                retries += 1
                time.sleep(1) # Short backoff
        
        raise RuntimeError(f"Failed to generate content after {retries} retries. All keys may be exhausted.")
