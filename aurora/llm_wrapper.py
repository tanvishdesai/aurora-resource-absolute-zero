
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from aurora import config

logger = logging.getLogger(__name__)

class QwenHandler:
    def __init__(self, model_name_or_path=None):
        self.model_name = model_name_or_path or config.MODEL_PATH
        logger.info(f"Loading Qwen model from: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_name}: {e}")
            raise e

    def generate(self, prompt, system_instruction=None, max_new_tokens=2048):
        """
        Generates content using the Qwen model with chat template.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        logger.info("Generating content...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=self.streamer
            )
            
        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
