from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os


class Models:
    _local_llm_instance = None
    
    @staticmethod
    def get_openai_model(model_name: str, temperature: float = 0.0):
        return ChatOpenAI(model=model_name, temperature=temperature)
    
    @staticmethod
    def get_anthropic_model(model_name: str, temperature: float = 0.0):
        return ChatAnthropic(model=model_name, temperature=temperature)
    
    @staticmethod
    def get_local_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature: float = 0.1):
        """
        Load a local HuggingFace model. Default is TinyLlama (1.1B params, ~2GB).
        - Fast to load on CPU
        - Small memory footprint
        - Can be fine-tuned with PPO/RLHF
        """
        if Models._local_llm_instance is None:
            print(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model - simple loading that works on both CPU and GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                model = model.to(device)
            
            print("Model loaded!")
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            Models._local_llm_instance = HuggingFacePipeline(pipeline=pipe)
        
        return Models._local_llm_instance
    
    # Alias for backward compatibility
    @staticmethod
    def get_vicuna_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature: float = 0.1):
        """Alias for get_local_model - uses TinyLlama by default (RLHF trainable)"""
        return Models.get_local_model(model_name, temperature)


# if __name__ == "__main__":
#     print("Testing TinyLlama model...")
#     model = Models.get_local_model()
#     response = model.invoke("What is 2+2? Answer briefly:")
#     print(f"Response: {response}")
