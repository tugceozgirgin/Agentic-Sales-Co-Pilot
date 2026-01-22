"""
PPO Fine-tuning for Reflection Agent using TRL (Transformer Reinforcement Learning)

This script fine-tunes the TinyLlama model to better evaluate CRM agent outputs
using Proximal Policy Optimization (PPO) with human feedback data.

Requirements:
    pip install trl peft accelerate datasets
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import re


@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "models/reflection_agent_ppo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    ppo_epochs: int = 4
    max_length: int = 512
    num_train_epochs: int = 3


class ReflectionRewardModel:
    def __init__(self):
        self.correct_patterns = [
            r'"correctness":\s*true',
            r'correctness.*true',
            r'correct.*used.*tool',
            r'appropriate.*tool',
        ]
        self.incorrect_patterns = [
            r'"correctness":\s*false',
            r'correctness.*false',
            r'should have used',
            r'missing.*tool',
            r'only used one',
        ]
    
    def compute_reward(
        self, 
        generated_text: str, 
        expected_correctness: bool,
        expected_tools: List[str],
        actual_tools: List[str]
    ) -> float:
        """
        Compute reward based on:
        1. Whether the model correctly identified correctness
        2. Whether the rationale mentions appropriate tools
        """
        reward = 0.0
        text_lower = generated_text.lower()

        predicted_correct = self._extract_correctness(generated_text)
        
        if predicted_correct == expected_correctness:
            reward += 1.0  
        else:
            reward -= 1.0 
        
        if expected_correctness:
            if any(tool.lower() in text_lower for tool in actual_tools):
                reward += 0.5
        else:
            missing_tools = set(expected_tools) - set(actual_tools)
            if missing_tools:
                if any(tool.lower() in text_lower for tool in missing_tools):
                    reward += 0.5
                if "should" in text_lower or "missing" in text_lower:
                    reward += 0.3
        
        if self._is_valid_json_output(generated_text):
            reward += 0.3
        
        return reward
    
    def _extract_correctness(self, text: str) -> bool:
        json_match = re.search(r'"correctness":\s*(true|false)', text.lower())
        if json_match:
            return json_match.group(1) == 'true'
        

        if 'correctness: true' in text.lower() or 'correctness":true' in text.lower():
            return True
        if 'correctness: false' in text.lower() or 'correctness":false' in text.lower():
            return False

        positive = ['correct', 'appropriate', 'proper']
        negative = ['incorrect', 'wrong', 'should have', 'missing']
        
        has_positive = any(p in text.lower() for p in positive)
        has_negative = any(n in text.lower() for n in negative)
        
        if has_negative and not has_positive:
            return False
        return True
    
    def _is_valid_json_output(self, text: str) -> bool:
        try:
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                json.loads(json_match.group())
                return True
        except:
            pass
        return False


class RLHFTrainer:
    def __init__(self, config: RLHFConfig = None):
        self.config = config or RLHFConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_model = ReflectionRewardModel()
        
        print(f"Initializing RLHF Trainer on device: {self.device}")
        
    def load_training_data(self, data_path: str = None) -> List[Dict[str, Any]]:
        if data_path is None:
            data_path = Path(__file__).parent / "rlhf.json"
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} training examples")
        return data
    
    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Build reflection prompt from training example"""
        tools_used = ", ".join(example["tool_calls_made"]) if example["tool_calls_made"] else "None"
        
        prompt = f"""Evaluate this CRM agent response:

User Query: {example["user_input"]}
CRM Output: {example["crm_output"]}
Tools Used: {tools_used}

Is the CRM response correct? Output JSON: {{"correctness": true/false, "rationale": "explanation"}}

Evaluation:"""
        return prompt
    
    def prepare_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        prompts = []
        for example in data:
            prompt = self.build_prompt(example)
            prompts.append({
                "query": prompt,
                "expected_correctness": example["correctness"],
                "expected_tools": self._get_expected_tools(example["user_input"]),
                "actual_tools": example["tool_calls_made"],
            })
        
        return Dataset.from_list(prompts)
    
    def _get_expected_tools(self, user_input: str) -> List[str]:
        input_lower = user_input.lower()

        if any(kw in input_lower for kw in ["recommend", "pitch", "suggest", "based on"]):
            return ["search_structured_db", "search_semantic_db"]

        if any(kw in input_lower for kw in ["how much", "spent", "manager", "industry", "when", "last meeting"]):
            return ["search_structured_db"]
        
        if any(kw in input_lower for kw in ["policy", "rule", "discount", "commission", "process"]):
            return ["search_semantic_db"]
        
        if any(kw in input_lower for kw in ["weather", "email", "help me write"]):
            return []
        
        return []
    
    def train(self, data_path: str = None):
        print("=" * 50)
        print("Starting PPO Fine-tuning for Reflection Agent")
        print("=" * 50)

        training_data = self.load_training_data(data_path)
        dataset = self.prepare_dataset(training_data)
        
        print(f"\nLoading tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model with value head...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            log_with=None,
        )
        
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )

        print(f"\nStarting training for {self.config.num_train_epochs} epochs...")
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_train_epochs} ---")
            
            epoch_rewards = []
            
            for i, example in enumerate(dataset):
                query_tensors = tokenizer(
                    example["query"],
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                ).input_ids.squeeze()
                
                response_tensors = ppo_trainer.generate(
                    query_tensors.unsqueeze(0),
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                response_text = tokenizer.decode(
                    response_tensors[0][len(query_tensors):],
                    skip_special_tokens=True
                )
                
                reward = self.reward_model.compute_reward(
                    generated_text=response_text,
                    expected_correctness=example["expected_correctness"],
                    expected_tools=example["expected_tools"],
                    actual_tools=example["actual_tools"],
                )
                
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                epoch_rewards.append(reward)
                
                stats = ppo_trainer.step(
                    [query_tensors],
                    [response_tensors[0][len(query_tensors):]],
                    [reward_tensor]
                )
                
                if (i + 1) % 5 == 0:
                    print(f"  Step {i + 1}/{len(dataset)}, Reward: {reward:.2f}")
            
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"Epoch {epoch + 1} Average Reward: {avg_reward:.3f}")
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving fine-tuned model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("Training complete!")
        return output_path


def create_synthetic_examples(num_examples: int = 50) -> List[Dict[str, Any]]:
    """Generate additional synthetic training examples"""
    
    clients = ["TechGiant", "GlobalRetail", "MegaCorp", "StartupXYZ", "HealthCo", 
               "AutoDrive", "MediaGroup", "DataCorp", "FinanceFirst", "RetailMax"]
    
    industries = ["Electronics", "Retail", "Finance", "Healthcare", "Automotive",
                  "Media", "Technology", "Manufacturing", "Energy", "Telecom"]
    
    managers = ["John", "Sarah", "Mike", "Emily", "David", "Lisa", "Tom", "Anna"]
    
    fact_templates = [
        ("What is {client}'s total spend?", ["search_structured_db"], True),
        ("Who manages {client}?", ["search_structured_db"], True),
        ("When was the last meeting with {client}?", ["search_structured_db"], True),
        ("What industry is {client} in?", ["search_structured_db"], True),
    ]
    
    policy_templates = [
        ("What's the discount policy for new clients?", ["search_semantic_db"], True),
        ("What are the rules for free trials?", ["search_semantic_db"], True),
        ("How do we handle client complaints?", ["search_semantic_db"], True),
    ]
    
    recommendation_templates = [
        ("What should I pitch to {client}?", ["search_structured_db", "search_semantic_db"], True),
        ("Recommend products for {client} based on their profile", ["search_structured_db", "search_semantic_db"], True),
    ]
    
    irrelevant_templates = [
        ("What's the weather today?", [], True),
        ("Help me write a poem", [], True),
        ("What's 2+2?", [], True),
    ]
    
    examples = []
    import random
    
    for _ in range(num_examples):
        template_type = random.choice(["fact", "policy", "recommendation", "irrelevant"])
        
        if template_type == "fact":
            template, tools, correct = random.choice(fact_templates)
            client = random.choice(clients)
            query = template.format(client=client)
        elif template_type == "policy":
            query, tools, correct = random.choice(policy_templates)
        elif template_type == "recommendation":
            template, tools, correct = random.choice(recommendation_templates)
            client = random.choice(clients)
            query = template.format(client=client)
        else:
            query, tools, correct = random.choice(irrelevant_templates)
        
        examples.append({
            "user_input": query,
            "crm_output": f"Sample CRM response for: {query}",
            "tool_calls_made": tools if correct else tools[:1] if tools else [],
            "reflection_output": "Sample reflection",
            "correctness": correct
        })
    
    return examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF PPO Training for Reflection Agent")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/reflection_agent_ppo", help="Output directory")
    
    args = parser.parse_args()
    
    config = RLHFConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    trainer = RLHFTrainer(config)
    trainer.train(args.data)
