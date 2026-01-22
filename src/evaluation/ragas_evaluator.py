"""
RAGAS-based evaluation for CRM Agent RAG system.
Evaluates retrieval quality, answer faithfulness, and overall correctness.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("[WARNING] RAGAS not installed. Install with: pip install ragas")


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float
    overall_score: float
    details: Dict[str, Any]


class RAGASEvaluator:
    """
    RAGAS evaluator for CRM Agent responses.
    Evaluates:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Are retrieved contexts relevant?
    - Context Recall: Are all relevant contexts retrieved?
    - Answer Correctness: Is the answer factually correct?
    """
    
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS is not installed. Install with: pip install ragas"
            )
        
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        import numpy as np
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if not answer or not answer.strip():
            return EvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                answer_correctness=0.0,
                overall_score=0.0,
                details={}
            )
        
        valid_contexts = [ctx for ctx in contexts if ctx and ctx.strip()]
        
        if not valid_contexts:
            valid_contexts = ["No context retrieved"]
        
        clean_answer = answer.strip()
        if clean_answer.startswith('{') and clean_answer.endswith('}'):
            try:
                import json
                parsed = json.loads(clean_answer)
                if isinstance(parsed, dict):
                    if 'output_message' in parsed:
                        clean_answer = parsed['output_message']
                    elif 'message' in parsed:
                        clean_answer = parsed['message']
                    elif 'answer' in parsed:
                        clean_answer = parsed['answer']
            except:
                pass
        
        data = {
            "question": [question],
            "answer": [clean_answer],
            "contexts": [valid_contexts],
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        else:
            print(f"[WARNING] No ground_truth provided - answer_correctness will be 0")
        
        try:

            if ground_truth:

                if clean_answer.lower() in ground_truth.lower() or ground_truth.lower() in clean_answer.lower():
                    print(f"[DEBUG] ✓ Answer and ground truth have some overlap")
                else:
                    print(f"[DEBUG] ⚠ Answer and ground truth don't overlap - correctness may be low")
            else:
                print(f"[DEBUG] ⚠ No ground_truth provided - answer_correctness will be 0")
            
            dataset = Dataset.from_dict(data)
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
            )
            
            scores = result.to_pandas().iloc[0].to_dict()
            
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        return default
                    return float(value)
                return default
            
            faithfulness_score = safe_float(scores.get('faithfulness'))
            answer_relevancy_score = safe_float(scores.get('answer_relevancy'))
            context_precision_score = safe_float(scores.get('context_precision'))
            context_recall_score = safe_float(scores.get('context_recall'))
            answer_correctness_score = safe_float(scores.get('answer_correctness'))
            
            print(f"[DEBUG] Raw RAGAS scores: {scores}")
            
            if answer_relevancy_score == 0.0 and len(clean_answer) > 10:
                print(f"[WARNING] Answer Relevancy is 0 - answer might not be in natural language format")
                print(f"[WARNING] Consider checking if answer contains JSON or special formatting")
            
            if answer_correctness_score == 0.0 and ground_truth:
                print(f"[WARNING] Answer Correctness is 0 - answer doesn't match ground truth")
                print(f"[WARNING] This is expected if answers differ, but check format compatibility")
            
            overall = (
                faithfulness_score * 0.3 +
                answer_relevancy_score * 0.2 +
                context_precision_score * 0.15 +
                context_recall_score * 0.15 +
                answer_correctness_score * 0.2
            )
            
            return EvaluationResult(
                faithfulness=faithfulness_score,
                answer_relevancy=answer_relevancy_score,
                context_precision=context_precision_score,
                context_recall=context_recall_score,
                answer_correctness=answer_correctness_score,
                overall_score=overall,
                details=scores
            )
            
        except Exception as e:
            print(f"[WARNING] RAGAS evaluation failed: {str(e)}")
            return EvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                answer_correctness=0.0,
                overall_score=0.0,
                details={"error": str(e)}
            )
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        results = []
        
        for case in test_cases:
            result = self.evaluate_single(
                question=case["question"],
                answer=case["answer"],
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth")
            )
            results.append(result)
        
        avg_scores = {
            "faithfulness": sum(r.faithfulness for r in results) / len(results),
            "answer_relevancy": sum(r.answer_relevancy for r in results) / len(results),
            "context_precision": sum(r.context_precision for r in results) / len(results),
            "context_recall": sum(r.context_recall for r in results) / len(results),
            "answer_correctness": sum(r.answer_correctness for r in results) / len(results),
            "overall_score": sum(r.overall_score for r in results) / len(results),
        }
        
        return {
            "average_scores": avg_scores,
            "individual_results": [
                {
                    "question": case["question"],
                    "overall_score": r.overall_score,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "answer_correctness": r.answer_correctness,
                }
                for case, r in zip(test_cases, results)
            ]
        }
    
    def evaluate_from_agent_run(
        self,
        question: str,
        agent_output: str,
        tool_results: Dict[str, List[Dict[str, Any]]],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:

        contexts = []
        
        if 'structured_db_results' in tool_results and tool_results['structured_db_results']:
            for result in tool_results['structured_db_results']:
                if isinstance(result, dict):
                    context = f"Client: {result.get('client_name', 'N/A')}, "
                    context += f"Industry: {result.get('industry', 'N/A')}, "
                    spend = result.get('total_spend_ytd', 0)
                    context += f"Spend: ${spend:,.0f}, " if isinstance(spend, (int, float)) else f"Spend: {spend}, "
                    context += f"Manager: {result.get('account_manager', 'N/A')}"
                    if context.strip() and context != "Client: N/A, Industry: N/A, Spend: 0, Manager: N/A":
                        contexts.append(context)
        
        if 'semantic_db_results' in tool_results and tool_results['semantic_db_results']:
            for result in tool_results['semantic_db_results']:
                if isinstance(result, dict):
                    text = result.get('text', '')
                    if text and text.strip():
                        contexts.append(text.strip())
        
        if not contexts:
            for key, value in tool_results.items():
                print(f"[DEBUG] {key}: {len(value) if isinstance(value, list) else 'not a list'} items")
        else:
            if contexts:
                print(f"[DEBUG] First context: {contexts[0][:100]}...")
        
        return self.evaluate_single(
            question=question,
            answer=agent_output,
            contexts=contexts,
            ground_truth=ground_truth
        )


def load_test_dataset(path: str = "src/evaluation/test_dataset.json") -> List[Dict[str, Any]]:
    """Load test dataset from JSON file"""
    test_path = Path(path)
    if not test_path.exists():
        return []
    
    with open(test_path, 'r') as f:
        return json.load(f)


def save_evaluation_results(results: Dict[str, Any], path: str = "src/evaluation/evaluation_results.json"):
    """Save evaluation results to JSON file"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
