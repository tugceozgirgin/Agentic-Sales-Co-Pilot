"""
RAG Evaluation Script using RAGAS framework.
Runs evaluation on test dataset and generates quality reports.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try root directory
        load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("=" * 60)
    print("ERROR: OPENAI_API_KEY not found!")
    print("=" * 60)
    print("\nPlease set your OpenAI API key using one of these methods:\n")
    print("Method 1: Environment Variable")
    print("  Windows PowerShell:")
    print("    $env:OPENAI_API_KEY='your-key-here'")
    print("  Windows CMD:")
    print("    set OPENAI_API_KEY=your-key-here")
    print("  Linux/Mac:")
    print("    export OPENAI_API_KEY='your-key-here'")
    print("\nMethod 2: .env file (recommended)")
    print("  Create a .env file in the project root with:")
    print("    OPENAI_API_KEY=your-key-here")
    print("  Then install: pip install python-dotenv")
    print("\nMethod 3: Set in your IDE/terminal before running")
    print("=" * 60)
    sys.exit(1)

from src.evaluation.ragas_evaluator import (
    RAGASEvaluator,
    load_test_dataset,
    save_evaluation_results
)
from src.agents.main_graph import CRMGraph
from src.agents.tools import search_structured_db, search_semantic_db


class RAGEvaluator:
    """Main evaluator that runs agent and evaluates responses"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.ragas_evaluator = RAGASEvaluator()
        self.graph = CRMGraph(model_name=model_name)
    
    def run_evaluation(self, test_dataset_path: str = "src/evaluation/test_dataset.json"):
        print("=" * 60)
        print("RAG Evaluation using RAGAS Framework")
        print("=" * 60)
        
        test_cases = load_test_dataset(test_dataset_path)
        if not test_cases:
            print(f"[ERROR] No test cases found at {test_dataset_path}")
            return
        
        print(f"\nLoaded {len(test_cases)} test cases\n")
        
        evaluation_data = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['question']}")
            
            try:
                from langchain_core.messages import HumanMessage, ToolMessage
                from src.agents.state import MainState
                
                initial_state = MainState(
                    input=test_case['question'],
                    output="",
                    stop=False,
                    reflection_count=0,
                    last_tool_calls=None,
                    reflection_output="",
                    messages=[HumanMessage(content=test_case['question'])],
                    crm_agent_output=None
                )
                
                graph = self.graph.build_graph()
                final_state = graph.invoke(initial_state)
                
                raw_answer = final_state.get("output", "")
                answer = raw_answer
                
                crm_output = final_state.get("crm_agent_output")
                if crm_output and hasattr(crm_output, 'output_message'):
                    answer = crm_output.output_message
                else:
                    try:
                        import json
                        if raw_answer.strip().startswith('{'):
                            parsed = json.loads(raw_answer)
                            if isinstance(parsed, dict) and 'output_message' in parsed:
                                answer = parsed['output_message']
                    except:
                        pass

                if answer:
                    answer = answer.strip()
                    if answer.startswith('{') and answer.endswith('}'):
                        try:
                            import json
                            parsed = json.loads(answer)
                            if isinstance(parsed, dict) and 'output_message' in parsed:
                                answer = parsed['output_message']
                        except:
                            pass
                
                # Extract tool results from messages
                tool_results = self._extract_tool_results_from_messages(final_state.get("messages", []))
                
              
                result = self.ragas_evaluator.evaluate_from_agent_run(
                    question=test_case['question'],
                    agent_output=answer,
                    tool_results=tool_results,
                    ground_truth=test_case.get('ground_truth')
                )
                
                evaluation_data.append({
                    "question": test_case['question'],
                    "answer": answer,
                    "ground_truth": test_case.get('ground_truth'),
                    "contexts": self._extract_contexts_from_tools(tool_results),
                    "overall_score": result.overall_score,
                    "faithfulness": result.faithfulness,
                    "answer_relevancy": result.answer_relevancy,
                    "context_precision": result.context_precision,
                    "context_recall": result.context_recall,
                    "answer_correctness": result.answer_correctness,
                })
                
                print(f"  ✓ Overall Score: {result.overall_score:.3f}")
                print(f"    - Faithfulness: {result.faithfulness:.3f}")
                print(f"    - Answer Relevancy: {result.answer_relevancy:.3f}")
                print(f"    - Context Precision: {result.context_precision:.3f}")
                print(f"    - Answer Correctness: {result.answer_correctness:.3f}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                evaluation_data.append({
                    "question": test_case['question'],
                    "error": str(e),
                    "overall_score": 0.0
                })
        
        self._generate_report(evaluation_data)
        
        save_evaluation_results({
            "summary": self._calculate_summary(evaluation_data),
            "detailed_results": evaluation_data
        })
    
    def _extract_client_name(self, question: str) -> str:
        clients = ["TechGiant", "MegaBev", "GlobalRetail", "MegaCorp"]
        for client in clients:
            if client.lower() in question.lower():
                return client
        return ""
    
    def _extract_tool_results_from_messages(self, messages) -> Dict[str, List]:
        """Extract tool results from LangGraph messages"""
        from langchain_core.messages import ToolMessage, AIMessage
        import json
        
        tool_results = {
            'structured_db_results': [],
            'semantic_db_results': []
        }
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    content = msg.content
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                # Check which tool this came from
                                tool_name = getattr(msg, 'name', '')
                                if 'structured' in tool_name.lower() or 'structured_db' in tool_name.lower():
                                    tool_results['structured_db_results'].extend(parsed)
                                elif 'semantic' in tool_name.lower() or 'semantic_db' in tool_name.lower():
                                    tool_results['semantic_db_results'].extend(parsed)
                        except json.JSONDecodeError:
                            if content.startswith('['):
                                try:
                                    parsed = eval(content)
                                    if isinstance(parsed, list):
                                        tool_name = getattr(msg, 'name', '')
                                        if 'structured' in tool_name.lower():
                                            tool_results['structured_db_results'].extend(parsed)
                                        elif 'semantic' in tool_name.lower():
                                            tool_results['semantic_db_results'].extend(parsed)
                                except:
                                    pass
                except Exception as e:
                    print(f"  [WARNING] Failed to parse tool message: {e}")
                    continue
        
        if not tool_results['structured_db_results'] and not tool_results['semantic_db_results']:
            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', '')
                        tool_args = tc.get('args', {})
                        
                        if 'structured_db' in tool_name:
                            client_name = tool_args.get('client_name', '')
                            if client_name:
                                try:
                                    tool_results['structured_db_results'] = search_structured_db(client_name)
                                except:
                                    pass
                        elif 'semantic_db' in tool_name:
                            query = tool_args.get('query', '')
                            if query:
                                try:
                                    tool_results['semantic_db_results'] = search_semantic_db(query)
                                except:
                                    pass
        
        return tool_results
    
    def _extract_contexts_from_tools(self, tool_results: Dict[str, List]) -> List[str]:
        """Extract context strings from tool results"""
        contexts = []
        
        if 'structured_db_results' in tool_results:
            for result in tool_results['structured_db_results']:
                ctx = f"Client: {result.get('client_name', '')}, "
                ctx += f"Industry: {result.get('industry', '')}, "
                ctx += f"Spend: ${result.get('total_spend_ytd', 0):,.0f}, "
                ctx += f"Manager: {result.get('account_manager', '')}"
                contexts.append(ctx)
        
        if 'semantic_db_results' in tool_results:
            for result in tool_results['semantic_db_results']:
                contexts.append(result.get('text', ''))
        
        return contexts
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        valid_results = [r for r in results if 'error' not in r]
        
        summary = {
            "total_tests": len(results),
            "successful_tests": len(valid_results),
            "failed_tests": len(results) - len(valid_results),
        }
        
        if not valid_results:
            summary["error"] = "No valid results - all tests failed"
            summary["average_overall_score"] = 0.0
            summary["average_faithfulness"] = 0.0
            summary["average_answer_relevancy"] = 0.0
            summary["average_context_precision"] = 0.0
            summary["average_answer_correctness"] = 0.0
            return summary
        
        summary.update({
            "average_overall_score": sum(r.get('overall_score', 0) for r in valid_results) / len(valid_results),
            "average_faithfulness": sum(r.get('faithfulness', 0) for r in valid_results) / len(valid_results),
            "average_answer_relevancy": sum(r.get('answer_relevancy', 0) for r in valid_results) / len(valid_results),
            "average_context_precision": sum(r.get('context_precision', 0) for r in valid_results) / len(valid_results),
            "average_answer_correctness": sum(r.get('answer_correctness', 0) for r in valid_results) / len(valid_results),
        })
        
        return summary
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Print evaluation report"""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        summary = self._calculate_summary(results)
        
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        
        if 'error' in summary:
            print(f"\n⚠ {summary['error']}")
            print("\nAll tests failed. Please check:")
            print("  1. OPENAI_API_KEY environment variable is set")
            print("  2. Database files are initialized")
            print("  3. All dependencies are installed")
        else:
            print("\nAverage Scores:")
            print(f"  Overall Score: {summary['average_overall_score']:.3f}")
            print(f"  Faithfulness: {summary['average_faithfulness']:.3f}")
            print(f"  Answer Relevancy: {summary['average_answer_relevancy']:.3f}")
            print(f"  Context Precision: {summary['average_context_precision']:.3f}")
            print(f"  Answer Correctness: {summary['average_answer_correctness']:.3f}")
            
            print("\nQuality Assessment:")
            overall = summary['average_overall_score']
            if overall >= 0.8:
                print("  ✓ EXCELLENT - System is performing well")
            elif overall >= 0.6:
                print("  ⚠ GOOD - Some improvements needed")
            elif overall >= 0.4:
                print("  ⚠ FAIR - Significant improvements needed")
            else:
                print("  ✗ POOR - Major issues detected")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system with RAGAS")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--dataset", type=str, default="src/evaluation/test_dataset.json", help="Test dataset path")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(model_name=args.model)
    evaluator.run_evaluation(test_dataset_path=args.dataset)
