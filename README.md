# CRM Sales Co-Pilot MVP

An intelligent CRM assistant built with LangGraph that combines structured database queries, semantic search, and self-reflection capabilities to provide accurate sales recommendations and policy information.

## 🏗️ Project Structure

```
CaseStudy/
├── src/
│   ├── agents/              # LangGraph agents and workflow
│   │   ├── crm_agent.py     # Main CRM agent with tool calling
│   │   ├── reflection_agent.py  # QA agent for self-correction
│   │   ├── main_graph.py    # LangGraph workflow orchestration
│   │   ├── state.py         # State management with TypedDict
│   │   ├── prompts/         # Agent prompts
│   │   │   ├── crm_prompts.py
│   │   │   └── reflection_prompts.py
│   │   └── tools/           # Tool implementations
│   │       ├── structured_db_search_tool.py
│   │       ├── semantic_search_tool.py
│   │       └── cache.py      # Caching layer for latency optimization
│   │
│   ├── database/            # Data layer
│   │   ├── structured_db.py # SQLite client database
│   │   ├── semantic_db.py   # Faiss vector store for policies
│   │   ├── embedding.py     # Embedding pipeline
│   │   └── async_db.py      # Async database wrappers
│   │
│   ├── evaluation/          # RAG evaluation framework
│   │   ├── ragas_evaluator.py    # RAGAS metrics implementation
│   │   ├── evaluate_rag.py       # Evaluation script
│   │   └── test_dataset.json     # Test cases
│   │
│   ├── tasks/               # RLHF training
│   │   ├── rlhf.py          # PPO fine-tuning script
│   │   └── rlhf.json        # Training examples
│   │
│   ├── models/              # LLM model wrappers
│   │   └── chat_models.py   # OpenAI, Anthropic, TinyLlama support
│   │
│   ├── app.py               # Streamlit web interface
│   └── main.py              # Application entry point
│
├── faiss_store/             # Persistent vector store
├── structured_db.sqlite     # SQLite database
├── requirements.txt          # Main dependencies
├── requirements_eval.txt     # Evaluation dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (for CRM agent)
- Optional: CUDA-capable GPU (for local models)

### Quick Commands Reference

```bash
# 1. Setup (first time only)
pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt
pip install -r requirements_eval.txt

# 2. Set API key
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
# OR
$env:OPENAI_API_KEY="your-key-here"    # Windows PowerShell

# 3. Run application
python -m src.main

# 4. Run RAGAS evaluation
python -m src.evaluation.evaluate_rag --model gpt-4o-mini

# 5. Run RLHF training
python -m src.tasks.rlhf --epochs 3 --lr 1e-5
```

### Environment Setup

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL_NAME=gpt-4o-mini
```

Or set environment variables:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"

# Windows CMD
set OPENAI_API_KEY=your-key-here

# Linux/Mac
export OPENAI_API_KEY="your-key-here"
```

### 3. Initialize Databases

The databases are automatically initialized on first run, but you can manually initialize:

```python
from src.database.semantic_db import SemanticDatabase
from src.database.structured_db import StructuredDatabase

# Initialize semantic database (vector store)
SemanticDatabase.initialize_and_populate()

# Initialize structured database (SQLite)
StructuredDatabase.initialize_and_populate()
```

### 4. Run the Application

```bash
# Option 1: Using main.py (recommended)
python -m src.main

# Option 2: Direct Streamlit
streamlit run src/app.py
```

The app will be available at `http://localhost:8501`

## 📊 Components Overview

### Agents (`src/agents/`)

#### CRM Agent (`crm_agent.py`)
- **Purpose**: Main agent that answers sales-related queries
- **Capabilities**:
  - Tool selection (structured DB vs semantic DB)
  - JSON-formatted responses
  - Handles client data, policies, and recommendations
- **Tools**:
  - `search_structured_db(client_name)`: Query client facts
  - `search_semantic_db(query)`: Search sales policies/rules

#### Reflection Agent (`reflection_agent.py`)
- **Purpose**: QA agent that evaluates CRM agent responses
- **Capabilities**:
  - Evaluates tool usage correctness
  - Provides feedback for improvement
  - Supports up to 3 reflection iterations
- **Model**: Uses TinyLlama (local) or GPT-4o-mini (configurable)

#### Main Graph (`main_graph.py`)
- **Purpose**: Orchestrates the agent workflow
- **Flow**:
  ```
  START → CRM Agent → Tools (if needed) → Reflection Agent → END/Retry
  ```
- **Features**:
  - Parallel tool execution support
  - Streaming capabilities
  - Async/await support

### Database Layer (`src/database/`)

#### Structured Database (`structured_db.py`)
- **Type**: SQLite relational database
- **Data**: Client information (spend, industry, account manager)
- **Operations**: Search by client name with fuzzy matching

#### Semantic Database (`semantic_db.py`)
- **Type**: Faiss vector store
- **Data**: Sales playbook policies and rules
- **Operations**: Semantic similarity search using embeddings
- **Model**: `all-MiniLM-L6-v2` (default) or OpenAI embeddings

### Evaluation (`src/evaluation/`)

#### RAGAS Evaluator (`ragas_evaluator.py`)
- **Metrics**:
  - **Faithfulness**: Answer grounded in context?
  - **Answer Relevancy**: Answer relevant to question?
  - **Context Precision**: Retrieved contexts relevant?
  - **Context Recall**: All relevant contexts retrieved?
  - **Answer Correctness**: Factually correct vs ground truth?

#### Evaluation Script (`evaluate_rag.py`)
- Runs agent on test dataset
- Extracts tool results from execution
- Computes RAGAS metrics
- Generates quality reports

### RLHF Training (`src/tasks/`)

#### PPO Fine-tuning (`rlhf.py`)
- **Purpose**: Fine-tune reflection agent using Proximal Policy Optimization
- **Training Data**: `rlhf.json` (20 examples)
- **Model**: TinyLlama (default, RLHF-trainable)
- **Framework**: TRL (Transformer Reinforcement Learning)

## 🧪 Running Evaluations

### RAGAS Evaluation

Evaluate your RAG system using the RAGAS framework:

```bash
# Basic evaluation (uses default test dataset)
python -m src.evaluation.evaluate_rag --model gpt-4o-mini

# Custom dataset path
python -m src.evaluation.evaluate_rag \
    --model gpt-4o-mini \
    --dataset src/evaluation/test_dataset.json

# Results are saved to: src/evaluation/evaluation_results.json
```

**What it does**:
1. Loads test cases from `src/evaluation/test_dataset.json`
2. Runs each query through the CRM agent
3. Extracts tool results and contexts
4. Evaluates with RAGAS metrics (Faithfulness, Relevancy, Correctness, etc.)
5. Generates a quality report

**Output Example**:
```
Average Scores:
  Overall Score: 0.750
  Faithfulness: 1.000
  Answer Relevancy: 0.850
  Context Precision: 0.900
  Answer Correctness: 0.800

Quality Assessment:
  ✓ EXCELLENT - System is performing well
```

**Output includes**:
- Average scores across all metrics
- Individual test case results
- Quality assessment (Excellent/Good/Fair/Poor)

### Test Dataset Format

```json
{
  "question": "How much has TechGiant spent?",
  "ground_truth": "TechGiant has spent $120,000 YTD.",
  "expected_tools": ["search_structured_db"],
  "expected_contexts": ["TechGiant, Electronics, 120000"]
}
```

## 🎓 RLHF Training

Fine-tune the reflection agent using Proximal Policy Optimization (PPO).

### Prepare Training Data

The training data is in `src/tasks/rlhf.json` (20 examples included). Format:

```json
{
  "user_input": "What is our policy on pricing?",
  "crm_output": "Our policy states...",
  "tool_calls_made": ["search_semantic_db"],
  "reflection_output": "CRM Agent correctly used search_semantic_db.",
  "correctness": true
}
```

### Install RLHF Dependencies

```bash
pip install trl peft accelerate datasets bitsandbytes
```

### Run Training

```bash
# Basic training (3 epochs, default learning rate)
python -m src.tasks.rlhf

# Custom configuration
python -m src.tasks.rlhf \
    --epochs 5 \
    --lr 1e-5 \
    --output models/reflection_agent_ppo \
    --data src/tasks/rlhf.json

# Arguments:
#   --epochs: Number of training epochs (default: 3)
#   --lr: Learning rate (default: 1e-5)
#   --output: Output directory for fine-tuned model (default: models/reflection_agent_ppo)
#   --data: Path to training data JSON (default: src/tasks/rlhf.json)
```

**Training Process**:
1. Loads TinyLlama model (1.1B parameters, ~2GB)
2. Creates reward model based on training examples
3. Runs PPO training loop
4. Saves fine-tuned model to output directory

**Note**: 
- Training requires GPU for reasonable speed
- Uses 4-bit quantization automatically on GPU
- CPU training is possible but very slow
- Fine-tuned model can be loaded in `reflection_agent.py`

## ⚡ Performance Optimizations

### 1. Caching (`src/agents/tools/cache.py`)
- **TTL-based caching** for tool results
- Structured DB: 5 minutes
- Semantic DB: 10 minutes
- Thread-safe implementation
- Cache stats available in Streamlit sidebar

### 2. Parallel Tool Execution
- Tools can run concurrently using `asyncio.gather()`
- Async versions: `search_structured_db_async()`, `search_semantic_db_async()`

### 3. Streaming (`src/app.py`)
- Real-time response streaming
- Progress indicators for each graph node
- Better perceived latency

### 4. Async Database Operations (`src/database/async_db.py`)
- Async SQLite with `aiosqlite`
- Thread pool for Faiss operations
- Non-blocking I/O

## 🔧 Configuration

### Model Selection

```python
# In src/app.py or src/main.py
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")  # CRM agent
reflection_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Reflection agent
```

### Cache Settings

```python
# In src/agents/tools/cache.py
structured_db_cache = ToolCache(ttl_seconds=300)  # 5 min
semantic_db_cache = ToolCache(ttl_seconds=600)    # 10 min
```

## 📝 Requirements Files

### Main Requirements

The project uses `pip-compile` for dependency management:

1. **Edit `requirements.in`** to add/update dependencies
2. **Compile** to generate `requirements.txt`:
   ```bash
   pip-compile requirements.in -o requirements.txt
   ```
3. **Install**:
   ```bash
   pip install -r requirements.txt
   ```

**Note**: `requirements.in` is the source file. Always edit this file, then recompile.

### Evaluation Requirements (`requirements_eval.txt`)

```bash
pip install -r requirements_eval.txt
```

## 🐛 Troubleshooting

### API Key Issues
```bash
# Check if API key is set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY   # Windows PowerShell
```

### Database Initialization Errors
- Ensure `src/snippets.pdf` exists for semantic database
- Check `src/mock_sql_data.json` for structured database
- Verify write permissions for `faiss_store/` and `structured_db.sqlite`

### RAGAS Evaluation Returns 0
- Check that contexts are being extracted from tool results
- Verify ground_truth format matches answer format
- Ensure answers are in natural language (not JSON)

### RLHF Training Fails
- Install `bitsandbytes` for quantization
- Use smaller model if GPU memory insufficient
- Check training data format in `rlhf.json`

## 📚 Key Features

- ✅ **Multi-tool Agent**: Structured + Semantic search
- ✅ **Self-Reflection**: QA agent with up to 3 retry iterations
- ✅ **Caching**: TTL-based caching for reduced latency
- ✅ **Streaming**: Real-time response streaming
- ✅ **Evaluation**: RAGAS framework integration
- ✅ **RLHF Ready**: PPO fine-tuning support
- ✅ **Async Support**: Non-blocking database operations

## 🔄 System Architecture

### Agent Workflow

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   CRM Agent     │ ◄──┐
│  (Tool Calling) │    │
└──────┬──────────┘    │
       │               │
       ├─► Tool Calls  │
       │   (Parallel)   │
       │               │
       ▼               │
┌─────────────────┐   │
│  Tool Execution │   │
│  (Structured +  │   │
│   Semantic DB)  │   │
└──────┬──────────┘   │
       │               │
       └───────────────┘
       │
       ▼
┌─────────────────┐
│ Reflection Agent│
│  (QA Evaluation)│
└──────┬──────────┘
       │
       ├─► Correct? ──► END
       │
       └─► Incorrect? ──► Retry (max 3x)
```

### Workflow Example

1. **User Query**: "What should I pitch to TechGiant?"
2. **CRM Agent**: 
   - Calls `search_structured_db("TechGiant")` → Gets client data (Electronics, $120k)
   - Calls `search_semantic_db("product recommendation Electronics 120k")` → Gets policies
   - Generates recommendation: "Mobile App Banners and Online Display Ads"
3. **Reflection Agent**: 
   - Evaluates: "Used both tools correctly ✓"
   - Sets `correctness=True` → END
4. **Final Answer**: Returns to user

### Retry Scenario

If reflection agent finds issues:
1. Sets `correctness=False`
2. Provides feedback: "Should have used both tools"
3. CRM agent retries with feedback
4. Up to 3 iterations

## 📁 Data Files

### Required Files

- **`src/snippets.pdf`**: Sales playbook PDF (automatically indexed into Faiss)
- **`src/mock_sql_data.json`**: Client data for SQLite database
- **`.env`**: Environment variables (create from `.env.example`)

### Generated Files

- **`faiss_store/`**: Vector store index (auto-generated)
- **`structured_db.sqlite`**: SQLite database (auto-generated)
- **`models/reflection_agent_ppo/`**: Fine-tuned model (after RLHF training)

## 🔍 Example Queries

### Simple Retrieval
```
"How much has TechGiant spent so far?"
→ Uses: search_structured_db
→ Returns: Client spend data
```

### Policy Retrieval
```
"What is our policy on discussing 2026 pricing?"
→ Uses: search_semantic_db
→ Returns: Policy from playbook
```

### Complex Reasoning
```
"I have a meeting with TechGiant. Based on their spending and industry, what product should I pitch?"
→ Uses: search_structured_db + search_semantic_db
→ Returns: Recommendation based on client data + policies
```

## 🛠️ Development

### Running Tests

```bash
# Run RAGAS evaluation (acts as integration test)
python -m src.evaluation.evaluate_rag

# Check cache performance
# (Available in Streamlit sidebar)
```

### Debugging

Enable debug logging by setting environment variable:
```bash
export DEBUG=1
```

Or modify agent files to add print statements (already included in key locations).

## 📊 Performance Metrics

### Cache Statistics

View in Streamlit sidebar:
- Hit rate
- Cache size
- Hits vs misses

### Evaluation Metrics

RAGAS provides:
- **Faithfulness**: 0-1 (higher = more grounded)
- **Answer Relevancy**: 0-1 (higher = more relevant)
- **Context Precision**: 0-1 (higher = better retrieval)
- **Answer Correctness**: 0-1 (higher = more accurate)

## 🐛 Common Issues

### "No message found in input" Error
- **Cause**: Message history not properly maintained
- **Fix**: Ensure `add_messages` reducer is used in state

### Tool calls not working
- **Cause**: API key not set or model doesn't support tool calling
- **Fix**: Set `OPENAI_API_KEY` and use `gpt-4o-mini` or newer

### Reflection agent returns 0 scores
- **Cause**: TinyLlama output format issues
- **Fix**: Parser handles this, but check if model loaded correctly

### RAGAS evaluation fails
- **Cause**: Missing contexts or malformed data
- **Fix**: Check tool results extraction in `evaluate_rag.py`

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [TRL (RLHF) Documentation](https://huggingface.co/docs/trl/)
