# Ricoh AI - ML Pipeline (MLOps)

RAG evaluation pipeline with quality gates for the Ricoh AI Knowledge Agent.

## What it does

```
Eval Dataset → Retrieval Test → Answer Quality Test → LLM Judge → MLflow Metrics → Quality Gate
                                                                                        │
                                                                              ✓ PASS → Deploy allowed
                                                                              ✗ FAIL → Deploy blocked
```

## Metrics Tracked

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `retrieval_accuracy` | % of queries returning the correct source document | ≥ 60% |
| `answer_keyword_coverage` | % of expected keywords present in answers | ≥ 60% |
| `answer_relevance_llm_judge` | LLM-as-judge relevance score (0-1) | informational |

## Run Locally

```bash
export AZURE_AI_ENDPOINT="https://ai-ricoh-xxx.services.ai.azure.com/anthropic"
export AZURE_AI_KEY="<key>"
export AZURE_AI_CHAT_DEPLOYMENT="claude-sonnet-4-6"
export AZURE_OPENAI_ENDPOINT="https://oai-ricoh-xxx.openai.azure.com/"
export AZURE_OPENAI_KEY="<key>"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export PG_CONNECTION_STRING="host=... port=5432 dbname=ricoh_kb user=pgadmin password=... sslmode=require"

pip install -r requirements.txt
python eval/rag_evaluation.py
```

## MLflow

Metrics are logged to local MLflow. To view:

```bash
mlflow ui --port 5000
```

## Integration with other repos

- **ricoh-ai-data** → triggers eval when KB data changes
- **ricoh-ai-agent** → eval gates deployment of new agent versions
- **azure-bicep-mlops** → provides infrastructure
