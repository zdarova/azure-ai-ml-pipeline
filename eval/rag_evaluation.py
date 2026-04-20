"""RAG Evaluation - Azure ML pipeline component."""

import os
import json
import argparse
import mlflow
from langchain_openai import AzureOpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EVAL_DATASET = [
    {
        "question": "Dove ha sede Ricoh Italia?",
        "expected_keywords": ["vimodrone", "milano", "mi"],
        "expected_source": "Ricoh Italia - Chi Siamo",
    },
    {
        "question": "Quali soluzioni AI offre Ricoh?",
        "expected_keywords": ["idp", "document processing", "chatbot", "automazione"],
        "expected_source": "Ricoh - Soluzioni AI e Automazione",
    },
    {
        "question": "Che cloud usa Ricoh?",
        "expected_keywords": ["azure", "microsoft"],
        "expected_source": "Ricoh - Partnership Microsoft e Azure",
    },
    {
        "question": "Cos'è RICOH Spaces?",
        "expected_keywords": ["workplace", "prenotazione", "iot"],
        "expected_source": "Ricoh - Piattaforma RICOH Spaces",
    },
    {
        "question": "Come gestisce Ricoh la sicurezza?",
        "expected_keywords": ["gdpr", "entra", "crittografia"],
        "expected_source": "Ricoh - Sicurezza e Compliance",
    },
]


def pg_conn_to_sqlalchemy(pg_str: str) -> str:
    parts = dict(p.split("=", 1) for p in pg_str.split() if "=" in p)
    return f"postgresql+psycopg://{parts['user']}:{parts['password']}@{parts['host']}:{parts['port']}/{parts['dbname']}?sslmode={parts['sslmode']}"


def evaluate_retrieval(vectorstore, dataset):
    hits = 0
    for item in dataset:
        docs = vectorstore.similarity_search(item["question"], k=4)
        titles = [d.metadata.get("title", "") for d in docs]
        if item["expected_source"] in titles:
            hits += 1
    return hits / len(dataset)


def evaluate_answer_quality(chain, dataset):
    scores = []
    for item in dataset:
        response = chain.invoke(item["question"]).lower()
        matched = sum(1 for kw in item["expected_keywords"] if kw in response)
        scores.append(matched / len(item["expected_keywords"]))
    return sum(scores) / len(scores)


def evaluate_answer_relevance(llm, dataset, chain):
    judge_prompt = ChatPromptTemplate.from_template(
        "Rate the relevance of this answer to the question on a scale 0-10.\n"
        "Question: {question}\nAnswer: {answer}\n"
        "Reply with ONLY a number 0-10."
    )
    scores = []
    for item in dataset:
        answer = chain.invoke(item["question"])
        result = (judge_prompt | llm).invoke({"question": item["question"], "answer": answer})
        try:
            score = int(result.content.strip()) / 10.0
            scores.append(min(max(score, 0), 1))
        except ValueError:
            scores.append(0.5)
    return sum(scores) / len(scores)


def build_chain(vectorstore, llm):
    prompt = ChatPromptTemplate.from_template(
        "Contesto: {context}\n\nDomanda: {question}\nRispondi in italiano."
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 4}) | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def run_evaluation(threshold: float, output_dir: str):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version="2024-06-01",
    )
    conn_str = pg_conn_to_sqlalchemy(os.environ["PG_CONNECTION_STRING"])
    vectorstore = PGVector(connection=conn_str, embeddings=embeddings, collection_name="ricoh_knowledge")

    llm = ChatAnthropic(
        model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
        api_key=os.environ["AZURE_AI_KEY"],
        base_url=os.environ["AZURE_AI_ENDPOINT"],
        temperature=0.3,
        max_tokens=1024,
    )
    chain = build_chain(vectorstore, llm)

    retrieval_accuracy = evaluate_retrieval(vectorstore, EVAL_DATASET)
    answer_quality = evaluate_answer_quality(chain, EVAL_DATASET)
    answer_relevance = evaluate_answer_relevance(llm, EVAL_DATASET, chain)

    passed = retrieval_accuracy >= threshold and answer_quality >= threshold

    # Log to Azure ML workspace MLflow
    mlflow.log_metric("retrieval_accuracy", retrieval_accuracy)
    mlflow.log_metric("answer_keyword_coverage", answer_quality)
    mlflow.log_metric("answer_relevance_llm_judge", answer_relevance)
    mlflow.log_metric("quality_gate_passed", int(passed))
    mlflow.log_param("model", os.environ["AZURE_AI_CHAT_DEPLOYMENT"])
    mlflow.log_param("embedding", os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
    mlflow.log_param("retriever_k", 4)
    mlflow.log_param("eval_dataset_size", len(EVAL_DATASET))
    mlflow.log_param("threshold", threshold)

    # Write results to output dir for downstream steps
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "retrieval_accuracy": retrieval_accuracy,
        "answer_keyword_coverage": answer_quality,
        "answer_relevance_llm_judge": answer_relevance,
        "quality_gate_passed": passed,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Retrieval Accuracy:      {retrieval_accuracy:.2%}")
    print(f"Answer Keyword Coverage: {answer_quality:.2%}")
    print(f"Answer Relevance (LLM):  {answer_relevance:.2%}")
    print(f"\nQuality Gate ({'✓ PASSED' if passed else '✗ FAILED'}) - threshold: {threshold:.0%}")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    passed = run_evaluation(args.threshold, args.output_dir)
    exit(0 if passed else 1)
