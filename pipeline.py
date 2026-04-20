"""Azure ML Pipeline - RAG evaluation with quality gate."""

import os
import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    return MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)


def build_pipeline(ml_client: MLClient, threshold: float, compute: str, env_vars: dict):
    env = Environment(
        name="ricoh-rag-eval-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file="conda.yml",
    )

    eval_component = command(
        name="rag_evaluation",
        display_name="RAG Evaluation",
        description="Evaluate retrieval and answer quality of the RAG pipeline",
        command="python rag_evaluation.py --threshold ${{inputs.threshold}} --output_dir ${{outputs.results}}",
        code="./eval",
        environment=env,
        inputs={"threshold": Input(type="number", default=threshold)},
        outputs={"results": Output(type="uri_folder")},
        environment_variables=env_vars,
    )

    gate_component = command(
        name="quality_gate",
        display_name="Quality Gate",
        description="Check evaluation results and block/allow deployment",
        command=(
            'python -c "'
            "import json, sys; "
            "r=json.load(open('${{inputs.results}}/results.json')); "
            "print('Quality gate:', 'PASSED' if r['quality_gate_passed'] else 'FAILED'); "
            "sys.exit(0 if r['quality_gate_passed'] else 1)"
            '"'
        ),
        environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
        inputs={"results": Input(type="uri_folder")},
    )

    @pipeline(
        display_name="ricoh-rag-eval-pipeline",
        description="RAG evaluation pipeline with quality gate",
        compute=compute,
    )
    def rag_eval_pipeline(threshold: float = 0.6):
        eval_step = eval_component(threshold=threshold)
        gate_step = gate_component(results=eval_step.outputs.results)
        return {"results": eval_step.outputs.results}

    return rag_eval_pipeline(threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="Submit RAG eval pipeline to Azure ML")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--compute", default="cpu-cluster")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--experiment-name", default="ricoh-rag-evaluation")
    args = parser.parse_args()

    # Pass secrets from runner environment into the pipeline job
    env_vars = {
        "AZURE_AI_ENDPOINT": os.environ["AZURE_AI_ENDPOINT"],
        "AZURE_AI_KEY": os.environ["AZURE_AI_KEY"],
        "AZURE_AI_CHAT_DEPLOYMENT": os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
        "AZURE_OPENAI_ENDPOINT": os.environ["AZURE_OPENAI_ENDPOINT"],
        "AZURE_OPENAI_KEY": os.environ["AZURE_OPENAI_KEY"],
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        "PG_CONNECTION_STRING": os.environ["PG_CONNECTION_STRING"],
    }

    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace)
    pipeline_job = build_pipeline(ml_client, args.threshold, args.compute, env_vars)
    pipeline_job.experiment_name = args.experiment_name

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline submitted: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")

    ml_client.jobs.stream(submitted.name)


if __name__ == "__main__":
    main()
