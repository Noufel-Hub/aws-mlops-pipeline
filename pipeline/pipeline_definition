"""
pipeline/pipeline_definition.py
---------------------------------
Assembles and upserts the full SageMaker MLOps Pipeline.

Pipeline DAG:
    preprocessing → training → evaluation → [condition]
                                                ├─ approved → register model → deploy endpoint
                                                └─ rejected → fail step (halt)

Run this script locally to create or update the pipeline in your AWS account:
    python pipeline_definition.py \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        --bucket your-s3-bucket \
        --region eu-west-1
"""

import argparse
import json
import logging

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRAMEWORK_VERSION = "1.2-1"
PYTHON_VERSION    = "py3"
INSTANCE_TYPE     = "ml.m5.xlarge"


# ── Pipeline Parameters (overridable at execution time) ──────────────────────

def get_pipeline_parameters():
    return {
        "input_data_uri": ParameterString(
            name="InputDataUri",
            default_value="s3://your-bucket/raw-data/",
        ),
        "approval_threshold": ParameterFloat(
            name="ApprovalThreshold",
            default_value=0.80,
        ),
        "n_estimators": ParameterInteger(
            name="NEstimators",
            default_value=100,
        ),
        "max_depth": ParameterInteger(
            name="MaxDepth",
            default_value=10,
        ),
    }


# ── Step 1: Preprocessing ────────────────────────────────────────────────────

def get_preprocessing_step(params, role, session, bucket):
    processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step = ProcessingStep(
        name="Preprocessing",
        processor=processor,
        code="pipeline/preprocessing.py",
        inputs=[
            ProcessingInput(
                source=params["input_data_uri"],
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/pipeline/preprocessing/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{bucket}/pipeline/preprocessing/validation",
            ),
        ],
    )
    logger.info("Preprocessing step defined.")
    return step


# ── Step 2: Training ─────────────────────────────────────────────────────────

def get_training_step(params, preprocessing_step, role, session, bucket):
    estimator = SKLearn(
        entry_point="pipeline/training.py",
        framework_version=FRAMEWORK_VERSION,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        output_path=f"s3://{bucket}/pipeline/model-artefacts",
        hyperparameters={
            "n-estimators": params["n_estimators"],
            "max-depth": params["max_depth"],
        },
    )

    step = TrainingStep(
        name="Training",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    logger.info("Training step defined.")
    return step


# ── Step 3: Evaluation ───────────────────────────────────────────────────────

def get_evaluation_step(training_step, preprocessing_step, role, session):
    processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_report.json",
    )

    step = ProcessingStep(
        name="Evaluation",
        processor=processor,
        code="pipeline/evaluation.py",
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        property_files=[evaluation_report],
    )
    logger.info("Evaluation step defined.")
    return step, evaluation_report


# ── Step 4: Condition — Approve or Halt ──────────────────────────────────────

def get_condition_step(params, evaluation_step, evaluation_report, training_step, role, session, bucket):
    # Register model step (runs if approved)
    model = sagemaker.sklearn.model.SKLearnModel(
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="pipeline/training.py",
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=session,
    )
    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=[INSTANCE_TYPE],
            transform_instances=[INSTANCE_TYPE],
            model_package_group_name="aws-mlops-pipeline-models",
            approval_status="PendingManualApproval",
        ),
    )

    # Fail step (runs if rejected)
    fail_step = FailStep(
        name="ModelRejected",
        error_message="Model did not meet the minimum F1 threshold. Pipeline halted.",
    )

    condition_step = ConditionStep(
        name="ApprovalGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=sagemaker.workflow.functions.JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="metrics.f1_weighted",
                ),
                right=params["approval_threshold"],
            )
        ],
        if_steps=[register_step],
        else_steps=[fail_step],
    )
    logger.info("Condition (approval gate) step defined.")
    return condition_step


# ── Pipeline Assembly ────────────────────────────────────────────────────────

def build_pipeline(role: str, bucket: str, region: str) -> Pipeline:
    boto3.setup_default_session(region_name=region)
    session = sagemaker.Session()

    params = get_pipeline_parameters()

    preprocessing_step = get_preprocessing_step(params, role, session, bucket)
    training_step = get_training_step(params, preprocessing_step, role, session, bucket)
    evaluation_step, evaluation_report = get_evaluation_step(training_step, preprocessing_step, role, session)
    condition_step = get_condition_step(params, evaluation_step, evaluation_report, training_step, role, session, bucket)

    pipeline = Pipeline(
        name="aws-mlops-pipeline",
        parameters=list(params.values()),
        steps=[preprocessing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=session,
    )
    logger.info("Pipeline assembled: %d steps.", 4)
    return pipeline


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the SageMaker MLOps Pipeline")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", required=True, help="S3 bucket name for artefacts")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args.role_arn, args.bucket, args.region)

    logger.info("Upserting pipeline in AWS...")
    pipeline.upsert(role_arn=args.role_arn)
    logger.info("✅ Pipeline created/updated successfully.")
    logger.info("Start a run with: pipeline.start()")


if __name__ == "__main__":
    main()
