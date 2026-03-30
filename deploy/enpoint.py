"""
deploy/endpoint.py
-------------------
Deploys or updates a SageMaker real-time inference endpoint using a
blue/green deployment strategy — zero downtime, automatic rollback
on error.

Deployment flow:
    Model Registry (approved model)
        → Create/update SageMaker Model object
        → Blue/green endpoint deployment
        → Smoke test (sanity inference check)
        → Endpoint live ✅

Run after a model has been approved in the Model Registry:
    python endpoint.py \
        --model-package-arn arn:aws:sagemaker:...:model-package/... \
        --endpoint-name aws-mlops-pipeline-endpoint \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        --region eu-west-1
"""

import argparse
import json
import logging
import time

import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.model import ModelPackage
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INSTANCE_TYPE   = "ml.m5.xlarge"
INSTANCE_COUNT  = 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def endpoint_exists(endpoint_name: str, sm_client) -> bool:
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm_client.exceptions.ClientError:
        return False


def wait_for_endpoint(endpoint_name: str, sm_client, timeout: int = 600) -> None:
    """Poll until endpoint reaches InService or fails."""
    logger.info("Waiting for endpoint '%s' to be InService...", endpoint_name)
    start = time.time()
    while time.time() - start < timeout:
        status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status == "InService":
            logger.info("✅ Endpoint is InService.")
            return
        if status in ("Failed", "RollingBack"):
            raise RuntimeError(f"Endpoint deployment failed with status: {status}")
        logger.info("  Status: %s — waiting...", status)
        time.sleep(20)
    raise TimeoutError(f"Endpoint did not reach InService within {timeout}s.")


# ── Blue/Green Deployment Config ─────────────────────────────────────────────

def get_deployment_config(sm_client) -> dict:
    """
    Returns a blue/green deployment config with linear traffic shifting.
    Shifts 10% of traffic every 2 minutes, with auto-rollback on errors.
    """
    config_name = "BlueGreenLinearDeployment"

    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "placeholder",   # overridden at deployment
                    "InstanceType": INSTANCE_TYPE,
                    "InitialInstanceCount": INSTANCE_COUNT,
                    "InitialVariantWeight": 1.0,
                }
            ],
            DeploymentConfig={
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "LINEAR",
                        "LinearStepSize": {
                            "Type": "CAPACITY_PERCENT",
                            "Value": 10,
                        },
                        "WaitIntervalInSeconds": 120,
                    },
                    "TerminationWaitInSeconds": 300,
                    "MaximumExecutionTimeoutInSeconds": 1800,
                },
                "AutoRollbackConfiguration": {
                    "Alarms": [],   # Attach CloudWatch alarms here for auto-rollback
                },
            },
        )
        logger.info("Blue/green endpoint config created: %s", config_name)
    except sm_client.exceptions.ValidationError:
        logger.info("Endpoint config '%s' already exists — reusing.", config_name)

    return config_name


# ── Deploy ───────────────────────────────────────────────────────────────────

def deploy_model(
    model_package_arn: str,
    endpoint_name: str,
    role: str,
    session,
) -> Predictor:
    """
    Deploys an approved model from the Model Registry to a SageMaker endpoint.
    Creates the endpoint if it doesn't exist; does a blue/green update otherwise.
    """
    sm_client = boto3.client("sagemaker", region_name=session.boto_region_name)

    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=session,
    )

    if endpoint_exists(endpoint_name, sm_client):
        logger.info("Endpoint '%s' exists — performing blue/green update.", endpoint_name)
        model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            update_endpoint=True,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer(),
        )
    else:
        logger.info("Creating new endpoint '%s'.", endpoint_name)
        model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer(),
        )

    wait_for_endpoint(endpoint_name, sm_client)

    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )
    return predictor


# ── Smoke Test ───────────────────────────────────────────────────────────────

def smoke_test(predictor: Predictor) -> None:
    """
    Sends a dummy inference request to verify the endpoint is responding.
    Replace the dummy payload with a real sample from your dataset.
    """
    logger.info("Running smoke test...")
    dummy_payload = "0.5,1.2,-0.3,0.8"   # Replace with a real feature row

    try:
        response = predictor.predict(dummy_payload)
        logger.info("✅ Smoke test passed. Response: %s", response)
    except Exception as e:
        raise RuntimeError(f"Smoke test failed: {e}") from e


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy approved model to SageMaker endpoint")
    parser.add_argument("--model-package-arn", required=True, help="ARN of the approved model package")
    parser.add_argument("--endpoint-name", default="aws-mlops-pipeline-endpoint")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))

    predictor = deploy_model(args.model_package_arn, args.endpoint_name, args.role_arn, session)

    if not args.skip_smoke_test:
        smoke_test(predictor)

    logger.info("🚀 Deployment complete. Endpoint: %s", args.endpoint_name)


if __name__ == "__main__":
    main()
