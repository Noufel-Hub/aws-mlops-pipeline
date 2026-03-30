"""
monitoring/drift_config.py
---------------------------
Sets up SageMaker Model Monitor on a deployed endpoint:
  1. Captures live inference request/response data to S3
  2. Creates a baseline from the training data distribution
  3. Schedules hourly data quality checks against that baseline
  4. Any violation triggers a CloudWatch alarm → auto-retraining (see alerts.py)

Run once after deploying your endpoint:
    python drift_config.py \
        --endpoint-name aws-mlops-pipeline-endpoint \
        --baseline-data-uri s3://your-bucket/pipeline/preprocessing/train/train.csv \
        --bucket your-bucket \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole
"""

import argparse
import logging

import boto3
import sagemaker
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
    EndpointInput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MONITOR_INSTANCE_TYPE = "ml.m5.xlarge"


# ── Step 1: Enable Data Capture on the Endpoint ──────────────────────────────

def enable_data_capture(endpoint_name: str, bucket: str, session) -> None:
    """
    Configures the endpoint to capture 100% of requests and responses to S3.
    This is the raw material for drift detection.
    """
    capture_uri = f"s3://{bucket}/monitoring/data-capture/{endpoint_name}"

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,          # Capture all traffic
        destination_s3_uri=capture_uri,
        capture_options=["REQUEST", "RESPONSE"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"],
    )

    sm_client = boto3.client("sagemaker", region_name=session.boto_region_name)
    sm_client.update_endpoint(
        EndpointName=endpoint_name,
        RetainAllVariantProperties=True,
        DeploymentConfig={
            "DataCaptureConfig": {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": capture_uri,
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            }
        },
    )
    logger.info("✅ Data capture enabled → %s", capture_uri)


# ── Step 2: Create Baseline from Training Data ───────────────────────────────

def create_baseline(
    baseline_data_uri: str,
    bucket: str,
    role: str,
    session,
) -> str:
    """
    Runs a SageMaker baseline job to compute statistics and constraints
    from the training data. These become the reference distribution for
    future drift comparisons.
    Returns the S3 URI of the baseline results.
    """
    baseline_results_uri = f"s3://{bucket}/monitoring/baseline-results"

    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=MONITOR_INSTANCE_TYPE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    logger.info("Running baseline job — this may take a few minutes...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        wait=True,
        logs=False,
    )
    logger.info("✅ Baseline complete → %s", baseline_results_uri)
    return baseline_results_uri


# ── Step 3: Schedule Hourly Monitoring ──────────────────────────────────────

def schedule_monitoring(
    endpoint_name: str,
    baseline_results_uri: str,
    bucket: str,
    role: str,
    session,
) -> None:
    """
    Creates a monitoring schedule that runs every hour, comparing live
    captured data against the baseline statistics and constraints.
    Violations are published to CloudWatch as metrics.
    """
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=MONITOR_INSTANCE_TYPE,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    violations_uri = f"s3://{bucket}/monitoring/violations/{endpoint_name}"

    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-monitor",
        endpoint_input=EndpointInput(
            endpoint_name=endpoint_name,
            destination="/opt/ml/processing/input/endpoint",
        ),
        output_s3_uri=violations_uri,
        statistics=f"{baseline_results_uri}/statistics.json",
        constraints=f"{baseline_results_uri}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,   # Publishes metrics → triggers alarms
    )
    logger.info("✅ Monitoring schedule created (hourly) for endpoint: %s", endpoint_name)
    logger.info("   Violations will be written to: %s", violations_uri)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure drift monitoring for a SageMaker endpoint")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--baseline-data-uri", required=True, help="S3 URI of training CSV used for baseline")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="eu-west-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))

    enable_data_capture(args.endpoint_name, args.bucket, session)
    baseline_uri = create_baseline(args.baseline_data_uri, args.bucket, args.role_arn, session)
    schedule_monitoring(args.endpoint_name, baseline_uri, args.bucket, args.role_arn, session)

    logger.info("🎯 Drift monitoring fully configured for endpoint: %s", args.endpoint_name)


if __name__ == "__main__":
    main()
