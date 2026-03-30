"""
monitoring/alerts.py
---------------------
Creates CloudWatch alarms that fire when SageMaker Model Monitor detects
data quality violations (drift). On alarm, an SNS notification is sent
AND an EventBridge rule automatically triggers a new pipeline execution
to retrain the model on fresh data.

Monitoring flow:
    Model Monitor (hourly) → CloudWatch metric violation
        → CloudWatch Alarm → SNS notification (email alert)
                           → EventBridge rule → SageMaker Pipeline re-run

Run once after drift_config.py:
    python alerts.py \
        --endpoint-name aws-mlops-pipeline-endpoint \
        --pipeline-arn arn:aws:sagemaker:eu-west-1:<account>:pipeline/aws-mlops-pipeline \
        --alert-email your@email.com \
        --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        --region eu-west-1
"""

import argparse
import json
import logging

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Step 1: SNS Topic for Email Alerts ───────────────────────────────────────

def create_sns_topic(endpoint_name: str, alert_email: str, region: str) -> str:
    """
    Creates an SNS topic and subscribes an email address to it.
    Returns the topic ARN.
    """
    sns = boto3.client("sns", region_name=region)
    topic_name = f"{endpoint_name}-drift-alerts"

    response = sns.create_topic(Name=topic_name)
    topic_arn = response["TopicArn"]
    logger.info("SNS topic created: %s", topic_arn)

    sns.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=alert_email,
    )
    logger.info("✅ Email subscription created for %s (confirm the email AWS sends you)", alert_email)
    return topic_arn


# ── Step 2: CloudWatch Alarm on Drift Metric ─────────────────────────────────

def create_drift_alarm(
    endpoint_name: str,
    topic_arn: str,
    region: str,
    threshold: float = 1.0,
) -> str:
    """
    Creates a CloudWatch alarm that fires when Model Monitor reports
    any data quality violation (metric >= threshold).
    Returns the alarm name.
    """
    cw = boto3.client("cloudwatch", region_name=region)
    alarm_name = f"{endpoint_name}-data-drift-alarm"

    cw.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription=(
            f"Fires when SageMaker Model Monitor detects data quality violations "
            f"on endpoint '{endpoint_name}'. Triggers automatic retraining."
        ),
        # SageMaker Model Monitor publishes this metric automatically
        Namespace="aws/sagemaker/Endpoints/data-metrics",
        MetricName="feature_baseline_drift",
        Dimensions=[{"Name": "Endpoint", "Value": endpoint_name}],
        Statistic="Sum",
        Period=3600,               # 1-hour window (matches monitoring schedule)
        EvaluationPeriods=1,
        Threshold=threshold,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        OKActions=[topic_arn],
    )
    logger.info("✅ CloudWatch alarm created: %s", alarm_name)
    return alarm_name


# ── Step 3: EventBridge Rule → Auto-Retrain on Alarm ─────────────────────────

def create_retraining_trigger(
    alarm_name: str,
    pipeline_arn: str,
    role_arn: str,
    region: str,
) -> None:
    """
    Creates an EventBridge rule that listens for the CloudWatch alarm
    state change to ALARM, then automatically starts a new SageMaker
    Pipeline execution to retrain the model.
    """
    events = boto3.client("events", region_name=region)
    rule_name = f"{alarm_name}-retrain-trigger"

    # EventBridge pattern: fires when the alarm transitions to ALARM state
    event_pattern = json.dumps({
        "source": ["aws.cloudwatch"],
        "detail-type": ["CloudWatch Alarm State Change"],
        "detail": {
            "alarmName": [alarm_name],
            "state": {"value": ["ALARM"]},
        },
    })

    events.put_rule(
        Name=rule_name,
        EventPattern=event_pattern,
        State="ENABLED",
        Description=f"Auto-retrain trigger: fires when {alarm_name} enters ALARM state",
    )
    logger.info("EventBridge rule created: %s", rule_name)

    # Target: start a new SageMaker Pipeline execution
    events.put_targets(
        Rule=rule_name,
        Targets=[
            {
                "Id": "SageMakerPipelineTarget",
                "Arn": pipeline_arn,
                "RoleArn": role_arn,
                "SageMakerPipelineParameters": {
                    "PipelineParameterList": []   # Uses pipeline defaults; add overrides here if needed
                },
            }
        ],
    )
    logger.info("✅ EventBridge target set → pipeline will auto-retrain on drift: %s", pipeline_arn)


# ── Step 4: Summary ──────────────────────────────────────────────────────────

def print_summary(endpoint_name: str, alert_email: str, alarm_name: str, pipeline_arn: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎯  DRIFT ALERTING FULLY CONFIGURED")
    logger.info("=" * 60)
    logger.info("  Endpoint monitored : %s", endpoint_name)
    logger.info("  CloudWatch alarm   : %s", alarm_name)
    logger.info("  Email alerts       : %s", alert_email)
    logger.info("  Auto-retrain target: %s", pipeline_arn)
    logger.info("")
    logger.info("  ⚠️  Check your inbox and confirm the SNS email subscription.")
    logger.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up CloudWatch alarms and auto-retraining trigger")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--pipeline-arn", required=True, help="ARN of the SageMaker Pipeline to retrigger")
    parser.add_argument("--alert-email", required=True, help="Email address to receive drift alerts")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for EventBridge → SageMaker")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--drift-threshold", type=float, default=1.0,
                        help="Number of violations to trigger the alarm (default: 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    topic_arn = create_sns_topic(args.endpoint_name, args.alert_email, args.region)
    alarm_name = create_drift_alarm(args.endpoint_name, topic_arn, args.region, args.drift_threshold)
    create_retraining_trigger(alarm_name, args.pipeline_arn, args.role_arn, args.region)
    print_summary(args.endpoint_name, args.alert_email, alarm_name, args.pipeline_arn)


if __name__ == "__main__":
    main()
