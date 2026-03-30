# 🚀 End-to-End MLOps Pipeline on AWS

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![AWS](https://img.shields.io/badge/cloud-AWS-orange)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade MLOps pipeline built on AWS, demonstrating automated model training, versioning, deployment, and monitoring. Developed as a hands-on complement to the **AWS Certified Machine Learning Engineer – Associate** certification.

---

## 🎯 Objective

Design and implement a fully automated ML lifecycle on AWS — from raw data ingestion to live inference — incorporating best practices in model governance, reproducibility, and drift detection.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                               │
│                                                                 │
│   S3 (Raw Data)                                                 │
│       │                                                         │
│       ▼                                                         │
│   SageMaker Pipelines ──────────────────────────────────────┐  │
│   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │  │
│   │ Preprocessing│──▶│  Training    │──▶│  Evaluation &   │  │  │
│   │   Step       │   │  Step        │   │  Model Registry │  │  │
│   └─────────────┘   └──────────────┘   └────────┬────────┘  │  │
│                                                  │            │  │
│                                         ┌────────▼────────┐  │  │
│                                         │  SageMaker       │  │  │
│                                         │  Endpoint        │  │  │
│                                         └────────┬────────┘  │  │
│                                                  │            │  │
│   CloudWatch ◀───────────────────────────────────┘            │  │
│   (Drift Detection + Alerts + Auto-Retraining Trigger) ───────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 AWS Stack

| Service | Role |
|---|---|
| **S3** | Raw data storage, model artefact versioning |
| **SageMaker Pipelines** | Orchestration of preprocessing → training → evaluation |
| **SageMaker Model Registry** | Model versioning and approval workflow |
| **SageMaker Endpoints** | Real-time inference deployment |
| **CloudWatch** | Monitoring, drift detection alerts, retraining triggers |
| **IAM** | Least-privilege role management across services |

---

## 📦 Project Structure

```
aws-mlops-pipeline/
│
├── pipeline/
│   ├── preprocessing.py       # Feature engineering & data validation
│   ├── training.py            # Model training step (scikit-learn / PyTorch)
│   ├── evaluation.py          # Metrics computation & threshold check
│   └── pipeline_definition.py # SageMaker Pipeline assembly
│
├── monitoring/
│   ├── drift_config.py        # CloudWatch data quality baseline
│   └── alerts.py              # SNS alarm definitions
│
├── deploy/
│   └── endpoint.py            # Endpoint creation & update logic
│
├── notebooks/
│   └── exploration.ipynb      # EDA and baseline experiments
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Steps

1. **Data Ingestion** — Pull raw data from S3, validate schema and completeness
2. **Preprocessing** — Feature engineering, train/validation split, artefact upload to S3
3. **Training** — SageMaker managed training job with hyperparameter logging
4. **Evaluation** — Compute metrics (accuracy, F1, RMSE); gate on threshold before registration
5. **Model Registry** — Version and tag approved models; enforce approval workflow
6. **Deployment** — Blue/green endpoint update with zero-downtime rollout
7. **Monitoring** — CloudWatch data quality checks; automatic retraining trigger on drift

---

## 📊 Key Features

- ✅ Fully automated retraining triggered by data drift detection
- ✅ Model versioning with approval gates before production deployment
- ✅ Infrastructure defined as code — reproducible and portable
- ✅ End-to-end experiment tracking via SageMaker Experiments
- ✅ Cost-aware design — training jobs terminate on completion

---

## 🗺️ Roadmap

- [x] Pipeline architecture design
- [x] S3 data ingestion & preprocessing step
- [x] SageMaker training & evaluation steps
- [x] Model Registry integration
- [x] Endpoint deployment with blue/green rollout
- [x] CloudWatch monitoring & drift alerts
- [x] Full pipeline integration test

---

## 👤 Author

**Noufel Anougmar** — AI Engineer, EPFL M.Sc. Robotics + Data Science  
[github.com/Noufel-Hub](https://github.com/Noufel-Hub) · noufel.anougmar@gmail.com

---

*Built as part of preparation for the AWS Certified Machine Learning Engineer – Associate certification.*
