# Real-Time Anti-Fraud Microservice

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange.svg)

An end-to-end Machine Learning Operations (MLOps) pipeline and real-time microservice built to detect fraudulent transactions. Engineered specifically for high-scale environments (like Ad-Tech and FinTech), this project breaks away from standard academic notebooks by prioritizing **ultra-low latency**, **concept drift monitoring**, and **Business Risk Optimization** over simple accuracy.

---

### Visual Walkthrough



Once the FastAPI server is running locally, you can easily test the real-time inference engine using the auto-generated Swagger UI.

**1. Access the UI:**
Navigate to `http://127.0.0.1:8000/docs` in your browser.

**2. Submit a Test Payload:**
Click on the `POST /predict` endpoint, click **"Try it out"**, and paste the following JSON payload. This specific payload represents a high-risk transaction (e.g., an online order far from home, not using a PIN/Chip, with a massive purchase price ratio):

```json
{
  "distance_from_home": 150.5,
  "distance_from_last_transaction": 0.5,
  "ratio_to_median_purchase_price": 8.5,
  "repeat_retailer": 0,
  "used_chip": 0,
  "used_pin_number": 0,
  "online_order": 1
}
```
The payload submitted in the demo — `ratio_to_median_purchase_price: 8.5`, `used_chip: 0`, `online_order: 1` — was deliberately constructed to represent a high-risk account takeover pattern.

**3. The Prediction Response:**
Click "Execute". The dual-engine pipeline will process the payload in milliseconds and return a response indicating whether the transaction crossed the cost-optimized threshold (0.15) and is flagged as fraud.

---

## Why This Project & Dataset

I chose fraud detection deliberately — not because it is a popular Kaggle topic, but because class imbalance is one of the most consistently mishandled problems in production ML. Most tutorials SMOTE their way through it without considering what synthetic oversampling costs at inference time or how it pollutes real-world drift signals. I wanted to build something that would survive a code review at a real fintech company.

The dataset (1M credit card transactions from Kaggle) was chosen because its ~9% fraud rate sits in a realistic range — high enough to train on, low enough that naïve accuracy metrics become dangerously misleading. It also contains behavioral signals (`distance_from_home`, `ratio_to_median_purchase_price`, `used_chip`) that map directly to real-world feature engineering patterns used in UPI and card-network fraud teams.

---

## Project Narrative (STAR)

### Situation
In high-scale industries processing billions of asynchronous user events (ad-tech, mobile payments, food delivery), fraud detection systems face three compounding hurdles:

1. **Extreme Class Imbalance:** Fraud represents < 0.1% to 9% of traffic, making standard models biased and accuracy metrics highly misleading.
2. **Strict Latency Limits:** Systems must identify anomalies and respond in milliseconds to prevent workflow friction or transaction drops.
3. **Concept Drift:** Fraudsters constantly change tactics, rendering static models obsolete within weeks.

### Task
Architect a production-ready, end-to-end ML pipeline that ingests data, engineers "velocity" features, trains a dual-engine anomaly detection system, and serves predictions via an API — all while tying model performance directly to a **Business Cost Matrix**, not abstract mathematical metrics.

### Action (The Pipeline)

* **Data Engineering:** Automated data ingestion from Kaggle with programmatic schema validation (`data_loader.py`) to catch upstream data quality issues before they corrupt training.
* **Feature Engineering:** Developed custom behavioral features like `purchase_velocity_risk` and `is_location_anomaly` to detect sudden behavioral shifts and geospatial discrepancies (`features.py`). These were not arbitrary — they were grounded in EDA findings that sudden spikes in `ratio_to_median_purchase_price` are statistically strong indicators of account takeover.
* **Dual-Engine ML:**
  * **XGBoost** for supervised learning. *Reasoning: Tabular data industry standard, natively handles extreme imbalance via `scale_pos_weight` (avoiding the latency overhead of SMOTE), and delivers millisecond inference.*
  * **Isolation Forest** as an unsupervised second engine. *Reasoning: Fraudsters constantly evolve. A supervised model trained on yesterday's fraud patterns is blind to new attack vectors. The Isolation Forest acts as a zero-day radar.*
* Wrapped both models in a **FastAPI** microservice and containerized using **Docker** for cloud-portable deployment.

### Result (Business Impact & Metrics)

Evaluated on a holdout test set of 200,000 highly imbalanced transactions:

| Metric | Value |
|---|---|
| Model AUPRC | 0.9997 |
| False Negatives (Missed Fraud) | 1 |
| False Positives (User Friction) | 401 |
| Inference Latency | 0.57ms – 11.7ms |

**The Business Cost Matrix — Translating ML into Revenue:**

In production, models are evaluated by their financial impact, not mathematical scores. The pipeline abandons standard accuracy metrics and evaluates predictions using a custom Business Cost Matrix (`src/evaluate.py`).

Different error types carry drastically different business costs:
* **False Negative (Missed Fraud):** Costs the business **$100** per occurrence — stolen funds, chargeback fees, and lost goods.
* **False Positive (User Friction):** Costs the business **$5** per occurrence — customer service overhead and potential churn from a declined legitimate card.

**Holdout Set Calculation:**
* 1 Missed Fraudster × $100 = $100
* 401 Declined Legitimate Users × $5 = $2,005
* **Total Theoretical Financial Loss: $2,105**

**Baseline Comparison (Justifying the 401 False Positives):**

A standard model optimized purely for accuracy or a balanced F1-Score might reduce False Positives to near-zero — but let 50 fraudsters slip through:
* 50 Missed Frauds × $100 = $5,000
* 0 Annoyed Users × $5 = $0
* **Baseline Total Loss: $5,000**

By explicitly tuning the classification threshold against the Business Cost Matrix, **the system saved an estimated $2,895** on this 200K batch alone — proving that optimizing for bottom-line revenue preservation is categorically different from chasing mathematical perfection.

---

## Key Decisions & Trade-offs

In production MLOps, there are no perfect solutions — only calculated compromises. Each decision below involved explicitly accepting a downside.

| Decision | Alternative Considered | What I Gave Up | Why It Was Worth It |
|---|---|---|---|
| `scale_pos_weight` over SMOTE | SMOTE oversampling | Marginal recall gains from synthetic minority examples | 10× faster CI/CD training loop; model grounded 100% in real behavior; no synthetic noise polluting drift signals |
| AUPRC over F1 | F1-Score | A single, easy-to-communicate number | F1 is threshold-dependent and collapses under class imbalance; AUPRC evaluates across all thresholds |
| Threshold at 0.15 over 0.5 | Default 0.5 binary threshold | Balanced F1-Score | $2,895 saved vs. baseline; the 0.5 threshold is arbitrary in business contexts |
| Isolation Forest over One-Class SVM | One-Class SVM | Kernel-based decision boundary precision | Scales linearly to 1M+ rows; no memory overhead from kernel trick |

### The Three Core Tensions

**1. SMOTE vs. `scale_pos_weight`**

SMOTE can sometimes produce cleaner decision boundaries by synthesizing nuanced minority-class examples. However, it adds massive computational overhead to the training pipeline and introduces synthetic noise that blurs the signal of real-world concept drift. `scale_pos_weight` trades a marginal theoretical recall gain for a drastically faster CI/CD loop and a model anchored entirely in real user behavior.

**2. Isolation Forest — "Weird" vs. "Fraudulent"**

The Isolation Forest flags statistical anomalies, but anomalous does not automatically mean fraudulent. A user legitimately traveling abroad for the first time will trigger the anomaly detector, generating a False Positive. This is an accepted cost. The alternative, relying solely on the supervised XGBoost model, is complete blindness to fraud patterns that have never appeared in training data. The `contamination` parameter was tuned conservatively to use the Isolation Forest as an early-warning radar, not a primary classifier.

**3. The Probability Threshold**

The default 0.5 classification threshold is completely arbitrary in business contexts. It treats a missed fraud the same as a wrongly blocked card, which is financially indefensible when one costs 20× more than the other. The threshold was shifted from 0.5 to 0.15 by systematically evaluating total business cost at each threshold increment and selecting the minimum — a calibration exercise that had more business impact than any hyperparameter tuning decision.

## What I Learned (Project Retrospective)

Building this pipeline from scratch fundamentally shifted my perspective on what makes an ML project successful in production.

**Threshold Calibration > Algorithm Selection.** Going in, I assumed the model choice was the critical decision. In reality, moving the classification threshold from 0.5 to 0.15 had more business impact than the entire XGBoost hyperparameter search. That calibration exercise forced me to quantify exactly how much user friction a business is willing to tolerate to prevent theft — which is ultimately a business negotiation, not a data science problem.

**ML is a Business Tool, Not a Math Test.** I initially felt frustrated that extreme class imbalance was producing a "misleading" F1-Score. Building the custom `evaluate_business_cost()` function reframed the entire problem: the goal was never to maximize F1, it was to minimize net financial loss. Once the right question was being asked, the right answer became obvious.

**Explainability is Not Optional in Production.** Implementing the FastAPI endpoint surfaced a real-world gap: outputting a fraud probability to a frontend is not enough. A customer support agent needs to know *why* a card was blocked ("Flagged due to geospatial velocity spike, 3 transactions in 2 cities within 4 minutes") to handle disputes effectively. Real-time SHAP values are already built into the evaluation pipeline (`src/evaluate.py`) — the next step is surfacing them in the API response. This is also increasingly a regulatory requirement: EU GDPR Article 22 and emerging RBI guidelines for automated credit decisions both mandate explainability for ML-driven blocking decisions.

## Scaling for High-TPS Environments (Real-World Applicability)

This repository demonstrates a functional microservice. Deploying to a high-volume environment — UPI transactions, ad-tech bidding, or a food delivery platform like Swiggy at peak — requires evolving the architecture from a static REST API into a continuous, self-healing system.

**Closing the Concept Drift Loop:**
Currently, the Isolation Forest flags zero-day anomalies passively. In a real system, any transaction flagged by Isolation Forest but passed by XGBoost would be routed to a human-in-the-loop review queue. Once reviewed, those records become labeled ground truth for the next training cycle, forcing the supervised model to rapidly absorb new fraud vectors without waiting for a periodic scheduled retrain.

**Automated Continuous Training (CT Pipeline):**
Fraud models degrade quickly — sometimes within days of a new attack pattern emerging. A production CT pipeline would train on a rolling 30-day window, validate on the most recent 3 days, and compute the new model's `Total Business Cost`. A deployment would only trigger if the challenger model demonstrably saves more money than the champion currently in production. Model quality gates, not schedules.

**Feature Store for Sub-5ms Latency:**
This baseline assumes all transaction features are passed in the API JSON payload. In production (Razorpay, PhonePe), pulling a user's `median_purchase_price` history from a SQL database at inference time is too slow and creates coupling between the model and the transactional database. An in-memory Feature Store (Redis) holding pre-computed user behavioral profiles would decouple these systems and bring true end-to-end latency well below 5ms.

**Event-Driven Architecture:**
A standard HTTP POST endpoint creates bottlenecks during traffic spikes (sale events, IPL match nights on gaming platforms). Decoupling the microservice by attaching it to an event stream (Apache Kafka or AWS Kinesis) allows the model to consume and score transactions asynchronously, without blocking the user's checkout flow. This is the standard architecture at scale for any Indian fintech operating on UPI rails.

---

## Data Source

The data is sourced from the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).
* Contains 1,000,000 records.
* Features include: `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`.

## Repository Architecture

```text
CreditCardFraudAnalysis/
├── app/
│   └── api.py                  # FastAPI real-time inference microservice
├── notebooks/
│   └── 01_EDA_and_Drift.ipynb  # Initial research, imbalance profiling, baseline drift metrics
├── src/
│   ├── data_loader.py          # Automated data ingestion and schema validation
│   ├── features.py             # Velocity & distance feature engineering logic
│   ├── models.py               # ML Engine (XGBoost & Isolation Forest)
│   └── evaluate.py             # AUPRC, SHAP values, and Business Cost Matrix
├── models/                     # Saved .pkl models (generated after pipeline run)
├── run_pipeline.py             # Main orchestrator for CI/CD training runs
├── Dockerfile                  # Containerization specifications
└── requirements.txt            # Python environment dependencies
```


## Instructions

**1. Clone and Install:**
```bash
git clone https://github.com/your-username/CreditCardFraudAnalysis.git
cd CreditCardFraudAnalysis
pip install -r requirements.txt
```

**2. Execute the Training Pipeline:**
This will automatically download the 1M+ row dataset, engineer features, train both models, output business metrics, and save the `.pkl` files.
```bash
python run_pipeline.py
```

**3. Launch the API Server:**
```bash
python -m uvicorn app.api:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to test the API via the interactive Swagger UI.

**4. Docker Deployment:**
To deploy this microservice in a cloud environment (e.g., AWS ECS, Google Cloud Run):
```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```
