# Machine Learning Fundamentals: A Production-Quality Engineering Guide

> *A deep-dive resource for engineers who want to truly understand ML — not just run it.*

---

## What You Will Learn

This document covers the **complete conceptual and practical foundation of Machine Learning** — from understanding what ML actually is, to formulating problems correctly, building pipelines, avoiding pitfalls, and deploying real systems.

By the end, you will be able to:
- Distinguish ML from statistics and rules-based systems — and know when *not* to use ML
- Formulate any real-world problem as an ML task
- Build robust train/validate/test pipelines that don't leak data
- Understand what a model *really* is, and why loss functions drive learning
- Select appropriate evaluation metrics for any problem
- Diagnose bias–variance tradeoffs in deployed models
- Execute an end-to-end ML workflow with production awareness

## Who This Is For

- Junior/mid-level engineers transitioning into ML roles
- Data scientists who want to solidify their foundations
- Engineers preparing for ML system design interviews
- Anyone who has "run sklearn" but wants to understand *why it works*

---

## Table of Contents

1. [What Machine Learning Really Is](#1-what-machine-learning-really-is)
   - [ML vs Traditional Programming](#11-ml-vs-traditional-programming)
   - [Why ML Works: Pattern Discovery from Data](#12-why-ml-works-pattern-discovery-from-data)
   - [ML vs Statistics vs Rules-Based Systems](#13-ml-vs-statistics-vs-rules-based-systems)
   - [When NOT to Use ML](#14-when-not-to-use-ml)
2. [Types of Machine Learning](#2-types-of-machine-learning)
   - [Supervised Learning](#21-supervised-learning)
   - [Unsupervised Learning](#22-unsupervised-learning)
   - [Semi-Supervised Learning](#23-semi-supervised-learning)
   - [Self-Supervised Learning](#24-self-supervised-learning)
   - [Reinforcement Learning](#25-reinforcement-learning)
   - [Batch vs Online Learning](#26-batch-vs-online-learning)
3. [ML Problem Formulation](#3-ml-problem-formulation)
   - [Regression vs Classification](#31-regression-vs-classification)
   - [Binary vs Multi-Class vs Multi-Label](#32-binary-vs-multi-class-vs-multi-label)
   - [Clustering vs Density Estimation](#33-clustering-vs-density-estimation)
   - [Anomaly Detection](#34-anomaly-detection)
   - [Ranking & Recommendation Problems](#35-ranking--recommendation-problems)
4. [Train / Validation / Test Split](#4-train--validation--test-split)
   - [Why Splitting Data Matters](#41-why-splitting-data-matters)
   - [Data Leakage](#42-data-leakage)
   - [Hold-Out Validation](#43-hold-out-validation)
   - [Cross-Validation](#44-cross-validation)
5. [Model Concepts](#5-model-concepts)
   - [What is a Model?](#51-what-is-a-model)
   - [Parameters vs Hyperparameters](#52-parameters-vs-hyperparameters)
   - [Hypothesis Space](#53-hypothesis-space)
   - [Decision Boundaries](#54-decision-boundaries)
   - [Linear vs Non-Linear Models](#55-linear-vs-non-linear-models)
6. [Loss Functions](#6-loss-functions)
   - [What is a Loss Function?](#61-what-is-a-loss-function)
   - [Loss vs Error vs Metric](#62-loss-vs-error-vs-metric)
   - [Regression Losses](#63-regression-losses)
   - [Classification Losses](#64-classification-losses)
   - [Why Models Minimize Loss, Not Accuracy](#65-why-models-minimize-loss-not-accuracy)
7. [Evaluation Metrics](#7-evaluation-metrics)
   - [Regression Metrics](#71-regression-metrics)
   - [Classification Metrics](#72-classification-metrics)
   - [Confusion Matrix](#73-confusion-matrix)
   - [Choosing the Right Metric](#74-choosing-the-right-metric)
8. [Bias–Variance Tradeoff](#8-biasvariance-tradeoff)
9. [End-to-End ML Workflow](#9-end-to-end-ml-workflow)
10. [Common ML Pitfalls](#10-common-ml-pitfalls)
11. [Cross-Topic Relationships](#11-cross-topic-relationships)
12. [End-to-End Real-World Projects](#12-end-to-end-real-world-projects)
    - [Project 1: Credit Card Fraud Detection](#project-1-credit-card-fraud-detection)
    - [Project 2: House Price Prediction](#project-2-house-price-prediction)
13. [Algorithm Comparison Tables](#13-algorithm-comparison-tables)
14. [Common Mistakes & Pitfalls Reference](#14-common-mistakes--pitfalls-reference)
15. [Interview Preparation](#15-interview-preparation)
16. [Resources](#16-resources)

---

# 1. What Machine Learning Really Is

## 1.1 ML vs Traditional Programming

### a. Intuition

Imagine you want to build a spam filter. In **traditional programming**, you'd sit down and write explicit rules:

```
IF email contains "Nigerian prince" → spam
IF email contains "FREE MONEY" → spam
IF sender not in contacts AND subject has "!!!" → spam
```

This works — until spammers change tactics. Now your filter breaks and you rewrite rules. Forever.

**Machine Learning flips this entirely.** Instead of writing rules, you show the system thousands of examples of spam and not-spam, and let it *discover the rules itself*. You provide data + outcomes; the algorithm figures out the pattern.

```
Traditional:  Rules + Data → Answers
ML:           Data + Answers → Rules (learned automatically)
```

Think of it like teaching a child. You don't give a child a rulebook for recognizing cats. You show them hundreds of cats and say "this is a cat." Eventually they generalize.

### b. How It Works (Step-by-Step)

1. **Collect labeled examples** (emails + spam/not-spam labels)
2. **Choose a model architecture** (what kind of function to fit)
3. **Train** — the model adjusts its internal numbers to minimize errors
4. **Evaluate** — check performance on unseen emails
5. **Deploy** — the learned rules handle new emails automatically

### c. Visual Representation

```
TRADITIONAL PROGRAMMING
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   Raw Data  │────▶│  Hand-coded  │────▶│  Output  │
│  (email)    │     │    Rules     │     │ (spam?)  │
└─────────────┘     └──────────────┘     └──────────┘
                           ▲
                    Engineer writes this

MACHINE LEARNING
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   Raw Data  │────▶│   ML Model   │────▶│  Output  │
│  (email)    │     │  (learned)   │     │ (spam?)  │
└─────────────┘     └──────────────┘     └──────────┘
      +                    ▲
┌─────────────┐            │
│   Labels    │────────────┘
│ (spam/not)  │     Training process discovers rules
└─────────────┘
```

### d. When to Use / Avoid

| Use ML When | Avoid ML When |
|---|---|
| Rules are too complex to write by hand | A simple if-else handles it reliably |
| Patterns change over time | You have very little data |
| Input space is huge (images, text) | Problem requires strict auditability |
| You have labeled data | Prediction errors have catastrophic costs |

---

## 1.2 Why ML Works: Pattern Discovery from Data

### a. Intuition

ML works because **real-world data is not random** — it has structure. House prices correlate with square footage. Fraudulent transactions cluster in unusual time windows. Customer churn follows behavioral patterns.

ML algorithms are essentially very sophisticated **curve-fitting machines**. They find mathematical relationships in data that generalize to new, unseen examples.

The key insight: **if your training data is representative of reality**, a model that learns from it can make predictions about reality it has never seen before.

### b. Mathematical Insight

A model learns a function `f` such that:

```
f(X) ≈ y
```

Where `X` is input features and `y` is the target. The model adjusts `f` to minimize prediction error across thousands of examples.

The generalization magic: by minimizing error on training data with the right model complexity, the learned `f` tends to work on new data — *if training data was representative*.

### c. The Core Assumption (Often Violated!)

ML rests on the **i.i.d. assumption**: training and test data are drawn from the **same distribution**, independently. Violate this, and your model fails in production even with perfect training metrics.

---

## 1.3 ML vs Statistics vs Rules-Based Systems

### a. Intuition

These three approaches often solve overlapping problems but differ in **goals, assumptions, and outputs**.

| Dimension | Rules-Based | Statistics | Machine Learning |
|---|---|---|---|
| Goal | Execute predefined logic | Understand relationships, test hypotheses | Predict / classify with high accuracy |
| How rules are created | Hand-coded by humans | Inferred with uncertainty estimates | Learned automatically from data |
| Interpretability | High — logic is explicit | High — coefficients have meaning | Low to medium (varies by model) |
| Data requirement | Minimal | Moderate | Often large |
| Adapts to new patterns | No | No (re-analyze needed) | Yes (retraining) |
| Uncertainty quantification | No | Yes (p-values, CIs) | Sometimes (Bayesian, calibration) |

### b. When Statistics Wins Over ML

- You have 50 samples, not 50,000
- You need to answer "does X *cause* Y?" (ML finds correlation, not causation)
- You need confidence intervals and hypothesis tests
- You're publishing research that requires interpretability

### c. The Practitioner's Rule of Thumb

> Start with a simple rule or a statistical model. Only reach for ML when the simpler approach demonstrably fails.

---

## 1.4 When NOT to Use ML

This is one of the most underrated skills in ML engineering. Knowing when *not* to use ML saves enormous time, money, and credibility.

### a. The Anti-Patterns

**1. When you don't have enough data**
ML models need examples to learn from. For complex tasks, "enough" can mean thousands to millions of samples. With 200 rows, a logistic regression or a lookup table often outperforms a neural network.

**2. When a deterministic rule works**
Calculating tax from income brackets doesn't need ML. A lookup table or formula is faster, more accurate, and fully auditable.

**3. When interpretability is required by law**
Credit scoring, medical diagnosis, and loan approval often require explanations ("why was I denied?"). Black-box ML can create legal liability.

**4. When the cost of errors is catastrophic**
Autonomous surgery or nuclear safety systems require formal verification, not probabilistic models.

**5. When data distribution shifts constantly**
If the world changes faster than you can retrain, ML models become stale and dangerous. Consider hybrid rule + ML systems.

**6. When you can't validate properly**
If you have no reliable way to measure model quality (no ground truth, no holdout set), you're flying blind.

### b. The Decision Framework

```
Is the task too complex for rules?
├─ No  → Write rules. Done.
└─ Yes → Do you have quality labeled data?
          ├─ No  → Collect data first, or use unsupervised
          └─ Yes → Is the error cost acceptable?
                    ├─ No  → Revisit constraints, add human review
                    └─ Yes → ML is appropriate
```

---

# 2. Types of Machine Learning

## 2.1 Supervised Learning

### a. Intuition

You are the teacher. You provide the model with inputs *and* correct answers. The model learns to map inputs to outputs by studying thousands of examples with known labels.

**Analogy**: Teaching a student by giving them textbook problems *with* an answer key. The student learns patterns that let them solve new problems.

### b. Formal Definition

Given a dataset `{(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}`, learn a function `f: X → Y` that generalizes to new (x, y) pairs not seen during training.

### c. Types of Supervised Tasks

- **Regression**: Predict continuous values (house price, temperature)
- **Classification**: Predict discrete categories (spam/not, disease type)

### d. Examples in Practice

| Problem | Input (X) | Output (y) |
|---|---|---|
| Email spam detection | Email text, metadata | Spam / Not Spam |
| House price prediction | Area, location, bedrooms | Price ($) |
| Medical diagnosis | Lab results, symptoms | Disease present / absent |
| Credit scoring | Financial history | Credit score |

### e. Python Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Supervised learning: binary classification example
# X: features, y: labels (0 or 1)
X = np.array([[2.5, 3.0], [1.0, 1.5], [3.5, 4.0], [0.5, 0.8],
              [4.0, 3.8], [1.2, 1.0], [3.8, 3.5], [0.8, 1.2]])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Split: model never sees test data during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train: model learns from labeled examples
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on unseen test data
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

---

## 2.2 Unsupervised Learning

### a. Intuition

No teacher, no labels. The model explores raw data and discovers **hidden structure on its own** — groupings, patterns, compressed representations.

**Analogy**: Giving a student a pile of library books (no subject labels) and asking them to organize by similarity. They'll likely create meaningful groups without being told what categories to use.

### b. Common Tasks

| Task | Algorithm | Example |
|---|---|---|
| Clustering | K-Means, DBSCAN | Customer segmentation |
| Dimensionality reduction | PCA, t-SNE | Visualizing high-dim data |
| Density estimation | GMM, KDE | Anomaly detection |
| Association rules | Apriori | Market basket analysis |

### c. Python Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Unsupervised: cluster customers by behavior (no labels)
customer_data = np.array([
    [500, 2],   # [monthly_spend, purchases_per_month]
    [520, 3],
    [480, 2],
    [5000, 25],
    [4800, 22],
    [5200, 28],
    [250, 1],
    [270, 1],
])

# Scale features — K-Means is distance-based, sensitive to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# Discover 3 customer segments automatically
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print("Cluster assignments:", labels)
# Output might reveal: [low-value, mid-value, high-value] segments
```

---

## 2.3 Semi-Supervised Learning

### a. Intuition

Labeling data is **expensive**. Doctors labeling medical scans, lawyers annotating contracts — this costs time and money. Semi-supervised learning uses a **small labeled dataset** combined with a **large unlabeled dataset**.

**Analogy**: You have 10 labeled photos of diseases and 10,000 unlabeled scans. Semi-supervised methods use the structure of the 10,000 unlabeled scans to improve the model trained on 10 labeled ones.

### b. Key Idea

The unlabeled data helps the model understand the **underlying distribution** (what "typical" data looks like), even without knowing the labels. This regularizes the model and improves generalization.

### c. Common Approaches

- **Self-training**: Train on labeled data → predict labels for unlabeled → retrain on combined
- **Label propagation**: Spread labels through a graph of similar points
- **Pseudo-labeling**: High-confidence predictions on unlabeled data become training examples

### d. When to Use

- Medical imaging (few labeled scans, many raw scans)
- NLP (small annotated corpus, huge raw text)
- Any scenario where labeling is the bottleneck

---

## 2.4 Self-Supervised Learning

### a. Intuition

Self-supervised learning is a clever trick: **the labels are generated from the data itself**. No human annotation needed.

**Classic example — BERT (language model)**: Mask 15% of words in a sentence. Train the model to predict the masked words. The "labels" are the original words — automatically available.

**Another example — image rotation**: Rotate images by 0°, 90°, 180°, 270°. Train the model to predict the rotation angle. No human labels needed — the rotation angle is the self-generated label.

### b. Why This Matters

Self-supervised learning is how modern LLMs (GPT, BERT, Claude) learn from the internet without anyone labeling billions of text examples. The task of "predict the next word" is self-supervised — the label is always the next word in the text.

### c. Conceptual Framework

```
Raw Data → Automatically create task → Train on self-generated labels
         ↑
     No human annotation needed
```

---

## 2.5 Reinforcement Learning

### a. Intuition (High-Level)

An **agent** interacts with an **environment**, takes **actions**, and receives **rewards** or **penalties**. The agent learns a **policy** — a strategy for which action to take in any situation — to maximize cumulative reward over time.

**Analogy**: Training a dog. You don't hand it a manual. When it sits on command, it gets a treat (reward). When it chews your shoes, it gets scolded (penalty). Over time, it learns which behaviors pay off.

### b. Key Components

```
        ┌──────────────────────────────────────┐
        │              Environment              │
        │                                      │
        │  State (sₜ) ──────────────────────── │
        │      │                               │
        ▼      ▼                               │
     Agent  Action (aₜ)                        │
        │      │                               │
        │      ▼                               │
        │   Reward (rₜ)  ◄──────────────────── │
        │   State (sₜ₊₁) ◄──────────────────── │
        └──────────────────────────────────────┘
```

### c. Real-World Applications

- Game playing (AlphaGo, chess engines)
- Robot locomotion and manipulation
- Personalized recommendation systems
- Trading strategies
- RLHF (Reinforcement Learning from Human Feedback) — used to fine-tune LLMs

### d. When to Use

RL is appropriate when: you have a **clear reward signal**, actions have **delayed consequences**, and simulating many episodes is feasible. It's computationally expensive and rarely the first tool to reach for.

---

## 2.6 Batch vs Online Learning

### a. Intuition

**Batch learning**: Train on all available data at once, deploy the model, use it until you retrain. Simple. Most common.

**Online learning**: The model **continuously updates** as new data arrives. Each new example adjusts the model in real-time.

**Analogy**:
- Batch learning = studying for an exam for 3 months, then taking it. No changes during the test.
- Online learning = learning continuously from every conversation. The model improves with each interaction.

### b. Comparison

| Dimension | Batch Learning | Online Learning |
|---|---|---|
| Training frequency | Periodic (daily, weekly) | Continuous |
| Memory requirement | Needs full dataset | Only current example |
| Adaptation speed | Slow (requires retraining) | Fast (immediate) |
| Complexity | Lower | Higher (concept drift management) |
| Use case | Most production ML systems | Recommender systems, trading, streaming |

### c. Python Implementation (Online Learning)

```python
from sklearn.linear_model import SGDClassifier
import numpy as np

# SGD (Stochastic Gradient Descent) supports online learning via partial_fit
model = SGDClassifier(loss='log_loss', random_state=42)

# Simulate streaming data — model updates with each batch
data_stream = [
    (np.array([[0.5, 1.2]]), np.array([0])),
    (np.array([[3.1, 2.8]]), np.array([1])),
    (np.array([[0.3, 0.8]]), np.array([0])),
    (np.array([[2.9, 3.2]]), np.array([1])),
]

for X_batch, y_batch in data_stream:
    # partial_fit updates model without retraining from scratch
    model.partial_fit(X_batch, y_batch, classes=[0, 1])

print("Model updated incrementally on streaming data")
```

---

# 3. ML Problem Formulation

> **This is the most underappreciated ML skill.** A misformulated problem leads to wasted months of engineering. Getting this right upfront is worth more than any algorithm choice.

## 3.1 Regression vs Classification

### a. Intuition

**Regression**: Predict a **continuous number**.
- What will this house sell for? → $347,000
- What will tomorrow's temperature be? → 72°F

**Classification**: Predict a **discrete category**.
- Will this email be spam? → Yes / No
- What digit is in this image? → 7

The **output type** determines the problem type. This sounds simple but engineers frequently misformulate:
- Predicting a **rating (1-5 stars)** can be treated as either regression (predict the number) or classification (predict the class) — your choice has consequences.
- Predicting **age** is regression. Predicting **age group** (child/adult/senior) is classification.

### b. Visual Representation

```
REGRESSION                         CLASSIFICATION
                                   
   Price                           ● ● ● ○ ○ ○
   $400K ─────────────────/        ● ● ○ ○ ○ ○
   $300K ──────────/               ● ○ ○ ○ ○ ○
   $200K ──/                       ─────────────
   $100K /                              Decision
          1000 2000 3000 sqft           Boundary

   Continuous output                Discrete output
```

### c. Key Algorithms by Task

| Task | Common Algorithms |
|---|---|
| Regression | Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost |
| Classification | Logistic Regression, SVM, Random Forest Classifier, XGBoost Classifier |

### d. Python Implementation

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# --- REGRESSION EXAMPLE ---
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
print(f"Regression prediction sample: {reg_model.predict(X_test_r[:3])}")
# Output: [147.3, -23.8, 89.1]  ← continuous numbers

# --- CLASSIFICATION EXAMPLE ---
X_clf, y_clf = make_classification(n_samples=200, n_features=5, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2)

clf_model = LogisticRegression()
clf_model.fit(X_train_c, y_train_c)
print(f"Classification prediction sample: {clf_model.predict(X_test_c[:3])}")
# Output: [1, 0, 1]  ← discrete classes
```

---

## 3.2 Binary vs Multi-Class vs Multi-Label

### a. Intuition

**Binary**: Two possible outputs. Spam or not. Disease present or absent.

**Multi-class**: More than two possible outputs, but exactly **one** applies to each example.
- Image classification: cat, dog, or bird (not cat AND dog)
- Digit recognition: 0, 1, 2, ..., 9

**Multi-label**: More than two possible outputs, and **multiple can apply** simultaneously.
- Movie genre tagging: [Action, Comedy, Romance] — a movie can be all three
- Medical diagnosis: a patient can have multiple conditions simultaneously

### b. Visual Representation

```
BINARY           MULTI-CLASS       MULTI-LABEL
Input → [0|1]    Input → [A|B|C]   Input → [A?, B?, C?]
                                   Each independently yes/no
```

### c. Python Implementation

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Multi-label: each sample can belong to multiple classes
X, y = make_multilabel_classification(
    n_samples=500, n_features=10, n_classes=4, n_labels=2, random_state=42
)
# y shape: (500, 4) — each row has multiple 1s

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# OneVsRestClassifier trains one binary classifier per label
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

predictions = model.predict(X_test[:3])
print("Multi-label predictions (each row = one sample):")
print(predictions)
# Output: [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]
```

---

## 3.3 Clustering vs Density Estimation

### a. Clustering: Intuition

Clustering **groups similar data points together** without predefined labels. You ask: "Are there natural groups in this data?"

**Example**: Given customer purchase histories, clustering might reveal: power users, casual browsers, and seasonal shoppers — segments you didn't define upfront.

**Key algorithms**: K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models

### b. Density Estimation: Intuition

Density estimation **models the probability distribution** of the data. Instead of "which group does this point belong to?", it answers "how likely is this point?"

**Example**: Modeling normal user behavior as a probability distribution. Points with very low probability under this distribution are anomalies.

**Key algorithms**: Kernel Density Estimation (KDE), Gaussian Mixture Models (GMM)

### c. Python Implementation

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np

# Generate sample data
np.random.seed(42)
cluster1 = np.random.normal([0, 0], 0.5, (50, 2))
cluster2 = np.random.normal([5, 5], 0.5, (50, 2))
X = np.vstack([cluster1, cluster2])

# --- CLUSTERING: K-Means ---
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print("K-Means cluster labels:", np.unique(labels))

# --- DENSITY ESTIMATION: KDE ---
# Fit a density model to understand the data distribution
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(X)

# Score a few points: log probability (higher = more likely)
test_points = np.array([[0, 0], [2.5, 2.5], [5, 5]])
log_probs = kde.score_samples(test_points)
print("Log densities at test points:", log_probs)
# [0,0] and [5,5] should be highest (cluster centers)
# [2.5,2.5] should be lowest (between clusters)
```

---

## 3.4 Anomaly Detection

### a. Intuition

Anomaly detection finds **data points that don't fit the expected pattern**. These outliers are often more interesting than normal points: fraud, equipment failure, medical abnormalities, network intrusions.

The core challenge: **anomalies are rare by definition**. You may have 10,000 normal transactions and 50 fraudulent ones. Standard classifiers struggle; you often need specialized techniques.

### b. Approaches

| Approach | How It Works | When to Use |
|---|---|---|
| Statistical | Flag points beyond N standard deviations | Univariate, Gaussian data |
| Isolation Forest | Anomalies are easier to isolate in trees | General-purpose, high-dimensional |
| One-Class SVM | Learn boundary of "normal" data | When labeled anomalies unavailable |
| Autoencoder | High reconstruction error = anomaly | Image/text anomalies |
| DBSCAN | Noise points = anomalies | Spatial/clustered data |

### c. Python Implementation

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulate transaction data with rare anomalies
np.random.seed(42)
normal_transactions = np.random.normal([100, 5], [20, 1], (1000, 2))
# [amount, transactions_per_day]
anomalous_transactions = np.array([
    [5000, 50],   # Huge amount, many transactions
    [1, 100],     # Tiny amounts, extremely high frequency
    [999, 1],     # Very large single transaction
])

X = np.vstack([normal_transactions, anomalous_transactions])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest: contamination = expected fraction of anomalies
iso_forest = IsolationForest(contamination=0.003, random_state=42)
predictions = iso_forest.fit_predict(X_scaled)
# Returns: 1 (normal), -1 (anomaly)

anomaly_indices = np.where(predictions == -1)[0]
print(f"Detected {len(anomaly_indices)} anomalies at indices: {anomaly_indices}")
```

---

## 3.5 Ranking & Recommendation Problems

### a. Intuition

**Ranking**: Given a query, order items by relevance. This is what search engines do. Predict not just "is this relevant?" but "how relevant *relative to other items*?"

**Recommendation**: Predict which items a user will like based on their history and the behavior of similar users.

These problems are subtle: you're not predicting an absolute value, you're predicting **relative ordering**. A model that predicts "item A will get 4.5 stars and item B will get 4.4 stars" is useful if item A actually ranks higher — even if the absolute predictions are wrong.

### b. Types of Recommendation Systems

```
Collaborative Filtering: "Users like you also liked..."
    → Uses behavior of similar users

Content-Based Filtering: "Because you liked X (which is sci-fi)..."
    → Uses item features

Hybrid: Combines both
    → Most production systems
```

### c. Key Metrics for Ranking

- **NDCG (Normalized Discounted Cumulative Gain)**: How good is your ranking? Penalizes relevant items ranked lower.
- **MAP (Mean Average Precision)**: Average precision across queries
- **Hit Rate**: Did the correct item appear in top-K recommendations?

---

# 4. Train / Validation / Test Split

> This section is critical. More ML bugs come from incorrect data splitting than from wrong algorithms.

## 4.1 Why Splitting Data Matters

### a. Intuition

Imagine you're a student who is given the **final exam questions in advance** while studying. You'd ace the exam — but you haven't truly learned. If someone gave you different questions, you'd fail.

ML models face the same problem. A model **evaluated on the same data it was trained on** will always look great. It has memorized the answers. To know if a model actually *generalized*, you must test it on data it has **never seen**.

This is why we split data: training data teaches, validation data guides choices, test data measures true performance.

### b. The Three-Way Split Explained

```
Total Dataset (100%)
        │
        ├──────────────────────────────────────────┐
        │                                          │
    Training Set (70%)           ┌─── Validation Set (15%)
    "Study material"             │    "Practice exams"
    Model learns from this       │    Tune hyperparameters
                                 └─── Test Set (15%)
                                      "Final exam"
                                      Touch ONCE, at the end
```

### c. When to Use Each Set

| Set | When Used | Purpose |
|---|---|---|
| Training | During model.fit() | Learn parameters |
| Validation | During model selection | Compare models, tune hyperparameters |
| Test | Once, at the very end | Estimate real-world performance |

### d. Python Implementation

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Generate sample dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Step 1: Split off test set first (never touch until the end)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Step 2: Split remainder into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
    # 0.176 of 0.85 ≈ 0.15 of total → gives ~70/15/15 split
)

print(f"Train size:      {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(X):.0%})")
print(f"Test size:       {len(X_test)} ({len(X_test)/len(X):.0%})")
```

---

## 4.2 Data Leakage

### a. Intuition

**Data leakage** is when information from the **test set or future** sneaks into the **training process**, making your model look better than it really is. It's one of the most dangerous bugs in ML — often invisible until deployment.

**Analogy**: A student who memorizes last year's exam questions performs amazingly in the practice run, then fails the actual exam (different questions). The "leak" was past exam questions contaminating their preparation for the real test.

### b. Types of Leakage

**1. Target Leakage**: A feature contains information about the target that wouldn't be available at prediction time.
- Example: Using "date of hospitalization" to predict "will patient be hospitalized?" — hospitalization is already known!
- Example: Using a "fraud flag" created by the fraud team as a feature to predict fraud.

**2. Train-Test Contamination**: Information from test data influences training.
- Example: Fitting a StandardScaler on the full dataset (including test), then splitting. The scaler "saw" test data statistics.
- Example: Doing feature selection using the full dataset, then training on the "selected" features.

**3. Time Leakage**: Using future data to predict the past.
- Example: Predicting stock prices using features calculated from future prices.

### c. The Canonical Leakage Mistake

```python
# ❌ WRONG — Scaler fitted on full dataset before splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Sees test data statistics!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# The scaler used test set mean/std → leaked!

# ✅ CORRECT — Scaler fitted only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)         # Only transform test
```

### d. Detecting Leakage

**Warning signs**:
- Model accuracy is suspiciously high (98%+ on a hard problem)
- Model performs perfectly during evaluation but fails in production
- Feature importance shows a feature shouldn't logically matter
- Validation loss is lower than training loss

---

## 4.3 Hold-Out Validation

### a. Intuition

The simplest validation strategy: **hold out** a fixed portion of data, train on the rest, evaluate once on the held-out set. Fast, simple, good enough for large datasets (100K+ examples).

### b. Limitations

- High variance: if your hold-out set happened to be "easy", metrics look inflated
- Wastes data: 20% of data never contributes to training
- Unreliable with small datasets (< 1000 samples)

---

## 4.4 Cross-Validation

### a. Intuition

Instead of one fixed train/validation split, **rotate through K different splits**. Each fold takes a turn being the validation set. You get K estimates of performance and average them.

**Result**: A more reliable estimate of true model performance, especially with small datasets.

```
K-Fold Cross-Validation (K=5):

Fold 1:  [TEST ][Train][Train][Train][Train]  → Score 1
Fold 2:  [Train][TEST ][Train][Train][Train]  → Score 2
Fold 3:  [Train][Train][TEST ][Train][Train]  → Score 3
Fold 4:  [Train][Train][Train][TEST ][Train]  → Score 4
Fold 5:  [Train][Train][Train][Train][TEST ]  → Score 5

Final Score = mean(Score 1...5) ± std(Score 1...5)
```

### b. Python Implementation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# StratifiedKFold preserves class balance in each fold — critical for imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=1000)

# cross_val_score handles the splitting, training, and evaluation automatically
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"Individual fold scores: {scores}")
# A small std indicates consistent performance across folds (good)
# A large std indicates high sensitivity to data split (bad)
```

### c. When to Use Which

| Situation | Recommended Approach |
|---|---|
| Large dataset (>100K) | Hold-out validation (fast) |
| Small dataset (<10K) | 5-fold or 10-fold CV |
| Very small dataset (<1K) | Leave-One-Out CV (LOOCV) |
| Imbalanced classes | Stratified K-Fold |
| Time-series data | TimeSeriesSplit (never shuffle!) |

### d. The Golden Rule

> **The test set is touched exactly once.** Not for feature selection, not for hyperparameter tuning, not for sanity checks. Once. At the very end. This is non-negotiable.

---

# 5. Model Concepts

## 5.1 What is a Model?

### a. Intuition

A model is a **mathematical function** that maps inputs to outputs. It's the "learned thing" — the set of rules the algorithm extracted from data.

More precisely: a model is a **parameterized function**. The parameters are numbers inside the model. Training is the process of finding the right parameter values.

**Analogy**: A model is like a recipe. The recipe structure (e.g., "mix flour, water, and salt") is the architecture. The exact amounts (2 cups flour, 1 tsp salt) are the parameters. Training finds the right amounts.

### b. Concrete Example

Linear regression model:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

Here:
- `ŷ` = prediction
- `x₁, x₂, ..., xₙ` = input features
- `w₀, w₁, ..., wₙ` = **parameters** (weights) — learned from data

The model structure (linear combination) is chosen by the engineer. The weights are learned by the algorithm.

---

## 5.2 Parameters vs Hyperparameters

### a. Intuition

This distinction confuses many beginners. Here's the clearest way to think about it:

**Parameters**: Numbers **inside** the model, learned **automatically** during training from data.
- Examples: weights in a linear regression, thresholds in a decision tree

**Hyperparameters**: Settings you configure **before** training begins. The algorithm doesn't learn these — you set them.
- Examples: learning rate, number of trees, maximum depth, regularization strength

**Analogy**:
- Hyperparameters are like the **oven temperature** and **baking time** — you set these before cooking.
- Parameters are like the **exact shape** the dough takes in the oven — determined by the cooking process itself.

### b. Comparison Table

| Property | Parameters | Hyperparameters |
|---|---|---|
| Who sets them? | Learning algorithm | Engineer |
| When are they set? | During training | Before training |
| Saved in model file? | Yes | Usually in config |
| Examples (LinearRegression) | Coefficients, intercept | `fit_intercept` |
| Examples (RandomForest) | Tree structure, thresholds | `n_estimators`, `max_depth` |

### c. Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, random_state=42)

# HYPERPARAMETERS: you set these (the algorithm doesn't learn them)
model = RandomForestClassifier(
    n_estimators=100,    # hyperparameter: number of trees
    max_depth=5,         # hyperparameter: tree depth limit
    min_samples_split=10, # hyperparameter: splitting threshold
    random_state=42
)

model.fit(X, y)

# PARAMETERS: learned from data (the structure of each tree)
# Access the learned trees:
print(f"Number of trees: {len(model.estimators_)}")
print(f"First tree depth: {model.estimators_[0].get_depth()}")
# These vary depending on the training data — they are learned
```

---

## 5.3 Hypothesis Space

### a. Intuition

The **hypothesis space** is the set of all possible functions your model architecture can represent. When you choose "logistic regression", you restrict your search to all possible linear decision boundaries. When you choose "deep neural network", you expand the search to a vastly larger space.

**Analogy**: Imagine you're searching for the perfect shape that fits your data.
- Linear regression: you can only consider straight lines
- Polynomial regression (degree 2): curves open up
- Neural networks: almost any shape imaginable

The hypothesis space is constrained by your model architecture choice.

### b. Why This Matters

- If the true relationship in your data is non-linear, and you choose linear regression, no amount of training data or tuning will fix it — the right answer isn't in your hypothesis space.
- If your hypothesis space is too large (e.g., a massive neural network on 100 samples), the model can fit noise perfectly (overfitting).

```
Hypothesis Space

     ┌─────────────────────────────────────┐
     │   All Possible Functions            │
     │                                     │
     │  ┌──────────────────────────────┐   │
     │  │  Non-linear Models           │   │
     │  │                              │   │
     │  │  ┌───────────────────────┐   │   │
     │  │  │  Linear Models        │   │   │
     │  │  │  (logistic, linear    │   │   │
     │  │  │   regression)         │   │   │
     │  │  └───────────────────────┘   │   │
     │  │  (SVMs, trees, forests)      │   │
     │  └──────────────────────────────┘   │
     │  (Neural networks expand this)      │
     └─────────────────────────────────────┘
```

---

## 5.4 Decision Boundaries

### a. Intuition

For classification, a **decision boundary** is the line (or surface) that separates different classes in feature space. Everything on one side → class A. Everything on the other → class B.

**Linear models** produce straight-line (or flat hyperplane) boundaries.
**Non-linear models** (trees, SVMs with kernels, neural nets) can produce curved, complex boundaries.

### b. Visual Representation

```
Linear Decision Boundary        Non-linear Decision Boundary
(Logistic Regression)           (Random Forest)

  ●●   |  ○○○                    ●●●│○○│●●
  ●  ● |   ○                     ●● │○○│ ●
  ●●   |  ○○                     ────┘  └────
       |                           ○○○○○○
  Class ●  Class ○               ●●●│○○○│●●●

Straight line separates         Jagged boundary fits data
                                exactly (risk of overfitting)
```

### c. Why This Matters in Practice

- If two classes aren't linearly separable, logistic regression will always make some errors — not because of bad training, but because the model **can't represent** the true boundary.
- Non-linear models can capture complex boundaries but are **more prone to overfitting**.

---

## 5.5 Linear vs Non-Linear Models

### a. Intuition

**Linear models** assume the relationship between features and target is a **weighted sum**. Simple, interpretable, but limited.

**Non-linear models** can represent **arbitrary relationships** — interactions between features, thresholds, exponential patterns.

### b. When Linear Models Surprisingly Work

For many real business problems (especially with good feature engineering), linear models are competitive with or beat complex non-linear models because:
- Data is often approximately linear after proper feature engineering
- Non-linear models overfit on small datasets
- Linear models are faster, more interpretable, and easier to debug

### c. The Rule of Thumb

> Try a strong linear baseline first. Only graduate to non-linear models when you have evidence the linear model is the bottleneck.

| Model Type | Examples | Hypothesis Space |
|---|---|---|
| Linear | Linear/Logistic Regression, SVM (linear) | Straight boundaries |
| Non-linear | Decision Trees, Random Forest, XGBoost, Neural Nets | Curved/complex boundaries |

---

# 6. Loss Functions

## 6.1 What is a Loss Function?

### a. Intuition

A **loss function** is the model's "report card" during training. It measures **how wrong** the model's predictions are. The learning algorithm's entire job is to minimize this number.

**Analogy**: Imagine you're teaching someone to throw darts. The loss function is the distance from the bullseye. Training is the process of making adjustments to minimize this distance. The learner doesn't try to "win" directly — they try to minimize the distance score.

### b. Mathematical Insight

For a single prediction:
```
loss = L(ŷ, y)
```
Where `ŷ` is the prediction and `y` is the true label.

The model's goal during training:
```
minimize (1/n) Σ L(ŷᵢ, yᵢ)   over all training examples
```

The optimizer (e.g., gradient descent) adjusts parameters to make this sum smaller.

---

## 6.2 Loss vs Error vs Metric

### a. The Crucial Distinction

| Concept | What It Is | Who Uses It | Example |
|---|---|---|---|
| **Loss function** | Mathematical function optimized during training | The optimizer | Cross-entropy, MSE |
| **Error** | A specific type of loss or prediction mistake | Ambiguous term | MAE, prediction residual |
| **Metric** | Human-interpretable measure of model quality | You, stakeholders | Accuracy, F1, AUC |

**Key insight**: The metric you care about (accuracy, F1) is often **not differentiable** — you can't take its gradient. So you minimize a **proxy loss** (cross-entropy for classification) that you can differentiate, hoping that minimizing the loss improves the metric.

### b. Example of the Disconnect

You care about **accuracy** (% correct predictions). But:
- Accuracy is not differentiable — you can't compute gradients to update weights
- So you minimize **cross-entropy loss**, which *is* differentiable
- Cross-entropy correlates with accuracy but isn't identical
- Sometimes you can minimize loss without improving accuracy (rare but possible)

---

## 6.3 Regression Losses

### a. Mean Squared Error (MSE)

**Formula**:
```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

**Intuition**: Square each prediction error, then average. Squaring does two things:
1. Makes all errors positive (no cancellation)
2. **Heavily penalizes large errors** — an error of 10 is penalized 100x more than an error of 1

**When to use**: When large errors are especially bad (e.g., predicting patient weight for drug dosing).
**Limitation**: Highly sensitive to outliers.

### b. Mean Absolute Error (MAE)

**Formula**:
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

**Intuition**: Take the absolute value of each error, then average. Linear treatment — an error of 10 is penalized exactly 10x more than an error of 1.

**When to use**: When you want robustness to outliers (e.g., house price prediction where luxury homes are outliers).

### c. Visual Comparison

```
                Error = 0.5        Error = 2.0
MSE:           0.5² = 0.25        2.0² = 4.0   (amplified)
MAE:           |0.5| = 0.5        |2.0| = 2.0  (linear)

MSE penalizes large errors MUCH more than MAE
```

### d. Python Implementation

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 195, 305, 390, 550])

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MAE:  {mae:.2f}")   # Interpretable: average error in same units as y
print(f"RMSE: {rmse:.2f}")  # Same units as y, but penalizes large errors more
print(f"MSE:  {mse:.2f}")   # Used for optimization, less interpretable directly

# Demonstrate outlier sensitivity
y_with_outlier = np.array([100, 200, 300, 400, 2000])  # 2000 is an outlier
print(f"\nWith outlier:")
print(f"MAE:  {mean_absolute_error(y_true, y_with_outlier):.2f}")  # Increases moderately
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_with_outlier)):.2f}")  # Increases dramatically
```

---

## 6.4 Classification Losses

### a. Log Loss (Binary Cross-Entropy)

**Intuition**: Penalizes **confident wrong predictions** exponentially. If the model says "99% probability of class 1" but the true label is 0 — it gets hammered. If it says "60% probability" and is wrong, the penalty is smaller.

**Formula**:
```
Log Loss = -(1/n) Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
```

Where `p̂ᵢ` is the predicted probability of class 1.

**Why not just use accuracy?**
Accuracy only cares about "did you get it right or wrong?" It gives no credit for being *more confident* in the right answer, and doesn't penalize *excessive confidence* in the wrong answer.

Log loss captures probability calibration — a model that always says "51% chance of spam" (right classification) is worse than one that says "99% chance of spam" (right classification with confidence).

### b. Python Implementation

```python
from sklearn.metrics import log_loss
import numpy as np

y_true = np.array([1, 0, 1, 1, 0])

# Model 1: confident and correct
y_pred_confident = np.array([0.95, 0.05, 0.90, 0.88, 0.10])

# Model 2: correct but uncertain
y_pred_uncertain = np.array([0.55, 0.45, 0.60, 0.58, 0.42])

# Model 3: confident and WRONG
y_pred_wrong = np.array([0.05, 0.95, 0.10, 0.12, 0.90])

print(f"Confident + Correct:   {log_loss(y_true, y_pred_confident):.4f}")
print(f"Uncertain + Correct:   {log_loss(y_true, y_pred_uncertain):.4f}")
print(f"Confident + Wrong:     {log_loss(y_true, y_pred_wrong):.4f}")
# Confident + Wrong will be HUGE — log loss severely penalizes this
```

---

## 6.5 Why Models Minimize Loss, Not Accuracy

### a. The Core Issue: Differentiability

Gradient-based learning (how nearly all ML models train) requires computing the **gradient** — the direction to adjust weights to reduce error. This requires the loss function to be **differentiable**.

**Accuracy is not differentiable**:
- Predict 0.499 → class 0. Predict 0.501 → class 1.
- A tiny weight change either flips the class or doesn't. No smooth gradient.
- You can't tell "which direction makes accuracy better" in a continuous way.

**Cross-entropy IS differentiable**:
- Predict probability 0.499 → a smooth gradient exists
- Weight updates smoothly improve the probability toward the correct class

### b. Visual Intuition

```
Accuracy (step function):     Log Loss (smooth curve):
                              
accuracy                      loss
  1.0 ─── ────────              high ─ \
         |                            \  
  0.5                                  \
         |                         low ─ ─────────
  0.0 ───                               
        threshold              0   0.5   1.0
                               predicted probability

Cannot compute gradient       Can compute gradient everywhere
(discontinuous jump)          (smooth, continuous)
```

---

# 7. Evaluation Metrics

## 7.1 Regression Metrics

### a. MAE (Mean Absolute Error)

**Formula**: `MAE = (1/n) Σ |yᵢ - ŷᵢ|`

**Intuition**: On average, predictions are off by MAE units. If predicting house prices and MAE = $15,000, your predictions are off by $15K on average. Directly interpretable.

**When to use**: When outliers are present and you want robust evaluation.

### b. MSE (Mean Squared Error)

**Formula**: `MSE = (1/n) Σ (yᵢ - ŷᵢ)²`

**Intuition**: Same units squared, less interpretable. Used mainly as an optimization target.

### c. RMSE (Root Mean Squared Error)

**Formula**: `RMSE = √MSE`

**Intuition**: Same units as the target (like MAE), but penalizes large errors more. Most commonly reported in practice.

### d. R² (Coefficient of Determination)

**Formula**: `R² = 1 - (SS_res / SS_tot)`

Where `SS_res = Σ(yᵢ - ŷᵢ)²` and `SS_tot = Σ(yᵢ - ȳ)²`

**Intuition**: What fraction of variance in y does the model explain? R² = 1.0 → perfect predictions. R² = 0.0 → model is no better than predicting the mean. R² < 0 → model is worse than just predicting the mean (catastrophic).

### e. Python Implementation

```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

y_true = np.array([250000, 300000, 350000, 450000, 200000])
y_pred = np.array([260000, 285000, 360000, 440000, 210000])

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"MAE:  ${mae:,.0f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"R²:   {r2:.4f}")

# Interpreting R²:
# R² = 0.95 → model explains 95% of variance in house prices
# R² = 0.40 → model explains only 40% → leave a lot unexplained
```

---

## 7.2 Classification Metrics

### a. Accuracy

**Formula**: `Accuracy = Correct Predictions / Total Predictions`

**Intuition**: "Of all predictions, what fraction were right?"

**When accuracy is misleading**: 
- Disease detection: 99% of patients are healthy. A model that always predicts "healthy" has 99% accuracy but is useless.
- Always ask: **what is the class distribution?** If it's imbalanced, accuracy is often the wrong metric.

### b. Precision

**Formula**: `Precision = TP / (TP + FP)`

**Intuition**: "Of all the times I said 'positive', how often was I right?"

**Use when false positives are costly**: Spam detection (false positive = legitimate email marked spam — bad for user). Drug candidate screening (false positive = wasted expensive experiments).

### c. Recall (Sensitivity)

**Formula**: `Recall = TP / (TP + FN)`

**Intuition**: "Of all the actual positives, how many did I catch?"

**Use when false negatives are costly**: Cancer screening (false negative = missed cancer — potentially fatal). Fraud detection (false negative = missed fraud — financial loss).

### d. F1-Score

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Intuition**: The **harmonic mean** of precision and recall. Forces a balance between the two. High F1 requires *both* to be high — you can't game it by optimizing just one.

### e. Python Implementation

```python
from sklearn.metrics import (accuracy_score, precision_score, 
                              recall_score, f1_score, classification_report)
import numpy as np

# Binary classification example
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

# Comprehensive report for multi-class problems
print("\n--- Full Report ---")
print(classification_report(y_true, y_pred, target_names=['Not Fraud', 'Fraud']))
```

---

## 7.3 Confusion Matrix

### a. Intuition

A confusion matrix is a **2D breakdown** of all predictions for a binary classifier. It shows not just "how many were right" but *how* the model is failing.

```
                    PREDICTED
                  Negative  Positive
ACTUAL  Negative │   TN   │   FP   │
        Positive │   FN   │   TP   │
```

- **True Positive (TP)**: Predicted positive, actually positive ✓
- **True Negative (TN)**: Predicted negative, actually negative ✓
- **False Positive (FP)**: Predicted positive, actually negative ✗ (Type I Error)
- **False Negative (FN)**: Predicted negative, actually positive ✗ (Type II Error)

### b. Reading a Confusion Matrix

```
FRAUD DETECTION EXAMPLE:
                    PREDICTED
              Not Fraud   Fraud
ACTUAL  Not Fraud  9850  |  150   ← FP: 150 legitimate transactions flagged
        Fraud        30  |   70   ← FN: 30 actual frauds missed

Precision = 70 / (70 + 150) = 31.8%  ← Most "fraud alerts" are wrong!
Recall    = 70 / (70 + 30)  = 70.0%  ← Catching 70% of actual fraud
```

### c. Python Implementation

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

y_true = np.array([0]*100 + [1]*20)  # Imbalanced: 100 negative, 20 positive
# Model that mostly predicts negative
y_pred = np.array([0]*95 + [1]*5 + [0]*15 + [1]*5)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
```

---

## 7.4 Choosing the Right Metric

### a. The Decision Framework

```
What type of problem?
├── Regression
│   ├── Outliers matter less → MAE
│   ├── Outliers should be penalized → RMSE
│   └── Relative performance needed → R²
│
└── Classification
    ├── Classes balanced? → Accuracy is okay
    └── Classes imbalanced?
        ├── FP costly (spam filter, alert fatigue) → Precision
        ├── FN costly (cancer, fraud, safety) → Recall
        └── Balance both → F1-score
```

### b. Metric Cheat Sheet

| Metric | Type | Good When |
|---|---|---|
| MAE | Regression | Outliers present, interpretability needed |
| RMSE | Regression | Large errors especially bad |
| R² | Regression | Comparing models on same problem |
| Accuracy | Classification | Balanced classes |
| Precision | Classification | FP is costly |
| Recall | Classification | FN is costly |
| F1 | Classification | Both FP and FN matter, imbalanced classes |
| AUC-ROC | Classification | Need threshold-independent evaluation |

---

# 8. Bias–Variance Tradeoff

## 8.1 Intuition

**Bias** is the error from wrong assumptions in the learning algorithm. A high-bias model is too **simple** — it underfits the data and misses real patterns.

**Variance** is the model's sensitivity to fluctuations in training data. A high-variance model is too **complex** — it memorizes training data including noise, and fails on new data.

**Analogy**: 
- High bias = always shooting to the left of the target, regardless of your stance (systematic error, consistent but wrong)
- High variance = shots scattered all over the range (unpredictable, sensitive to small changes)
- Low bias + Low variance = tight cluster around the bullseye (what you want)

```
High Bias           High Variance       Balanced
(Underfitting)      (Overfitting)       (Just Right)

    ●                  ●  ●  ●              ●●
                    ●                      ●●●
                       ●                    ●

Consistently wrong  Randomly scattered    Consistent + Accurate
```

## 8.2 Mathematical Insight

The expected error of a model can be decomposed:

```
Total Error = Bias² + Variance + Irreducible Noise

Bias²:      How far off are predictions on average?
Variance:   How much do predictions vary with different training sets?
Irreducible: Noise in the data — cannot be reduced
```

This decomposition explains why you **can't always win**: reducing bias (more complex model) tends to increase variance, and vice versa.

## 8.3 Underfitting vs Overfitting

### a. Diagnosing Each

**Underfitting (High Bias)**:
- Training loss is high
- Validation loss is similarly high
- Model is too simple

**Overfitting (High Variance)**:
- Training loss is very low (near zero)
- Validation loss is significantly higher
- Gap between train and validation performance

```
Loss
 │   Overfit zone ───────────────────────
 │                     /── Validation Loss
 │                    /
 │          ─────────── Training Loss
 │─────────
 └────────────────────────────────────
   Simple     ───────────────►    Complex
            Model Complexity

Sweet spot: Where validation loss is minimized
```

### b. Python Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate data with a true cubic relationship + noise
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 2 * X.ravel() + 0.5 * X.ravel()**2 - 0.02 * X.ravel()**3 + np.random.randn(100) * 3

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

results = []
for degree in [1, 2, 3, 5, 10, 20]:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    train_mse = mean_squared_error(y_train, pipeline.predict(X_train))
    val_mse   = mean_squared_error(y_val,   pipeline.predict(X_val))
    results.append((degree, train_mse, val_mse))

print(f"{'Degree':>8} {'Train MSE':>12} {'Val MSE':>12} {'Diagnosis':>15}")
print("-" * 55)
for degree, train_mse, val_mse in results:
    gap = val_mse / train_mse
    if degree <= 1:
        diagnosis = "Underfitting"
    elif gap < 1.5:
        diagnosis = "Good fit"
    else:
        diagnosis = "Overfitting"
    print(f"{degree:>8} {train_mse:>12.2f} {val_mse:>12.2f} {diagnosis:>15}")
```

## 8.4 Practical Implications

| Symptom | Root Cause | Fix |
|---|---|---|
| High train error, high val error | Underfitting (high bias) | More complex model, better features |
| Low train error, high val error | Overfitting (high variance) | Regularization, more data, simpler model |
| Both errors are low | Good balance | Deploy |
| Both errors high but val < train | Something is wrong | Check for data leakage |

## 8.5 Why This Tradeoff Never Disappears

No model is both perfectly flexible (zero bias) and perfectly stable (zero variance). As you increase model complexity:
- Bias decreases (model can express more)
- Variance increases (model is more sensitive to training data)

The optimal model sits at the **minimum of total error** — the sweet spot between these two forces. Finding this sweet spot is the essence of model selection and regularization.

---

# 9. End-to-End ML Workflow

## 9.1 The Full Pipeline

```
1. Problem Definition
       ↓
2. Data Collection
       ↓
3. Data Understanding (EDA)
       ↓
4. Data Cleaning & Preprocessing
       ↓
5. Feature Engineering
       ↓
6. Model Selection (Baseline First)
       ↓
7. Train–Validate Loop
   ┌──────────────────────────────────────────────┐
   │ a. Train model on training set               │
   │ b. Evaluate on validation set                │
   │ c. Analyze errors                            │
   │ d. Adjust: features, hyperparameters, model  │
   │ e. Repeat until satisfactory                 │
   └──────────────────────────────────────────────┘
       ↓
8. Final Evaluation on Test Set (ONCE)
       ↓
9. Deployment Awareness
```

## 9.2 Step-by-Step Breakdown

### Step 1: Problem Definition
Before writing any code:
- What exactly is the business question?
- What data do you have access to?
- What does success look like? (metric + target value)
- What are the constraints? (latency, interpretability, fairness)

### Step 2: Data Collection
- Identify sources (databases, APIs, scraped data, third-party vendors)
- Check data quality early — garbage in, garbage out
- Understand labeling strategy (who labeled? how? when?)

### Step 3: Data Understanding (EDA)

```python
import pandas as pd
import numpy as np

def eda_summary(df: pd.DataFrame) -> None:
    """Quick EDA summary for any dataset."""
    print("=== Shape ===")
    print(df.shape)
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({'missing': missing, 'pct': missing_pct})
          [missing > 0].sort_values('pct', ascending=False))
    
    print("\n=== Numeric Summary ===")
    print(df.describe())
    
    print("\n=== Target Distribution ===")
    target = df.iloc[:, -1]
    print(target.value_counts(normalize=True).round(4))
```

### Step 4: The Train–Validate Loop in Practice

The train-validate loop is **iterative**, not linear. You try things, observe what happens, and adjust.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Start with the simplest reasonable baseline
baseline = LogisticRegression(max_iter=1000)
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring='f1')
print(f"Baseline F1: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")

# 2. Graduate to more complex model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"Random Forest F1: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

# 3. Compare — is the improvement worth the complexity?
```

### Step 5: Deployment Awareness

Even at the ML fundamentals stage, think about deployment:
- **Latency**: How fast must predictions be? (real-time vs batch)
- **Data pipeline**: How will new data flow in production?
- **Retraining**: How often must the model be updated?
- **Monitoring**: How will you detect model degradation?

---

# 10. Common ML Pitfalls

## 10.1 Data Leakage

**The Pitfall**: Using information at training time that wouldn't be available at prediction time.

**Real-World Example**: Predicting loan default using "days past due" — this measure itself confirms the default. The model learns a tautology.

**Prevention**:
- Always ask: "Would I have this feature at the time of prediction?"
- Fit all transformers (scalers, encoders) on training data only
- Use time-based splits for time-series data

## 10.2 Improper Evaluation

**The Pitfall**: Evaluating on data that influenced training in any way.

**Common Mistakes**:
- Hyperparameter tuning on the test set
- Feature selection using the full dataset before splitting
- Reporting the best single fold score instead of average CV score

**Prevention**:
- Nested cross-validation for hyperparameter search
- Never touch the test set until the final evaluation

## 10.3 Ignoring Baseline Models

**The Pitfall**: Jumping to complex models without establishing what a simple model can achieve.

**Real Cost**: Engineers spend weeks tuning XGBoost to get 82% accuracy — when logistic regression with good features gets 80%. The 2% improvement may not justify the complexity.

**Always build**:
1. **Random baseline**: Predict most common class (or mean)
2. **Simple rule baseline**: One or two hand-crafted rules
3. **Simple model baseline**: Logistic regression or linear regression

Only beat these before pursuing complex models.

## 10.4 Blindly Trusting Metrics

**The Pitfall**: Optimizing a metric without understanding what it measures — and what it doesn't.

**Examples**:
- 99% accuracy on a dataset that's 99% class 0 → model predicts class 0 every time
- Low training loss but catastrophic failures on specific edge cases not captured in evaluation
- AUC of 0.95 but model is miscalibrated (probabilities are wrong)

**Prevention**:
- Look at the confusion matrix, not just summary metrics
- Analyze error cases manually — what kinds of mistakes is the model making?
- Validate metrics against business outcomes

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Classic imbalanced dataset trap
y_true = np.array([0]*99 + [1]*1)   # 99% class 0, 1% class 1

# "Dumb" model: always predict class 0
y_pred_dumb = np.zeros(100, dtype=int)

# "Mediocre" model: catches some positives but isn't perfect  
y_pred_model = np.array([0]*97 + [1]*2 + [0]*1)

print("=== Dumb Model (always predicts 0) ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred_dumb):.2%}")  # 99% accuracy!
print(f"F1-Score: {f1_score(y_true, y_pred_dumb, zero_division=0):.4f}")  # 0.0

print("\n=== Real Model ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred_model):.2%}")  # 98% accuracy
print(f"F1-Score: {f1_score(y_true, y_pred_model, zero_division=0):.4f}")  # Better

# Accuracy alone made the dumb model look better!
```

---

# 11. Cross-Topic Relationships

## 11.1 The Flow of Ideas

Understanding how these concepts connect is as important as understanding each individually.

```
Problem Formulation
       │
       ▼
Data Splitting (prevents leakage)
       │
       ▼
Model Choice (defines hypothesis space)
       │
       ▼
Loss Function (drives optimization)
       │                    ◄── Connected: wrong loss = wrong model
       ▼
Training (minimizes loss on training set)
       │
       ▼
Evaluation (on validation set with right metric)
       │
       ▼
Bias–Variance Analysis (diagnose under/overfitting)
       │                    ◄── Connected: high variance → regularize
       ▼
Hyperparameter Tuning (find sweet spot in hypothesis space)
       │
       ▼
Final Evaluation (test set, once)
```

## 11.2 Key Dependencies

| Concept | Directly Depends On | Affects |
|---|---|---|
| Loss function choice | Problem type (regression/classification) | What the model learns |
| Evaluation metric | Business objective, class distribution | Whether you can trust results |
| Model complexity | Dataset size, feature quality | Bias–variance balance |
| Data splitting | Dataset size, time structure | Reliability of all metrics |
| Bias–variance balance | Model complexity + regularization | Generalization |

## 11.3 The Interplay: A Concrete Example

> *Predicting customer churn (binary classification, imbalanced: 5% churn)*

1. **Problem formulation**: Classification, binary, imbalanced → F1 over accuracy
2. **Data splitting**: Temporal split (not random) — can't use future data to predict past churn
3. **Model**: Logistic regression first (linear baseline), then gradient boosting
4. **Loss function**: Binary cross-entropy (standard for binary classification)
5. **Evaluation**: F1-score on the minority (churn) class
6. **Bias–variance**: If train F1 = 0.85 and val F1 = 0.55 → overfitting → regularize or reduce features
7. **Pitfall to avoid**: Don't include "churned at time T" in features to predict "churned at time T" → leakage

---

# 12. End-to-End Real-World Projects

---

## Project 1: Credit Card Fraud Detection

### a. Problem Statement

**Business Context**: A fintech company processes 10 million transactions per day. Fraudulent transactions cost the company $5M monthly. The fraud team manually reviews flagged transactions — they can handle 1,000 reviews per day. Your model must flag the most suspicious transactions with high precision (minimize false alarms) while catching as many frauds as possible.

**ML Task**: Binary classification (fraud vs. not fraud). Highly imbalanced — ~0.17% of transactions are fraudulent.

**Success Metric**: F1-score on the fraud class (class 1), with secondary attention to Precision (false alarm rate matters for operations).

### b. Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Description**:
- 284,807 transactions over 2 days
- 492 fraudulent transactions (0.17%)
- Features: V1–V28 (PCA-transformed for privacy), Amount, Time
- Target: Class (0 = legitimate, 1 = fraud)

### c. Step-by-Step Pipeline

### d. Full Code Pipeline

```python
# ============================================================
# PROJECT 1: Credit Card Fraud Detection
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────
# STEP 1: DATA LOADING
# ──────────────────────────────────────
# Load dataset (download from Kaggle or generate synthetic version)
# df = pd.read_csv('creditcard.csv')

# For reproducibility, generate a realistic synthetic version
np.random.seed(42)
n_normal = 10000
n_fraud = 50  # ~0.5% fraud rate

# Normal transactions: lower amounts, PCA features near 0
X_normal = np.random.randn(n_normal, 28)
amounts_normal = np.abs(np.random.exponential(100, n_normal))
X_normal = np.column_stack([X_normal, amounts_normal])
y_normal = np.zeros(n_normal)

# Fraudulent transactions: different feature distribution
X_fraud = np.random.randn(n_fraud, 28) * 2 + 1  # Different distribution
amounts_fraud = np.abs(np.random.exponential(200, n_fraud))
X_fraud = np.column_stack([X_fraud, amounts_fraud])
y_fraud = np.ones(n_fraud)

X = np.vstack([X_normal, X_fraud])
y = np.hstack([y_normal, y_fraud])

feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
df = pd.DataFrame(X, columns=feature_cols)
df['Class'] = y

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {y.mean():.4%}")

# ──────────────────────────────────────
# STEP 2: DATA CLEANING
# ──────────────────────────────────────
# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# No missing values in this dataset — but always check
# If there were: df.fillna(df.median()) for numeric features

# ──────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────
print("\n=== EDA Summary ===")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# Check amount distribution by class
print(f"\nAvg amount (legitimate): ${df[df.Class==0]['Amount'].mean():.2f}")
print(f"Avg amount (fraud):      ${df[df.Class==1]['Amount'].mean():.2f}")

# ──────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ──────────────────────────────────────
# Log-transform Amount to handle skewed distribution
df['Amount_log'] = np.log1p(df['Amount'])

# Drop original Amount (replaced by log version)
feature_cols_final = [f'V{i}' for i in range(1, 29)] + ['Amount_log']

X = df[feature_cols_final].values
y = df['Class'].values

# ──────────────────────────────────────
# STEP 5: TRAIN / VALIDATION / TEST SPLIT
# ──────────────────────────────────────
# Split off test set first — never touch until final evaluation
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train:      {len(X_train):5d} ({y_train.mean():.2%} fraud)")
print(f"  Validation: {len(X_val):5d} ({y_val.mean():.2%} fraud)")
print(f"  Test:       {len(X_test):5d} ({y_test.mean():.2%} fraud)")

# ──────────────────────────────────────
# STEP 6: SCALING (fit on train ONLY)
# ──────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_val_scaled   = scaler.transform(X_val)         # Transform only (no fit)
X_test_scaled  = scaler.transform(X_test)         # Transform only

# ──────────────────────────────────────
# STEP 7: MODEL TRAINING & COMPARISON
# ──────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_v, y_v):
    """Train and evaluate a model, print key metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_v)
    
    f1  = f1_score(y_v, y_pred)
    pre = precision_score(y_v, y_pred)
    rec = recall_score(y_v, y_pred)
    
    # AUC-ROC uses probabilities if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_v)[:, 1]
        auc = roc_auc_score(y_v, y_prob)
    else:
        auc = None
    
    print(f"\n{'─'*50}")
    print(f"Model: {name}")
    print(f"  Precision: {pre:.4f}  (of all 'fraud' alerts, {pre:.1%} are real)")
    print(f"  Recall:    {rec:.4f}  (caught {rec:.1%} of actual frauds)")
    print(f"  F1-Score:  {f1:.4f}")
    if auc:
        print(f"  AUC-ROC:   {auc:.4f}")
    
    return {'name': name, 'model': model, 'f1': f1, 'precision': pre, 'recall': rec}

results = []

# Baseline: Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
results.append(evaluate_model(
    "Logistic Regression (balanced)", lr, X_train_scaled, y_train, X_val_scaled, y_val
))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
)
results.append(evaluate_model(
    "Random Forest", rf, X_train_scaled, y_train, X_val_scaled, y_val
))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
results.append(evaluate_model(
    "Gradient Boosting", gb, X_train_scaled, y_train, X_val_scaled, y_val
))

# ──────────────────────────────────────
# STEP 8: HYPERPARAMETER TUNING
# ──────────────────────────────────────
# Best model from above: tune it
# Using GridSearchCV on training data only (validation set separate)
print("\n\n=== Hyperparameter Tuning (Random Forest) ===")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 10]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=skf,
    scoring='f1',         # Optimize for F1 on fraud class
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1:      {grid_search.best_score_:.4f}")

# ──────────────────────────────────────
# STEP 9: FINAL EVALUATION ON TEST SET
# ──────────────────────────────────────
print("\n\n" + "="*50)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*50)

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test_scaled)

print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraud']))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
print(f"\n  FP: {cm[0,1]} legitimate transactions falsely flagged")
print(f"  FN: {cm[1,0]} frauds missed")
```

### e. Deployment Considerations

**Batch vs Real-Time**: Fraud detection is real-time — predictions must happen in < 200ms per transaction.

**FastAPI Deployment**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

# Load saved model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

class Transaction(BaseModel):
    features: list[float]  # 29 features (V1-V28 + Amount_log)

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    X = np.array(transaction.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    fraud_prob = model.predict_proba(X_scaled)[0][1]
    is_fraud = fraud_prob > 0.5  # Adjust threshold based on business needs
    
    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_fraud": bool(is_fraud),
        "action": "review" if fraud_prob > 0.3 else "approve"
    }
```

**Scaling Considerations**:
- 10M transactions/day = ~116 TPS (transactions per second) average
- Use model serving infrastructure (TorchServe, Triton, or simple FastAPI + Gunicorn)
- Cache scaler transform; precompute feature vectors where possible
- Monitor for concept drift: fraud patterns change → retrain monthly

---

## Project 2: House Price Prediction

### a. Problem Statement

**Business Context**: A real estate platform wants to provide automated property valuation estimates to sellers listing their homes. Sellers input property details; the model returns an estimated market price within a competitive range.

**ML Task**: Regression. Predict continuous house prices.

**Success Metric**: RMSE (since large errors are particularly bad for seller trust) and R² (relative model quality).

### b. Dataset

**Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Description**:
- 1,460 training examples
- 80 features: lot area, neighborhood, year built, garage type, etc.
- Target: SalePrice (continuous)
- Mix of numeric and categorical features; significant missing values

### c. Full Code Pipeline

```python
# ============================================================
# PROJECT 2: House Price Prediction
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────
# STEP 1: DATA LOADING & INSPECTION
# ──────────────────────────────────────
np.random.seed(42)

# Simulate realistic house price data
n_samples = 1000

data = {
    'GrLivArea': np.random.normal(1500, 400, n_samples).clip(500, 4000),
    'OverallQual': np.random.choice(range(1, 11), n_samples,
                                     p=[0.01, 0.02, 0.05, 0.10, 0.15,
                                        0.20, 0.20, 0.15, 0.08, 0.04]),
    'YearBuilt': np.random.randint(1900, 2020, n_samples),
    'TotalBsmtSF': np.random.normal(1000, 350, n_samples).clip(0, 3000),
    'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples,
                                    p=[0.05, 0.20, 0.50, 0.20, 0.05]),
    'Neighborhood': np.random.choice(
        ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'],
        n_samples, p=[0.2, 0.25, 0.2, 0.2, 0.15]
    ),
    'BldgType': np.random.choice(['1Fam', '2fmCon', 'Duplex'], n_samples,
                                   p=[0.7, 0.2, 0.1]),
}

# Introduce some missing values (realistic)
df = pd.DataFrame(data)
missing_mask = np.random.random(n_samples) < 0.08
df.loc[missing_mask, 'TotalBsmtSF'] = np.nan

# Generate price with realistic coefficients + noise
neighborhood_premium = {'NAmes': 0, 'CollgCr': 20000, 'OldTown': -10000,
                         'Edwards': -5000, 'Somerst': 25000}
bldg_premium = {'1Fam': 15000, '2fmCon': -5000, 'Duplex': -15000}

df['SalePrice'] = (
    50000 +
    80 * df['GrLivArea'] +
    15000 * df['OverallQual'] +
    300 * (df['YearBuilt'] - 1950) +
    50 * df['TotalBsmtSF'].fillna(0) +
    20000 * df['GarageCars'] +
    df['Neighborhood'].map(neighborhood_premium) +
    df['BldgType'].map(bldg_premium) +
    np.random.normal(0, 15000, n_samples)  # noise
).clip(50000, 750000)

print(f"Dataset shape: {df.shape}")
print(f"\nTarget (SalePrice) stats:")
print(df['SalePrice'].describe())

# ──────────────────────────────────────
# STEP 2: DATA CLEANING & EDA
# ──────────────────────────────────────
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Check for potential leakage: no feature should contain price info
# Check for outliers
q1, q99 = df['SalePrice'].quantile([0.01, 0.99])
print(f"\nPrice range (1st–99th percentile): ${q1:,.0f} – ${q99:,.0f}")

# ──────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ──────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features that might improve model performance."""
    df = df.copy()
    
    # Age of house at time of "sale" (assume 2024)
    df['HouseAge'] = 2024 - df['YearBuilt']
    
    # Total square footage
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
    
    # Quality × area interaction (premium quality large homes)
    df['QualArea'] = df['OverallQual'] * df['GrLivArea']
    
    return df

df_engineered = engineer_features(df)

# Define feature types for preprocessing pipeline
numeric_features = ['GrLivArea', 'OverallQual', 'TotalBsmtSF',
                    'GarageCars', 'HouseAge', 'TotalSF', 'QualArea']
categorical_features = ['Neighborhood', 'BldgType']
target = 'SalePrice'

X = df_engineered[numeric_features + categorical_features]
y = df_engineered[target].values

# ──────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT
# ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ──────────────────────────────────────
# STEP 5: PREPROCESSING PIPELINE
# ──────────────────────────────────────
# Numeric: impute missing values → scale
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Median robust to outliers
    ('scaler', StandardScaler())
])

# Categorical: impute → one-hot encode
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into a single preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ──────────────────────────────────────
# STEP 6: MODEL TRAINING & COMPARISON
# ──────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=100),
    'Lasso Regression': Lasso(alpha=1000),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Cross-Validation Results (5-fold) ===")
print(f"{'Model':<22} {'CV RMSE':>10} {'± Std':>8}")
print("-" * 45)

cv_results = {}
for name, model in models.items():
    # Build full pipeline: preprocess → model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Negative MSE because sklearn maximizes, we want to minimize
    neg_mse_scores = cross_val_score(
        pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'
    )
    rmse_scores = np.sqrt(-neg_mse_scores)
    cv_results[name] = {'mean': rmse_scores.mean(), 'std': rmse_scores.std(),
                         'pipeline': pipeline}
    
    print(f"{name:<22} ${rmse_scores.mean():>9,.0f} ± ${rmse_scores.std():>6,.0f}")

# ──────────────────────────────────────
# STEP 7: HYPERPARAMETER TUNING
# ──────────────────────────────────────
from sklearn.model_selection import GridSearchCV

print("\n\n=== Hyperparameter Tuning (Gradient Boosting) ===")

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(
    gb_pipeline, param_grid, cv=5,
    scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rmse = np.sqrt(-grid_search.best_score_)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV RMSE:    ${best_rmse:,.0f}")

# ──────────────────────────────────────
# STEP 8: FINAL EVALUATION ON TEST SET
# ──────────────────────────────────────
print("\n\n" + "="*55)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*55)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

test_mae  = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2   = r2_score(y_test, y_pred_test)

print(f"MAE:  ${test_mae:>10,.0f}  (predictions off by this much on average)")
print(f"RMSE: ${test_rmse:>10,.0f}  (penalizes large errors more)")
print(f"R²:   {test_r2:>11.4f}  (explains this fraction of price variance)")

# Error analysis: where does the model fail?
errors = y_test - y_pred_test
print(f"\nError Distribution:")
print(f"  Max overestimate:  ${errors.max():>10,.0f}")
print(f"  Max underestimate: ${errors.min():>10,.0f}")
print(f"  Within ±$20K:      {(np.abs(errors) < 20000).mean():.1%} of predictions")
```

### d. Deployment Considerations

**Deployment Mode**: Batch + On-Demand API

```python
# FastAPI endpoint for on-demand valuation
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Estimator")
model_pipeline = joblib.load("house_price_pipeline.pkl")

class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int
    YearBuilt: int
    TotalBsmtSF: float | None = None
    GarageCars: int
    Neighborhood: str
    BldgType: str

@app.post("/estimate")
def estimate_price(house: HouseFeatures):
    # Convert to DataFrame (pipeline expects this format)
    data = pd.DataFrame([house.dict()])
    
    # Engineer features (same as training)
    data['HouseAge'] = 2024 - data['YearBuilt']
    data['TotalSF']  = data['GrLivArea'] + data['TotalBsmtSF'].fillna(0)
    data['QualArea'] = data['OverallQual'] * data['GrLivArea']
    
    price_estimate = model_pipeline.predict(data)[0]
    
    # Return estimate with confidence range (simple heuristic: ±10%)
    return {
        "estimated_price": round(price_estimate, -3),  # Round to nearest $1000
        "range_low":  round(price_estimate * 0.90, -3),
        "range_high": round(price_estimate * 1.10, -3),
    }
```

**Retraining Strategy**: Monthly retrain on new closed sales data. Property markets shift seasonally — stale models undervalue in rising markets.

---

# 13. Algorithm Comparison Tables

## 13.1 Overall Algorithm Comparison

| Algorithm | Performance | Interpretability | Training Speed | Inference Speed | Handles Imbalance | Requires Scaling |
|---|---|---|---|---|---|---|
| Logistic Regression | Medium | High | Fast | Very Fast | With class_weight | Yes |
| Linear Regression | Medium | High | Fast | Very Fast | N/A | Yes |
| Ridge/Lasso | Medium | High | Fast | Very Fast | N/A | Yes |
| Decision Tree | Medium | High | Fast | Fast | With class_weight | No |
| Random Forest | High | Medium | Medium | Medium | With class_weight | No |
| Gradient Boosting | Very High | Low-Medium | Slow | Fast | With scale_pos_weight | No |
| SVM (RBF) | High | Low | Slow | Medium | With class_weight | Yes |
| K-Means | Medium | Medium | Fast | Fast | N/A | Yes |
| Isolation Forest | High (anomaly) | Low | Fast | Fast | Designed for it | No |

## 13.2 When to Use Each

| Situation | Recommended Start |
|---|---|
| Tabular data, need interpretability | Logistic/Linear Regression |
| Tabular data, maximize performance | Gradient Boosting (XGBoost/LightGBM) |
| Many features, regularization needed | Ridge/Lasso |
| High-dimensional sparse data (NLP) | Logistic Regression with L2 |
| Customer segmentation | K-Means |
| Fraud/anomaly detection | Isolation Forest |
| Small dataset | SVM |
| Large dataset, fast training needed | Random Forest or LightGBM |

## 13.3 Regression Algorithms

| Algorithm | Best For | Watch Out For |
|---|---|---|
| Linear Regression | Quick baseline, interpretability | Assumes linear relationships |
| Ridge | Multicollinearity, many features | Need to tune alpha |
| Lasso | Feature selection embedded | Can be unstable with correlated features |
| Random Forest | Non-linear patterns, robustness | Can overfit on noisy small datasets |
| Gradient Boosting | Winning Kaggle competitions | Slow training, many hyperparameters |

## 13.4 Classification Algorithms

| Algorithm | Best For | Watch Out For |
|---|---|---|
| Logistic Regression | Baseline, probabilistic output | Assumes linear boundary |
| Random Forest | Robust, handles mixed features | Memory intensive for large n_estimators |
| Gradient Boosting | Highest accuracy on tabular data | Prone to overfitting on small datasets |
| SVM | High-dimensional data | Slow on large datasets |
| K-Nearest Neighbors | Simple, non-parametric | Slow inference, no feature importance |

---

# 14. Common Mistakes & Pitfalls Reference

## 14.1 The Top 10 ML Mistakes

| # | Mistake | Consequence | Prevention |
|---|---|---|---|
| 1 | **Data leakage** | Inflated metrics, production failure | Strict split discipline, pipeline objects |
| 2 | **Evaluating on training data** | Think model works, it doesn't | Always separate train/eval |
| 3 | **Skipping baseline** | Complex model, trivial improvement | Always benchmark against simple rules |
| 4 | **Using accuracy on imbalanced data** | Miss the real problem | Use F1, AUC, precision@recall |
| 5 | **Scaling before splitting** | Test set statistics contaminate scaler | Split first, scale second |
| 6 | **Not analyzing errors** | Never understand failure modes | Manual review of wrong predictions |
| 7 | **Hypertuning on test set** | Overfitting to test → no true estimate | Use CV for tuning; test set once only |
| 8 | **Wrong cross-validation for time series** | Future data trains past model | Use TimeSeriesSplit, not KFold |
| 9 | **Ignoring class imbalance** | Model predicts majority class only | class_weight, resampling, threshold tuning |
| 10 | **Over-engineering features before EDA** | Build features for patterns that don't exist | EDA first, features second |

## 14.2 Data Leakage Checklist

Before finalizing any model, audit for leakage:

```
□ Were all transformers (scalers, encoders) fitted on training data only?
□ Does any feature contain information from after the prediction time?
□ Was feature selection done before splitting?
□ Was imputation done before splitting?
□ Are there any "future" variables in the feature set?
□ For time-series: is the split temporal, not random?
□ Were any "post-event" features included? (e.g., "treatment outcome" for treatment prediction)
```

---

# 15. Interview Preparation

## 15.1 Conceptual Questions

**Q1: What is the difference between supervised and unsupervised learning?**
> Supervised learning uses labeled data — the model learns to map inputs to known outputs. Unsupervised learning has no labels; the model discovers structure (clusters, patterns, distributions) in the data independently.

**Q2: Explain the bias–variance tradeoff.**
> Bias is error from overly simplistic model assumptions (underfitting). Variance is error from model sensitivity to training data fluctuations (overfitting). Increasing model complexity reduces bias but increases variance. The optimal model minimizes total error — the sum of both.

**Q3: Why would you use log loss instead of accuracy as your training objective?**
> Accuracy is non-differentiable (a tiny weight change either flips a prediction or doesn't — no gradient exists). Log loss is smooth and differentiable everywhere, allowing gradient-based optimization. Additionally, log loss captures probability calibration — it rewards confidence in the right direction.

**Q4: When is cross-validation preferred over a simple train/validation split?**
> When the dataset is small (< 10K samples), a single split is unreliable — performance estimates depend heavily on which random examples ended up in the validation set. Cross-validation averages over multiple splits, giving a more stable and reliable estimate.

**Q5: What is data leakage and why is it dangerous?**
> Leakage occurs when information from the test set or from the future contaminates the training process. It makes models appear better than they are, and the performance gap only reveals itself in production — sometimes catastrophically.

**Q6: Why should you always build a baseline model?**
> A baseline establishes the minimum bar: "what can be achieved without any ML?" If your baseline is 90% accuracy (from always predicting the majority class), and your fancy model gets 91%, you've spent enormous effort for 1% improvement that may not be worth the operational complexity.

**Q7: What's the difference between a parameter and a hyperparameter?**
> Parameters are learned automatically from data during training (e.g., model weights). Hyperparameters are set by the engineer before training begins and control how learning happens (e.g., learning rate, number of trees). Parameters are in the model; hyperparameters configure the training process.

## 15.2 Practical Scenario-Based Questions

**Scenario 1**: Your model achieves 99% accuracy but your stakeholder is unhappy. Why might this happen?
> Almost certainly an imbalanced dataset. If 99% of examples belong to one class, predicting that class always gives 99% accuracy while being completely useless for detecting the minority class (fraud, disease, etc.). Always report precision, recall, and F1 for classification problems with class imbalance.

**Scenario 2**: Your model performs perfectly in validation but fails in production. What could cause this?
> Several candidates: (1) data leakage — the validation was contaminated, (2) distribution shift — production data differs from training data, (3) temporal leakage — trained on future data accidentally, (4) different preprocessing in production vs. training, (5) serving stale feature values.

**Scenario 3**: How would you handle a dataset where 95% of labels are class 0?
> Options: (a) Use class_weight='balanced' in sklearn models, (b) Oversample the minority class (SMOTE), (c) Undersample the majority class, (d) Adjust decision threshold post-training, (e) Use precision/recall/F1 metrics instead of accuracy. The right choice depends on whether false positives or false negatives are more costly.

**Scenario 4**: You have a regression model with R² = 0.95 on training but 0.45 on validation. What do you do?
> Classic overfitting. Actions: (1) Investigate for data leakage first, (2) Add regularization (Ridge/Lasso for linear models, reduce max_depth/n_estimators for trees), (3) Collect more training data, (4) Reduce number of features, (5) Apply cross-validation to get a stable estimate.

**Scenario 5**: A colleague argues you should tune hyperparameters on the test set to maximize performance. What's wrong with this?
> This defeats the purpose of the test set. The test set is supposed to estimate how the model will perform on data it has never seen. If you tune on it, you're fitting to that specific set — and your "test performance" is now an optimistic estimate. In production, performance will be lower.

**Scenario 6**: You're building a cancer screening model. Should you optimize for precision or recall?
> Recall — catching every actual cancer case is paramount. A false negative (missed cancer) is life-threatening. A false positive (unnecessary follow-up test) is costly and uncomfortable but not fatal. This is a textbook case for high-recall optimization, potentially at the expense of precision.

## 15.3 Quick-Fire Conceptual Checks

| Question | Answer |
|---|---|
| What does a loss function measure? | How wrong predictions are; minimized during training |
| Name two regression metrics | MAE, RMSE (or MSE, R²) |
| What's the harmonic mean of precision and recall? | F1-score |
| Why can't you use accuracy on imbalanced data? | Majority class always predicted looks accurate |
| What is the test set for? | Final evaluation only — never for tuning |
| How does cross-validation prevent overfitting to validation data? | Averages over multiple splits; no single split is "the target" |
| What does high variance look like on a learning curve? | Low train error, high val error; large gap between them |

---

# 16. Resources

## 16.1 Essential Reading

### Books
- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" — Aurélien Géron**
  The best practical ML engineering book. Chapter 2 (end-to-end project) and Chapter 4 (training models) are essential.
- **"Pattern Recognition and Machine Learning" — Christopher Bishop**
  Rigorous mathematical foundation. Chapter 1 (introduction) is readable for all levels.
- **"The Elements of Statistical Learning" — Hastie, Tibshirani, Friedman**
  Free PDF available. The statistical bible of ML.

### Online Courses
- **fast.ai Practical Deep Learning** — fastai.com — Top-down, practical, free
- **Stanford CS229 Machine Learning** — cs229.stanford.edu — Andrew Ng's course, lecture notes available
- **Google Machine Learning Crash Course** — developers.google.com/machine-learning/crash-course

## 16.2 Key Documentation

- **scikit-learn User Guide**: scikit-learn.org/stable/user_guide.html
  - [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
  - [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
  - [Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- **pandas Documentation**: pandas.pydata.org/docs
- **numpy Documentation**: numpy.org/doc

## 16.3 High-Quality Blog Posts

- **"A Few Useful Things to Know about Machine Learning" — Pedro Domingos**
  [PDF](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — Essential practitioner wisdom
- **"Bias-Variance Tradeoff" — Scott Fortmann-Roe**
  Understanding the tradeoff with excellent visual explanations
- **"Rules of Machine Learning" — Google**
  [link](https://developers.google.com/machine-learning/guides/rules-of-ml)
  43 practical rules from Google's production ML experience
- **"No Free Lunch Theorem" — Wolpert & Macready**
  Why no algorithm dominates all problems

## 16.4 Papers Worth Reading

| Paper | Why Read It |
|---|---|
| "A Unified Approach to Interpreting Model Predictions" (SHAP) — Lundberg & Lee | Model interpretability |
| "XGBoost: A Scalable Tree Boosting System" — Chen & Guestrin | How gradient boosting actually works |
| "Random Forests" — Breiman | Original random forest paper, readable |
| "Practical Recommendations for Gradient-Based Training of Deep Architectures" — Bengio | Hyperparameter intuition |

## 16.5 Tools & Libraries

| Tool | Use |
|---|---|
| scikit-learn | All-purpose ML in Python |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib/seaborn | Visualization |
| XGBoost | Gradient boosting (competition standard) |
| LightGBM | Fast gradient boosting for large datasets |
| SHAP | Model interpretability |
| Optuna | Hyperparameter optimization |
| MLflow | Experiment tracking |
| FastAPI | Model serving |

---

*This document was built to serve as a living reference. Revisit the projects section after completing your first real ML project — the concepts will land much deeper the second time.*

*Last reviewed: 2025. Core ML fundamentals are stable — concepts here apply equally to 2010-era sklearn and 2025-era LLM fine-tuning pipelines.*
