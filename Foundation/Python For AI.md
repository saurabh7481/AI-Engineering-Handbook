# Foundations for AI Engineering
## A Deep, Practical, AI-Focused Python Reference for Aspiring ML Engineers

> *"You don't need to memorize everything. You need to understand why it matters, and know where to look."*
> — Every great senior engineer ever

---

## Overview

This document is not a Python tutorial. It is a **structured engineering reference** designed for one purpose: making you production-ready as an AI/ML Engineer.

Every concept here is taught through the lens of machine learning. Every code snippet reflects real patterns used in ML pipelines, data systems, and model training workflows. If a topic doesn't connect to AI/ML, it is not in here.

### Why These Topics Matter for AI/ML

| Foundation Area | Why It Matters in ML |
|---|---|
| Python Core | Every ML framework (PyTorch, TensorFlow, Sklearn) is Python-first |
| Data Structures | Feature stores, batch processing, and data loaders rely on these |
| OOP | Every ML library is object-oriented; custom models/datasets require it |
| NumPy | The mathematical backbone of all ML computations |
| Pandas | Data ingestion, cleaning, feature engineering |
| Error Handling | Production ML pipelines fail silently without this |
| Async/Parallel | Real-time inference and data loading require concurrency |
| FastAPI/Pydantic | Model serving and input validation in production |

### How These Connect to Real ML Systems

```
Raw Data (SQL/Files)
       ↓
Data Ingestion & Cleaning (Pandas, Python I/O)
       ↓
Feature Engineering (NumPy, Pandas, Python Logic)
       ↓
Model Training (PyTorch/Sklearn — built on Python OOP + NumPy)
       ↓
Evaluation & Monitoring (Statistics, Logging)
       ↓
Deployment (FastAPI, Pydantic, Async Python)
```

Every section of this document maps to one or more steps in this flow.

---

## Table of Contents

1. [Python Core Fundamentals](#1-python-core-fundamentals)
2. [Built-in Data Structures](#2-built-in-data-structures)
3. [Pythonic Thinking](#3-pythonic-thinking)
4. [Object-Oriented Programming](#4-object-oriented-programming-oop)
5. [Error Handling & Debugging](#5-error-handling--debugging)
6. [Modules, Packages & Environments](#6-modules-packages--environments)
7. [File Handling & Serialization](#7-file-handling--serialization)
8. [Numerical Computing with NumPy](#8-numerical-computing-with-numpy)
9. [Data Analysis with Pandas](#9-data-analysis-with-pandas)
10. [Data Visualization](#10-data-visualization)
11. [Standard Library High-ROI Parts](#11-standard-library-high-roi-parts)
12. [Asynchronous & Parallel Python](#12-asynchronous--parallel-python)
13. [FastAPI & Pydantic](#13-fastapi--pydantic)
14. [Production-Ready Python Mindset](#14-production-ready-python-mindset)
15. [Cross-Topic Connections](#15-cross-topic-connections)
16. [End-to-End Practical System View](#16-end-to-end-practical-system-view)
17. [Hands-On Projects](#17-hands-on-projects)
18. [Cheat Sheets](#18-cheat-sheets)
19. [Interview Preparation](#19-interview-preparation)
20. [Resources](#20-resources)

---

## 1. Python Core Fundamentals

### a. Why This Matters for AI Engineering

Python is the lingua franca of AI. PyTorch, TensorFlow, Scikit-learn, HuggingFace — all Python. But more importantly, **the way you write Python determines whether your ML code is maintainable, debuggable, and scalable**. Poorly structured Python in an ML pipeline creates bugs that are nearly impossible to trace.

PEP8 and clean syntax aren't aesthetic preferences — they are professional requirements when you're collaborating on a codebase that trains models on millions of records.

### b. Intuition (AI-Focused)

Think of Python syntax rules as the grammar of your ML experiments. A misplaced indentation in a training loop doesn't just cause a syntax error — it can silently change the logic of your gradient updates. Control flow (`if/else`, loops) defines your data processing logic, hyperparameter search, and early stopping. Functions are your reusable pipeline steps.

### c. Minimal Theory (Only What Matters)

**Indentation**: Python uses whitespace for block structure. 4 spaces per level (PEP8). Critical in loops and conditionals inside training code.

**Type system**: Python is dynamically typed but ML code increasingly uses **type hints** for clarity and tooling support.

**Scope rules (LEGB)**: Local → Enclosing → Global → Built-in. Matters when you have training loops that modify model state.

### d. Practical Usage in ML

- `if/else` → conditional data preprocessing, early stopping checks
- `for` loops → iterating over batches, epochs, hyperparameter grids
- `while` loops → streaming data ingestion, retry logic in data fetching
- Functions → reusable pipeline steps (load_data, preprocess, train, evaluate)
- Lambda → quick transformations in `df.apply()`, `map()`

### e. Python Implementation

```python
# PEP8-compliant ML pipeline function
from typing import Optional
import numpy as np


def preprocess_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.

    Args:
        X: Input feature matrix of shape (n_samples, n_features)
        mean: Precomputed mean (use training mean at inference time)
        std: Precomputed std (use training std at inference time)

    Returns:
        Tuple of (normalized_X, mean, std)
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero

    return (X - mean) / std, mean, std


# Control flow in training loop
def train_model(model, data_loader, epochs: int = 10, patience: int = 3):
    """Training loop with early stopping."""
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss

        avg_loss = epoch_loss / len(data_loader)

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save_checkpoint()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")


# Lambda in ML context — quick feature transformation
import pandas as pd

df = pd.DataFrame({"age": [25, 35, 45, 55], "income": [30000, 60000, 90000, 120000]})

# Bucketing age into groups using lambda
df["age_group"] = df["age"].apply(lambda x: "young" if x < 35 else "mid" if x < 50 else "senior")

# List comprehension for feature names
feature_names = [f"feature_{i}" for i in range(10)]
squared_features = [f"{name}_squared" for name in feature_names]
```

### f. Mini Use Case

```python
# Real scenario: Building a hyperparameter grid search from scratch
def grid_search(model_class, param_grid: dict, X_train, y_train, X_val, y_val):
    """
    Exhaustive grid search over hyperparameters.
    Returns best params and their validation score.
    """
    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = float("-inf")
    best_params = {}

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = model_class(**params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# Usage
param_grid = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "hidden_size": [64, 128, 256],
    "dropout": [0.1, 0.3, 0.5],
}
```

### g. Common Mistakes

- **Mutable default arguments**: `def fn(data=[])` — the list persists across calls. Use `None` as default.
- **Off-by-one in epoch loops**: `range(epochs)` gives 0 to epochs-1. Always verify.
- **Ignoring return values**: `df.dropna()` returns a new DataFrame by default. Not `inplace`. Many bugs start here.
- **Using `print` instead of `logging`**: In production ML, `print` disappears. `logging` gives you timestamps, levels, and file output.
- **Not using type hints**: Untyped ML code is a debugging nightmare when shapes are wrong.

---

## 2. Built-in Data Structures

### a. Why This Matters for AI Engineering

ML pipelines are fundamentally data transformation systems. Every batch of training data passes through Python data structures before reaching your model. Understanding the performance characteristics of lists vs. dicts vs. sets directly impacts how fast your data pipeline runs.

Feature stores, label encoders, vocabulary lookups, batch samplers — all use these structures under the hood.

### b. Intuition (AI-Focused)

| Structure | ML Usage |
|---|---|
| List | Ordered batches, sequences, token lists |
| Tuple | Immutable configs, dataset splits (train, val, test) |
| Set | Vocabulary deduplication, categorical unique values |
| Dict | Feature maps, label encoders, model configs, metric tracking |
| String | Text data, tokenization, feature names |

### c. Minimal Theory

**Time complexities that matter in ML:**

| Operation | List | Dict | Set |
|---|---|---|---|
| Lookup by index | O(1) | — | — |
| Lookup by key/value | O(n) | O(1) | O(1) |
| Insert | O(1) amortized | O(1) | O(1) |
| Membership test (`in`) | O(n) | O(1) | O(1) |

This matters when you have a vocabulary of 100,000 tokens. `word in vocab_set` is O(1). `word in vocab_list` is O(n). At scale, this is the difference between seconds and hours.

### d. Practical Usage in ML

```python
# Vocabulary building — sets prevent duplicates automatically
corpus = ["the cat sat", "the dog ran", "cat and dog"]
vocab = set()
for sentence in corpus:
    vocab.update(sentence.split())
# vocab = {'the', 'cat', 'sat', 'dog', 'ran', 'and'}
vocab = sorted(vocab)  # Sort for reproducibility
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Dicts for tracking metrics across epochs
metrics_history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
}

for epoch in range(10):
    # ... training ...
    metrics_history["train_loss"].append(0.45)
    metrics_history["val_loss"].append(0.52)
    metrics_history["val_accuracy"].append(0.87)

# Tuples for immutable dataset splits — prevents accidental modification
dataset_split = (X_train, X_val, X_test, y_train, y_val, y_test)
```

### e. Python Implementation

```python
import numpy as np
from collections import defaultdict, Counter

# --- Lists in ML: Batch management ---
class SimpleBatchLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = list(range(len(X)))

    def __iter__(self):
        np.random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


# --- Dicts in ML: Label encoding ---
class LabelEncoder:
    def __init__(self):
        self.label2idx: dict[str, int] = {}
        self.idx2label: dict[int, str] = {}

    def fit(self, labels: list[str]) -> "LabelEncoder":
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        return self

    def transform(self, labels: list[str]) -> list[int]:
        return [self.label2idx[label] for label in labels]

    def inverse_transform(self, indices: list[int]) -> list[str]:
        return [self.idx2label[idx] for idx in indices]


# Usage
encoder = LabelEncoder()
labels = ["cat", "dog", "cat", "bird", "dog"]
encoder.fit(labels)
encoded = encoder.transform(labels)  # [1, 2, 1, 0, 2]

# --- Strings in ML: Tokenization ---
def simple_tokenizer(text: str, vocab: dict[str, int], max_len: int = 128) -> list[int]:
    """Convert text to token IDs, truncate/pad to max_len."""
    tokens = text.lower().split()
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    # Truncate
    ids = ids[:max_len]
    # Pad
    ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    return ids


# --- Counter for class imbalance detection ---
from collections import Counter

y_train = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # Imbalanced
class_counts = Counter(y_train)
# Counter({0: 7, 1: 3})

# Calculate class weights for imbalanced training
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count)
                 for cls, count in class_counts.items()}
# {0: 0.714, 1: 1.667}
```

### f. Mini Use Case

```python
# Feature store using nested dicts — common pattern in production
# Stores precomputed features keyed by entity ID

feature_store: dict[str, dict[str, float]] = {}

def compute_user_features(user_id: str, transactions: list[dict]) -> dict[str, float]:
    amounts = [t["amount"] for t in transactions]
    return {
        "avg_transaction": sum(amounts) / len(amounts) if amounts else 0.0,
        "total_spend": sum(amounts),
        "transaction_count": float(len(amounts)),
        "max_transaction": max(amounts) if amounts else 0.0,
    }

# Populate feature store
for user_id, txns in raw_data.items():
    feature_store[user_id] = compute_user_features(user_id, txns)

# Fast O(1) lookup at inference time
user_features = feature_store.get("user_123", {})
```

### g. Common Mistakes

- **Using a list for membership checks**: `if item in my_list` inside a loop of 100K items = O(n²). Use a set.
- **Dict mutation during iteration**: `for k in d: del d[k]` raises `RuntimeError`. Use `list(d.keys())` first.
- **Confusing shallow equality**: `[1,2] == [1,2]` is `True`, but they are different objects. Matters with mutable defaults.
- **Forgetting tuples are hashable, lists are not**: Only hashable types can be dict keys or set members. Use tuples for composite keys.

---

## 3. Pythonic Thinking

### a. Why This Matters for AI Engineering

NumPy, PyTorch DataLoaders, and Scikit-learn all expose Python iterators. HuggingFace datasets are generator-based. Understanding generators means you can process datasets larger than RAM — which is the norm in production ML. Pythonic idioms also make your code dramatically faster because they use optimized C-level implementations internally.

### b. Intuition (AI-Focused)

A **generator** is a lazy sequence — it produces values one at a time instead of loading everything into memory. This is how you stream a dataset of 100M samples through a training loop without running out of RAM.

`zip`, `enumerate`, `map`, `filter` are the tools you use to avoid explicit loops — which are slow in Python. Always prefer these for data transformation.

### c. Minimal Theory

**Generator protocol**: Any object implementing `__iter__` and `__next__`. `yield` creates a generator function.

**Memory**: A list of 1M floats ≈ 8MB. A generator producing those floats ≈ 0 bytes until consumed.

**Copy semantics**:
- Shallow copy: copies the container, not the contents. `list.copy()`, `dict.copy()`, `[:]`
- Deep copy: copies everything recursively. `copy.deepcopy()`

### d. Practical Usage in ML

```python
import copy
import numpy as np
from functools import reduce


# --- Generators for large dataset streaming ---
def data_generator(file_paths: list[str], batch_size: int = 32):
    """
    Stream data from files in batches.
    Memory-efficient for datasets that don't fit in RAM.
    """
    batch = []
    for path in file_paths:
        with open(path) as f:
            for line in f:
                sample = parse_line(line)
                batch.append(sample)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
    if batch:  # Yield remaining samples
        yield batch


# --- zip in ML: pairing predictions with ground truth ---
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0]
errors = [(true, pred) for true, pred in zip(y_true, y_pred) if true != pred]


# --- enumerate for indexed batch processing ---
def train_epoch(model, data_loader, log_every: int = 100):
    total_loss = 0
    for batch_idx, (X, y) in enumerate(data_loader):
        loss = model.train_step(X, y)
        total_loss += loss
        if batch_idx % log_every == 0:
            print(f"  Batch {batch_idx:04d} | Loss: {loss:.4f}")
    return total_loss / len(data_loader)


# --- map/filter for data preprocessing pipelines ---
raw_texts = ["  Hello World  ", "", "Machine Learning", "   ", "AI is great"]

# Filter empty strings, then strip whitespace
clean_texts = list(map(str.strip, filter(lambda x: x.strip(), raw_texts)))

# --- *args and **kwargs for flexible model wrappers ---
def run_experiment(model_class, *args, **kwargs):
    """
    Generic experiment runner.
    Forwards args/kwargs to model constructor.
    """
    model = model_class(*args, **kwargs)
    model.fit(X_train, y_train)
    return model.evaluate(X_val, y_val)


# --- Shallow vs deep copy in ML: the danger ---
# Shallow copy — nested objects still shared
config = {"model": {"layers": [64, 128, 256], "dropout": 0.3}}
config_copy = config.copy()
config_copy["model"]["layers"].append(512)
print(config["model"]["layers"])  # [64, 128, 256, 512] — MODIFIED! Bug!

# Deep copy — fully independent
config_copy = copy.deepcopy(config)
config_copy["model"]["layers"].append(512)
print(config["model"]["layers"])  # [64, 128, 256] — Safe
```

### e. Itertools for ML Pipelines

```python
from itertools import islice, chain, cycle
import itertools


# islice: Take first N batches from a generator (useful for debugging)
gen = data_generator(file_paths, batch_size=32)
first_5_batches = list(islice(gen, 5))


# chain: Combine multiple data sources
train_gen = data_generator(train_files)
augmented_gen = data_generator(augmented_files)
combined_gen = chain(train_gen, augmented_gen)


# product: Hyperparameter grid
from itertools import product

param_grid = {
    "lr": [0.01, 0.001],
    "batch_size": [32, 64],
    "hidden": [128, 256],
}
configs = [
    dict(zip(param_grid.keys(), combo))
    for combo in product(*param_grid.values())
]
# 8 configs: all combinations of lr × batch_size × hidden
```

### f. Mini Use Case

```python
# Generator-based tokenized dataset — memory-efficient NLP pipeline
class TextDataset:
    def __init__(self, texts: list[str], vocab: dict, max_len: int = 512):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

    def __iter__(self):
        """Lazy tokenization — one sample at a time."""
        for text in self.texts:
            tokens = self._tokenize(text)
            yield tokens

    def _tokenize(self, text: str) -> list[int]:
        words = text.lower().split()[:self.max_len]
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        return ids

    def batches(self, batch_size: int = 32):
        """Yield batches as numpy arrays."""
        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) == batch_size:
                yield np.array(batch)
                batch = []
        if batch:
            yield np.array(batch)
```

### g. Common Mistakes

- **Exhausting generators**: Once consumed, a generator is empty. This is a common bug when you iterate a generator twice expecting the same data.
- **Deep copy in training loops**: Using `copy.deepcopy()` inside a training loop is very slow. Deep copy model parameters once, not every step.
- **`map()` returns an iterator in Python 3**: `map(fn, lst)` gives you a map object, not a list. Wrap in `list()` if you need random access.
- **`reduce` without a default**: `reduce(fn, [])` raises `TypeError`. Always provide a default: `reduce(fn, lst, 0)`.

---

## 4. Object-Oriented Programming (OOP)

### a. Why This Matters for AI Engineering

Every ML framework is built on OOP. In PyTorch, you subclass `nn.Module` to define models. In Scikit-learn, you subclass `BaseEstimator` for custom transformers. HuggingFace `Trainer`, PyTorch `Dataset`, `DataLoader` — all classes. If you don't understand OOP, you cannot write custom models, custom losses, or custom data pipelines.

### b. Intuition (AI-Focused)

A neural network layer is a class. A dataset is a class. A training loop configuration is a class. OOP gives you:
- **Encapsulation**: Your model's weights are hidden inside the object.
- **Inheritance**: A `ResNet` inherits from `nn.Module`.
- **Polymorphism**: Any model that implements `fit()`/`predict()` can be swapped in the same pipeline.

### c. Minimal Theory

**`__init__`**: Constructor — initializes the object's state (weights, config, etc.)

**Dunder methods (most important for ML)**:
- `__len__`: `len(dataset)` works
- `__getitem__`: `dataset[i]` works — enables DataLoader indexing
- `__iter__`: `for sample in dataset` works
- `__repr__`: Readable model description
- `__call__`: `model(X)` works (PyTorch uses this)

**`@property`**: Computed attributes. Useful for derived configs (e.g., `output_size = input_size * 2`).

**`@dataclass`**: Auto-generates `__init__`, `__repr__`, `__eq__`. Use for config objects.

### d. Practical Usage in ML

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# --- Dataclass for ML configs (cleaner than dicts) ---
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    hidden_size: int = 256
    dropout: float = 0.3
    seed: int = 42
    model_name: str = "baseline"
    features: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate config after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


config = TrainingConfig(learning_rate=1e-4, batch_size=64)
print(config)  # TrainingConfig(learning_rate=0.0001, batch_size=64, ...)


# --- Custom Dataset class (PyTorch-style) ---
class TabulaDataset:
    """
    Dataset for tabular ML data.
    Supports DataLoader-style iteration.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "Features and labels must have same length"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return (
            f"TabulaDataset("
            f"n_samples={len(self)}, "
            f"n_features={self.X.shape[1]}, "
            f"label_distribution={np.bincount(self.y.astype(int)).tolist()})"
        )


# --- Base class for ML transformers (Sklearn-style) ---
class BaseTransformer:
    """Abstract base for all feature transformers."""

    def fit(self, X: np.ndarray) -> "BaseTransformer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScaler(BaseTransformer):
    """Z-score normalization."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform()")
        return (X - self.mean_) / self.std_

    def __repr__(self) -> str:
        fitted = self.mean_ is not None
        return f"StandardScaler(fitted={fitted})"


class MinMaxScaler(BaseTransformer):
    """Scale features to [0, 1] range."""

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_


# Polymorphism in action — any scaler works in the same pipeline
def build_pipeline(scaler: BaseTransformer, X_train, X_val):
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled
```

### f. Mini Use Case

```python
# Custom pipeline class with method chaining
class FeaturePipeline:
    """
    Chainable feature engineering pipeline.
    Example usage:
        pipeline = (FeaturePipeline(df)
                    .drop_high_null_cols(threshold=0.5)
                    .encode_categoricals()
                    .scale_numerics()
                    .get_features())
    """

    def __init__(self, df):
        import pandas as pd
        self.df = df.copy()
        self._scalers = {}
        self._encoders = {}

    def drop_high_null_cols(self, threshold: float = 0.5) -> "FeaturePipeline":
        null_fractions = self.df.isnull().mean()
        cols_to_keep = null_fractions[null_fractions < threshold].index
        self.df = self.df[cols_to_keep]
        return self

    def encode_categoricals(self) -> "FeaturePipeline":
        import pandas as pd
        cat_cols = self.df.select_dtypes(include="object").columns
        for col in cat_cols:
            self.df[col] = self.df[col].astype("category").cat.codes
        return self

    def scale_numerics(self) -> "FeaturePipeline":
        num_cols = self.df.select_dtypes(include="number").columns
        for col in num_cols:
            col_data = self.df[col].values
            mean, std = col_data.mean(), col_data.std()
            std = std if std > 0 else 1.0
            self.df[col] = (col_data - mean) / std
            self._scalers[col] = (mean, std)
        return self

    def get_features(self):
        return self.df.values
```

### g. Common Mistakes

- **Not calling `super().__init__()`** in subclasses: Critical in PyTorch (`nn.Module`) — skipping it breaks parameter registration.
- **Using class variables for mutable state**: `class Foo: data = []` — all instances share the same list. Use `self.data = []` in `__init__`.
- **Forgetting `self`**: Methods without `self` are static. They can't access instance state.
- **Overusing inheritance**: Prefer composition. A model that *has* a scaler is better than a model that *is* a scaler.
- **Not implementing `__repr__`**: In ML, you need to quickly inspect what your objects contain. Always implement it.

---

## 5. Error Handling & Debugging

### a. Why This Matters for AI Engineering

ML pipelines run for hours or days. A missing file, a shape mismatch, a NaN in the data — any of these can crash a 12-hour training run. Proper error handling means you **catch problems early, log them, and recover gracefully**. In production inference, unhandled exceptions mean your model serving API crashes.

### b. Intuition (AI-Focused)

Think of `try/except` as a safety net around your data pipeline. The goal is not just to prevent crashes — it's to provide **useful error messages** that tell you exactly what went wrong, where, and with what data.

### c. Practical Usage in ML

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Custom exceptions for ML systems ---
class DataPipelineError(Exception):
    """Base exception for data pipeline failures."""
    pass


class ShapeMismatchError(DataPipelineError):
    """Raised when tensor/array shapes are incompatible."""
    def __init__(self, expected, got, context=""):
        message = f"Shape mismatch: expected {expected}, got {got}"
        if context:
            message += f" | Context: {context}"
        super().__init__(message)
        self.expected = expected
        self.got = got


class DataLoadError(DataPipelineError):
    """Raised when data cannot be loaded."""
    pass


# --- Robust data loading ---
def load_dataset(file_path: str) -> np.ndarray:
    """Load dataset with detailed error reporting."""
    path = Path(file_path)

    if not path.exists():
        raise DataLoadError(f"Dataset file not found: {path.absolute()}")

    if path.suffix not in {".csv", ".npy", ".parquet"}:
        raise DataLoadError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported: .csv, .npy, .parquet"
        )

    try:
        if path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix == ".csv":
            import pandas as pd
            data = pd.read_csv(path).values
        elif path.suffix == ".parquet":
            import pandas as pd
            data = pd.read_parquet(path).values

        logger.info(f"Loaded dataset: shape={data.shape}, dtype={data.dtype}")
        return data

    except MemoryError:
        raise DataLoadError(
            f"File {path.name} too large to load. "
            f"Consider chunked loading with pandas."
        )
    except Exception as e:
        raise DataLoadError(f"Failed to load {path.name}: {e}") from e


# --- Shape validation in pipelines ---
def validate_shapes(X: np.ndarray, y: np.ndarray, n_features: int):
    """Validate data shapes before training."""
    if X.ndim != 2:
        raise ShapeMismatchError(
            expected="(n_samples, n_features)",
            got=X.shape,
            context="Feature matrix X"
        )
    if X.shape[1] != n_features:
        raise ShapeMismatchError(
            expected=f"(*, {n_features})",
            got=X.shape,
            context=f"Expected {n_features} features"
        )
    if len(X) != len(y):
        raise ShapeMismatchError(
            expected=f"({len(X)},)",
            got=y.shape,
            context="y must match number of samples in X"
        )


# --- Training with full error handling ---
def safe_training_run(model, config: dict, X_train, y_train):
    """Production-grade training with error handling."""
    try:
        logger.info(f"Starting training: {config}")
        validate_shapes(X_train, y_train, config["n_features"])
        model.fit(X_train, y_train)
        logger.info("Training completed successfully")

    except ShapeMismatchError as e:
        logger.error(f"Data shape error: {e}")
        raise  # Re-raise — caller needs to fix data

    except MemoryError:
        logger.critical(
            "Out of memory during training. "
            "Reduce batch_size or use data generators."
        )
        raise

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint...")
        model.save_checkpoint("interrupted_checkpoint.pkl")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")
        raise

    finally:
        # Always runs — cleanup resources
        logger.info("Training run ended")
```

### g. Common Mistakes

- **Catching `Exception` and silencing it**: `except Exception: pass` hides real bugs. Always log or re-raise.
- **Not using `from e`**: `raise NewException("msg") from e` preserves the original traceback. Without it, you lose the root cause.
- **Catching too broadly**: Catch specific exceptions. Catching `Exception` on a data loader might suppress a `MemoryError` you need to know about.
- **Missing `finally` for cleanup**: File handles, database connections, and GPU memory must be released even if exceptions occur.

---

## 6. Modules, Packages & Environments

### a. Why This Matters for AI Engineering

ML projects have complex dependency trees. PyTorch requires specific CUDA versions. Different models may need different library versions. A broken environment means a broken experiment. Environment reproducibility is a **first-class concern** in ML engineering.

### b. Practical Usage in ML

```bash
# Create isolated environment for a project
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
ml_env\Scripts\activate     # Windows

# Install ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn fastapi pydantic

# Freeze for reproducibility
pip freeze > requirements.txt

# Recreate environment anywhere
pip install -r requirements.txt
```

```python
# --- Proper module structure for ML projects ---
# ml_project/
# ├── __init__.py
# ├── data/
# │   ├── __init__.py
# │   ├── loader.py
# │   └── preprocessor.py
# ├── models/
# │   ├── __init__.py
# │   ├── baseline.py
# │   └── neural_net.py
# ├── training/
# │   ├── __init__.py
# │   └── trainer.py
# ├── evaluation/
# │   ├── __init__.py
# │   └── metrics.py
# └── config.py

# config.py — centralized configuration
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# __main__.py — entry point
if __name__ == "__main__":
    from ml_project.training.trainer import Trainer
    from ml_project.config import TrainingConfig

    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.run()
```

### g. Common Mistakes

- **Global pip install**: Always use virtual environments. Global installs cause version conflicts between projects.
- **Not pinning versions in `requirements.txt`**: `pip freeze` pins exact versions. Without pins, `pip install -r requirements.txt` may install different versions on different machines.
- **Circular imports**: `module_a` imports from `module_b` which imports from `module_a`. Fix by restructuring or using lazy imports.
- **Missing `__init__.py`**: Without it, Python won't treat the directory as a package.

---

## 7. File Handling & Serialization

### a. Why This Matters for AI Engineering

ML workflows constantly read and write data: loading training sets, saving model checkpoints, writing predictions, logging metrics. Understanding serialization formats is critical for data pipeline design. The wrong format choice can mean 10x slower data loading.

### b. Intuition (AI-Focused)

| Format | Use in ML | Speed | Size |
|---|---|---|---|
| CSV | Small datasets, human-readable features | Slow | Large |
| JSON | Model configs, API payloads, metadata | Medium | Medium |
| Parquet | Large feature tables, columnar ML data | Fast | Small |
| Pickle | Python objects: models, scalers, encoders | Fast | Medium |
| NPY/NPZ | NumPy arrays: processed features, embeddings | Very Fast | Small |
| HDF5 | Very large arrays: embeddings, image data | Very Fast | Small |

### c. Python Implementation

```python
import json
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- JSON for configs and metadata ---
def save_config(config: dict, path: str) -> None:
    """Save training config to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Config saved to {path}")


def load_config(path: str) -> dict:
    """Load and validate training config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


# --- Pickle for ML objects (models, scalers, encoders) ---
def save_model_artifact(obj, path: str) -> None:
    """
    Safely pickle an ML artifact.
    WARNING: Never unpickle files from untrusted sources.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Artifact saved: {path} ({path.stat().st_size / 1024:.1f} KB)")


def load_model_artifact(path: str):
    """Load pickled ML artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Artifact loaded from {path}")
    return obj


# --- NumPy arrays: fastest format for processed data ---
def save_processed_data(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save processed feature matrix and labels."""
    np.savez_compressed(path, X=X, y=y)
    logger.info(f"Saved {X.shape} feature matrix to {path}.npz")


def load_processed_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load processed feature matrix and labels."""
    data = np.load(f"{path}.npz")
    return data["X"], data["y"]


# --- Parquet for large feature tables ---
def save_feature_table(df: pd.DataFrame, path: str) -> None:
    """Save feature table in parquet format (10x smaller than CSV)."""
    path = Path(path)
    df.to_parquet(path, index=False, compression="snappy")
    csv_size = df.memory_usage(deep=True).sum() / 1024
    parquet_size = path.stat().st_size / 1024
    logger.info(
        f"Saved {len(df)} rows. "
        f"Parquet: {parquet_size:.0f} KB | "
        f"Estimated CSV: {csv_size:.0f} KB"
    )


# --- Logging setup for ML projects ---
def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{experiment_name}.log"),
        ],
    )
    return logging.getLogger(experiment_name)
```

### g. Common Mistakes

- **Using Pickle for model deployment**: Pickle is Python-version-dependent and a security risk. For deployment, use ONNX or framework-native formats (`.pt`, `.h5`).
- **Loading CSV in a loop**: Reading a large CSV file inside a loop re-parses it every time. Load once, cache.
- **Not using compressed NumPy**: `np.save()` is uncompressed. `np.savez_compressed()` is 3-5x smaller with minimal speed cost.
- **Forgetting `mode='wb'` for binary files**: Pickle requires binary mode. `open(path, 'w')` instead of `'wb'` gives a cryptic error.

---

## 8. Numerical Computing with NumPy

### a. Why This Matters for AI Engineering

NumPy is the mathematical foundation of all ML in Python. PyTorch tensors are conceptually NumPy arrays on GPU. Scikit-learn models take NumPy arrays as input. Every matrix multiplication in neural networks is a NumPy-style operation. If you don't understand NumPy, you cannot understand what your model is actually doing mathematically.

### b. Intuition (AI-Focused)

A dataset of 10,000 samples with 50 features is a **matrix** of shape `(10000, 50)`. A neural network layer applies a **linear transformation**: `Y = X @ W + b`. That `@` operator is matrix multiplication — NumPy's core operation. Broadcasting allows you to apply this to an entire batch without writing a single loop.

**Vectorization vs loops:**
```
Python loop over 1M elements: ~1 second
NumPy vectorized operation on same: ~1 millisecond
```
This is why ML is fast. Never write Python loops over array elements.

### c. Minimal Theory (Only What Matters)

**Shape conventions in ML:**
- `(n_samples, n_features)` — feature matrix
- `(n_samples,)` — labels
- `(n_samples, seq_len, d_model)` — sequence data (NLP)
- `(batch, channels, height, width)` — image data (NCHW format)

**Axes:**
- `axis=0` → operations across rows (per column)
- `axis=1` → operations across columns (per row)

**Broadcasting rules:** Arrays broadcast when their shapes are compatible from right to left:
- `(100, 50)` and `(50,)` → OK, `(50,)` broadcasts to `(100, 50)`
- `(100, 50)` and `(100,)` → Error — shapes don't align

**Essential linear algebra for ML:**
- `X @ W` → matrix multiplication (forward pass)
- `X.T` → transpose (used in gradient computation)
- `np.linalg.norm(v)` → vector norm (L2 distance)
- `np.linalg.inv(A)` → matrix inverse (normal equations)
- `np.linalg.eig(A)` → eigendecomposition (PCA)

### d. Python Implementation

```python
import numpy as np

# Set seed for reproducibility — ALWAYS do this in ML
np.random.seed(42)

# ============================================================
# ARRAYS AND SHAPES
# ============================================================

# Creating feature matrices
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 2, size=(1000,))  # Binary labels

print(f"X shape: {X.shape}")       # (1000, 20)
print(f"X dtype: {X.dtype}")       # float64
print(f"X memory: {X.nbytes / 1024:.1f} KB")

# Dtype matters for GPU memory and speed
X_float32 = X.astype(np.float32)  # Half the memory, same precision for most ML
print(f"float32 memory: {X_float32.nbytes / 1024:.1f} KB")

# ============================================================
# INDEXING AND SLICING (critical for batch operations)
# ============================================================

# Select first 32 samples (batch 0)
X_batch = X[:32]

# Select specific features (columns 0-4 and 10-14)
X_subset = X[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]]

# Boolean indexing — filter samples by label
X_positive = X[y == 1]
X_negative = X[y == 0]
print(f"Positive samples: {len(X_positive)}, Negative: {len(X_negative)}")

# Fancy indexing — shuffle dataset
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# ============================================================
# VECTORIZED OPERATIONS
# ============================================================

# Z-score normalization (manual — understand what Sklearn does internally)
mean = X.mean(axis=0)    # Shape: (20,) — one mean per feature
std = X.std(axis=0)      # Shape: (20,) — one std per feature
X_normalized = (X - mean) / std  # Broadcasting: (1000,20) - (20,) → (1000,20)

# Verify normalization
print(f"Mean after norm: {X_normalized.mean(axis=0).round(2)}")  # All ~0
print(f"Std after norm: {X_normalized.std(axis=0).round(2)}")    # All ~1

# Min-Max scaling
x_min = X.min(axis=0)
x_max = X.max(axis=0)
x_range = np.where(x_max - x_min == 0, 1, x_max - x_min)
X_minmax = (X - x_min) / x_range

# ============================================================
# LINEAR ALGEBRA (THE HEART OF ML)
# ============================================================

# Manual linear regression: y_hat = X @ weights + bias
n_features = 20
weights = np.random.randn(n_features) * 0.01
bias = 0.0
y_pred = X @ weights + bias  # Shape: (1000,)

# MSE loss — vectorized, no loops
mse = ((y - y_pred) ** 2).mean()
print(f"MSE: {mse:.4f}")

# Gradient of MSE w.r.t. weights
errors = y_pred - y  # Shape: (1000,)
grad_w = (2 / len(X)) * (X.T @ errors)  # Shape: (20,)
grad_b = (2 / len(X)) * errors.sum()

# Gradient descent step
lr = 0.01
weights -= lr * grad_w
bias -= lr * grad_b

# Cosine similarity (used in embedding-based models)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Pairwise distances between embeddings (used in kNN, clustering)
def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Efficient pairwise distance matrix using broadcasting.
    No Python loops.
    """
    # |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2*x_i·x_j
    sq_norms = (X ** 2).sum(axis=1)  # Shape: (n,)
    distances = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
    return np.sqrt(np.clip(distances, 0, None))  # Clip negatives (numerical errors)


# ============================================================
# BROADCASTING (Essential to understand)
# ============================================================

# Per-sample feature importance weighting
feature_weights = np.array([0.5, 1.0, 0.8, 0.3] + [1.0] * 16)  # Shape: (20,)
X_weighted = X * feature_weights  # (1000,20) * (20,) — broadcasts correctly

# Adding bias term to each sample
bias_vector = np.ones((1000, 1))
X_with_bias = np.hstack([X, bias_vector])  # Shape: (1000, 21)

# Softmax (multiclass output) — numerically stable
def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Subtract max before exp to prevent overflow.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


logits = np.random.randn(100, 10)  # 100 samples, 10 classes
probs = softmax(logits)
print(f"Probabilities sum to 1: {probs.sum(axis=1)[:5].round(6)}")

# ============================================================
# RANDOM SAMPLING (critical for ML: splits, dropout, augmentation)
# ============================================================

rng = np.random.default_rng(seed=42)  # Modern API — prefer this

# Train/val/test split
n = len(X)
indices = rng.permutation(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train = X[indices[:train_end]]
X_val = X[indices[train_end:val_end]]
X_test = X[indices[val_end:]]
y_train = y[indices[:train_end]]
y_val = y[indices[train_end:val_end]]
y_test = y[indices[val_end:]]

# Bootstrap sampling (used in bagging, uncertainty estimation)
def bootstrap_sample(X: np.ndarray, y: np.ndarray):
    """Sample with replacement — used in Random Forest."""
    indices = rng.integers(0, len(X), size=len(X))
    return X[indices], y[indices]


# Dropout mask (concept behind neural network dropout)
def apply_dropout(X: np.ndarray, dropout_rate: float = 0.5, training: bool = True) -> np.ndarray:
    if not training:
        return X
    mask = rng.random(X.shape) > dropout_rate
    return X * mask / (1 - dropout_rate)  # Scale to maintain expected value
```

### f. Mini Use Case: PCA from Scratch

```python
def pca_from_scratch(X: np.ndarray, n_components: int) -> tuple:
    """
    Principal Component Analysis using NumPy.
    Used for dimensionality reduction before training.
    """
    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(X_centered.T)  # Shape: (n_features, n_features)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue (explained variance)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Select top n_components
    components = eigenvectors[:, :n_components]

    # Project data
    X_reduced = X_centered @ components

    # Explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    return X_reduced, components, explained_variance_ratio


X_reduced, components, evr = pca_from_scratch(X, n_components=5)
print(f"Explained variance by 5 components: {evr.sum():.2%}")
```

### g. Common Mistakes

- **Shape confusion**: `(100,)` and `(100, 1)` look similar but behave differently in broadcasting. Use `.reshape(-1, 1)` to add an axis.
- **In-place operations on views**: `X[:, 0] += 1` modifies the original array. This causes silent bugs when you expect X to be unchanged.
- **Using Python loops over array elements**: `for x in array: total += x` is 1000x slower than `array.sum()`. Always vectorize.
- **Forgetting `axis` parameter**: `X.mean()` gives a scalar. `X.mean(axis=0)` gives per-feature means. The latter is almost always what you want.
- **Numerical instability**: Computing `exp(large_number)` gives `inf`. Always use numerically stable versions (subtract max for softmax, use `log_softmax` for cross-entropy).

---

## 9. Data Analysis with Pandas

### a. Why This Matters for AI Engineering

Pandas is how raw data becomes ML-ready features. EDA (Exploratory Data Analysis), missing value imputation, feature engineering, train/test splits — all happen in Pandas before a single model is trained. In production ML pipelines, Pandas (or its faster alternatives like Polars) sits at the data layer, feeding cleaned data to NumPy and then to models.

### b. Intuition (AI-Focused)

A Pandas DataFrame is a table where each row is a training sample and each column is a feature. Every transformation you apply — filling nulls, encoding categories, creating lag features — is feature engineering that directly impacts your model's performance.

### c. Python Implementation

```python
import pandas as pd
import numpy as np

# ============================================================
# LOADING AND INSPECTING DATA
# ============================================================

# Always specify dtypes on load for large files (avoids memory waste)
df = pd.read_csv(
    "transactions.csv",
    dtype={
        "user_id": "str",
        "amount": "float32",
        "category": "category",  # Efficient for low-cardinality strings
        "fraud": "int8",
    },
    parse_dates=["timestamp"],
)

# First things to check after loading
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df["fraud"].value_counts(normalize=True))  # Class distribution


# ============================================================
# FILTERING AND SELECTION
# ============================================================

# Select features and labels
feature_cols = ["amount", "hour", "day_of_week", "user_tx_count"]
X = df[feature_cols]
y = df["fraud"]

# Boolean filtering
high_value = df[df["amount"] > 1000]
fraud_transactions = df[(df["fraud"] == 1) & (df["amount"] > 500)]

# Multiple conditions
risky = df.query("amount > 1000 and fraud == 1 and category == 'online'")


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Systematic missing value treatment.
    Strategy depends on feature type and missingness mechanism.
    """
    df = df.copy()

    # Numerical: fill with median (robust to outliers)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical: fill with mode or 'UNKNOWN'
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        mode = df[col].mode()
        fill_value = mode[0] if len(mode) > 0 else "UNKNOWN"
        df[col] = df[col].fillna(fill_value)

    # Add missingness indicator (can be informative)
    for col in df.columns:
        if df[col].isnull().any():
            df[f"{col}_was_null"] = df[col].isnull().astype(int)

    return df


# ============================================================
# FEATURE ENGINEERING WITH GROUPBY
# ============================================================

def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-level aggregated features.
    This is one of the most common patterns in tabular ML.
    """
    df = df.copy()
    df = df.sort_values("timestamp")

    # Aggregate features per user
    user_stats = df.groupby("user_id").agg(
        user_tx_count=("amount", "count"),
        user_avg_amount=("amount", "mean"),
        user_std_amount=("amount", "std"),
        user_max_amount=("amount", "max"),
        user_fraud_rate=("fraud", "mean"),
    ).reset_index()

    # Join back to original DataFrame
    df = df.merge(user_stats, on="user_id", how="left")

    # Category frequency encoding
    cat_freq = df.groupby("category").size() / len(df)
    df["category_freq"] = df["category"].map(cat_freq)

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(22, 6).astype(int)

    return df


# ============================================================
# WINDOW FUNCTIONS (Rolling features — critical for time series ML)
# ============================================================

def create_rolling_features(df: pd.DataFrame, windows: list[int] = [7, 30]) -> pd.DataFrame:
    """
    Create rolling window aggregations.
    These capture temporal patterns for time-series ML.
    """
    df = df.copy().sort_values("timestamp")

    for window in windows:
        # Rolling within each user
        df[f"amount_rolling_mean_{window}d"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(f"{window}D", on=df.loc[x.index, "timestamp"]).mean())
        )
        df[f"tx_count_rolling_{window}d"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(f"{window}D", on=df.loc[x.index, "timestamp"]).count())
        )

    return df


# ============================================================
# MERGING AND JOINING (building feature tables)
# ============================================================

# Left join — standard for feature enrichment
df_enriched = df.merge(
    user_profile_df,    # user demographic features
    on="user_id",
    how="left",         # Keep all transactions even if no profile
    suffixes=("", "_profile")
)

# Multiple joins — building a complete feature table
df_final = (
    transactions_df
    .merge(user_df, on="user_id", how="left")
    .merge(merchant_df, on="merchant_id", how="left")
    .merge(card_df, on="card_id", how="left")
)


# ============================================================
# ENCODING CATEGORICALS
# ============================================================

def encode_categorical_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit encodings on train, apply to val.
    Critical: NEVER fit on val/test data.
    """
    df_train = df_train.copy()
    df_val = df_val.copy()

    for col in cat_cols:
        # Target encoding — encode based on target mean
        target_mean = df_train.groupby(col)["fraud"].mean()

        df_train[f"{col}_encoded"] = df_train[col].map(target_mean)
        df_val[f"{col}_encoded"] = df_val[col].map(target_mean)

        # Fill unknown categories with global mean
        global_mean = df_train["fraud"].mean()
        df_val[f"{col}_encoded"] = df_val[f"{col}_encoded"].fillna(global_mean)

    return df_train, df_val


# ============================================================
# PREPARING DATA FOR ML
# ============================================================

def prepare_ml_dataset(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> dict:
    """
    Final step: prepare clean DataFrame for model training.
    Returns dict with train/val/test splits.
    """
    drop_cols = drop_cols or []
    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in drop_cols
        and df[col].dtype in ["float32", "float64", "int8", "int32", "int64"]
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values

    n = len(X)
    indices = np.random.permutation(n)
    test_end = int(n * (1 - test_size))
    val_end = int(test_end * (1 - val_size))

    return {
        "X_train": X[indices[:val_end]],
        "X_val": X[indices[val_end:test_end]],
        "X_test": X[indices[test_end:]],
        "y_train": y[indices[:val_end]],
        "y_val": y[indices[val_end:test_end]],
        "y_test": y[indices[test_end:]],
        "feature_names": feature_cols,
    }
```

### g. Common Mistakes

- **Fitting encoders on the full dataset**: Target encoding, scaling, and imputation must be fit on train only. Leaking val/test statistics inflates performance metrics.
- **Ignoring `copy()`**: `df2 = df` creates a reference, not a copy. Modifying `df2` modifies `df`. Use `df.copy()`.
- **`apply()` when vectorized operations exist**: `df["col"].apply(np.log)` is 10x slower than `np.log(df["col"])`. Always vectorize first.
- **Integer overflow**: `int8` max is 127. Summing int8 columns can silently overflow. Use `int32` for sums.
- **Not resetting index after filtering**: `df[condition].reset_index(drop=True)` — without `reset_index`, index has gaps which causes subtle merge bugs.

---

## 10. Data Visualization

### a. Why This Matters for AI Engineering

You cannot improve what you cannot see. Visualization is how you understand your data before modeling, diagnose problems during training, and communicate results after. An ML engineer who can't visualize data is flying blind. EDA (Exploratory Data Analysis) is the first step in any real ML project.

### b. Python Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_theme(style="whitegrid")


# ============================================================
# EDA VISUALIZATION TOOLKIT
# ============================================================

def plot_class_distribution(y: np.ndarray, class_names: list = None, title: str = "Class Distribution"):
    """Check class imbalance — one of the first plots in any ML project."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    unique, counts = np.unique(y, return_counts=True)
    labels = class_names or [str(c) for c in unique]

    # Bar chart
    axes[0].bar(labels, counts, color=["#2196F3", "#F44336"])
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_ylabel("Count")
    for i, (label, count) in enumerate(zip(labels, counts)):
        axes[0].text(i, count, f"{count:,}", ha="center", va="bottom", fontweight="bold")

    # Pie chart
    axes[1].pie(counts, labels=labels, autopct="%1.1f%%", colors=["#2196F3", "#F44336"])
    axes[1].set_title(f"{title} (proportions)")

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Plot feature distributions split by target class.
    Reveals which features discriminate between classes.
    """
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for class_val in df[target_col].unique():
            subset = df[df[target_col] == class_val][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, label=f"Class {class_val}", density=True)
        ax.set_title(col)
        ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Class", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list):
    """
    Correlation heatmap — identify multicollinearity.
    Highly correlated features are redundant for many ML models.
    """
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(history: dict):
    """
    Plot train/val loss and metric curves.
    Essential for diagnosing overfitting/underfitting.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # Metric curves (e.g., accuracy, AUC)
    if "train_auc" in history:
        axes[1].plot(epochs, history["train_auc"], "b-", label="Train AUC", linewidth=2)
        axes[1].plot(epochs, history["val_auc"], "r-", label="Val AUC", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Training & Validation AUC")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 20):
    """Plot feature importances (from tree-based models)."""
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(sorted_features)), sorted_importances[::-1], color="#2196F3")
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### g. Common Mistakes

- **Not saving plots**: In notebook environments, plots display inline. In scripts, `plt.show()` blocks execution. Always `plt.savefig()` before `plt.show()`.
- **Plotting before checking data**: Always check `.describe()` and `.isnull().sum()` before visualizing. Outliers and nulls distort distributions.
- **Using 3D plots for ML**: Rarely useful for high-dimensional data. Use t-SNE or UMAP for dimensionality reduction visualization instead.
- **Not setting `tight_layout()`**: Subplots overlap without it. Always call it.

---

## 11. Standard Library High-ROI Parts

### a. Why This Matters for AI Engineering

The Python standard library contains tools you'll use daily in ML projects: file path management, timing operations, date-based feature engineering, logging. These are the plumbing of every ML system.

### b. Python Implementation

```python
import os
import sys
import time
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta


# --- pathlib: Modern, cross-platform path handling ---
# Always use pathlib, never string path concatenation

project_root = Path(__file__).parent.parent  # Navigate up directories
data_dir = project_root / "data" / "raw"
output_dir = project_root / "outputs"

# Create directories
output_dir.mkdir(parents=True, exist_ok=True)

# List all CSV files recursively
csv_files = list(data_dir.rglob("*.csv"))

# File properties
for f in csv_files:
    print(f.name, f.suffix, f.stat().st_size)


# --- os/sys: Environment and system info ---
# Environment variables for secrets (never hardcode API keys!)
db_password = os.environ.get("DB_PASSWORD", "")
model_dir = os.environ.get("MODEL_DIR", str(output_dir / "models"))

# Add project to Python path (useful in ML projects)
sys.path.insert(0, str(project_root))

# CPU count for parallelism
n_workers = os.cpu_count()


# --- datetime: Time-based features (critical in time-series ML) ---
def extract_time_features(timestamp: datetime) -> dict:
    """
    Extract rich temporal features from a timestamp.
    These are standard features in fraud detection and recommendation systems.
    """
    return {
        "hour": timestamp.hour,
        "day_of_week": timestamp.weekday(),      # 0=Mon, 6=Sun
        "day_of_month": timestamp.day,
        "month": timestamp.month,
        "quarter": (timestamp.month - 1) // 3 + 1,
        "is_weekend": int(timestamp.weekday() >= 5),
        "is_month_start": int(timestamp.day == 1),
        "is_month_end": int(timestamp.day == timestamp.replace(day=1,
                           month=timestamp.month % 12 + 1).day - 1),
        "days_since_epoch": (timestamp - datetime(2020, 1, 1)).days,
    }


# --- time: Performance profiling in ML ---
def profile_operation(func, *args, n_runs: int = 100, **kwargs):
    """Simple profiler for ML operations."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    import numpy as np
    print(f"{func.__name__}:")
    print(f"  Mean: {np.mean(times)*1000:.2f}ms")
    print(f"  Std:  {np.std(times)*1000:.2f}ms")
    print(f"  Min:  {np.min(times)*1000:.2f}ms")


# --- math: Mathematical utilities ---
# Sigmoid function (output layer for binary classification)
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

# Log with numerical stability
def safe_log(x: float, eps: float = 1e-7) -> float:
    return math.log(max(x, eps))


# --- logging: Production-grade logging ---
def setup_ml_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — all levels including DEBUG
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

---

## 12. Asynchronous & Parallel Python

### a. Why This Matters for AI Engineering

Modern ML systems have two parallelism needs:
1. **Data loading**: Load batches in parallel while the model trains (CPU-bound → multiprocessing)
2. **Model serving**: Handle many concurrent inference requests (I/O-bound → async)

Understanding when to use each — and the performance implications — is essential for production ML systems.

### b. Intuition (AI-Focused)

| Scenario | Pattern | Why |
|---|---|---|
| Loading 10 data files simultaneously | `ThreadPoolExecutor` | I/O-bound (disk reads) |
| Preprocessing batches in parallel | `ProcessPoolExecutor` | CPU-bound (compute) |
| Serving 100 concurrent API requests | `async/await` | I/O-bound (network) |
| Training on multiple GPUs | Distributed frameworks | Compute-bound |

**The GIL (Global Interpreter Lock)**: Python threads cannot run Python code in parallel. They can overlap on I/O waits. For true CPU parallelism, use `multiprocessing`.

### c. Python Implementation

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable
import numpy as np


# ============================================================
# MULTITHREADING — I/O-bound tasks (reading files, API calls)
# ============================================================

def load_csv_file(file_path: str) -> np.ndarray:
    """Load a single CSV file (I/O-bound)."""
    import pandas as pd
    return pd.read_csv(file_path).values


def load_dataset_parallel(file_paths: list[str], n_workers: int = 4) -> list:
    """Load multiple files in parallel using threads."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(load_csv_file, file_paths))
    return results


# ============================================================
# MULTIPROCESSING — CPU-bound tasks (feature engineering, encoding)
# ============================================================

def preprocess_chunk(chunk: np.ndarray) -> np.ndarray:
    """Preprocess a data chunk (CPU-bound)."""
    # Simulate heavy computation
    return (chunk - chunk.mean(axis=0)) / chunk.std(axis=0)


def preprocess_parallel(X: np.ndarray, n_workers: int = 4) -> np.ndarray:
    """
    Parallel preprocessing of large dataset.
    Split into chunks, process each in a separate process.
    """
    chunks = np.array_split(X, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        processed_chunks = list(executor.map(preprocess_chunk, chunks))

    return np.vstack(processed_chunks)


# ============================================================
# ASYNC — High-concurrency inference serving
# ============================================================

import httpx  # Async HTTP client

async def call_model_api(session: httpx.AsyncClient, payload: dict) -> dict:
    """Single async model API call."""
    response = await session.post(
        "http://localhost:8000/predict",
        json=payload,
        timeout=5.0,
    )
    return response.json()


async def batch_inference_async(payloads: list[dict]) -> list[dict]:
    """
    Send many inference requests concurrently.
    Much faster than sequential requests.
    """
    async with httpx.AsyncClient() as client:
        tasks = [call_model_api(client, payload) for payload in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


# Usage
async def main():
    payloads = [{"features": [1.0, 2.0, 3.0]} for _ in range(100)]
    start = time.perf_counter()
    results = await batch_inference_async(payloads)
    print(f"100 async requests: {time.perf_counter() - start:.2f}s")


asyncio.run(main())
```

### g. Common Mistakes

- **Using threads for CPU-bound work**: Due to the GIL, Python threads don't speed up computation. Use `ProcessPoolExecutor`.
- **Spawning too many processes**: More processes than CPU cores doesn't help and adds overhead. `n_workers = os.cpu_count()`.
- **Not using `asyncio.gather`**: Sequential `await` calls are just sequential. Use `gather` to run coroutines concurrently.
- **Mixing sync and async code incorrectly**: Calling a blocking function inside an async context blocks the entire event loop. Use `loop.run_in_executor` to run blocking code in a thread pool from async context.

---

## 13. FastAPI & Pydantic

### a. Why This Matters for AI Engineering

Training a model is only half the job. Deploying it as an API that other services can call is the other half. FastAPI is the industry standard for ML model serving in Python. Pydantic ensures your model receives correctly typed, validated inputs — preventing a malformed request from crashing your inference service.

### b. Python Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


# ============================================================
# PYDANTIC — Input validation for ML APIs
# ============================================================

class FraudPredictionRequest(BaseModel):
    """Input schema for fraud detection API."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, le=100000, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    category: str = Field(..., description="Transaction category")
    user_tx_count: int = Field(..., ge=0, description="Historical transaction count for user")
    user_avg_amount: float = Field(..., ge=0, description="User's historical average amount")

    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: str) -> str:
        valid_categories = {"online", "retail", "restaurant", "travel", "other"}
        if v.lower() not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v.lower()

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)


class FraudPredictionResponse(BaseModel):
    """Output schema for fraud detection API."""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    transactions: list[FraudPredictionRequest]


# ============================================================
# FASTAPI — ML model serving
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using ML",
    version="1.0.0",
)

# Global model object — loaded once at startup
model = None
scaler = None
label_encoder = None
FEATURE_COLS = ["amount", "hour", "day_of_week", "user_tx_count", "user_avg_amount"]
MODEL_VERSION = "v1.2.0"


@app.on_event("startup")
async def load_model():
    """Load model artifacts at startup — not on every request."""
    global model, scaler, label_encoder
    try:
        with open("artifacts/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("artifacts/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Model {MODEL_VERSION} loaded successfully")
    except FileNotFoundError as e:
        logger.critical(f"Model artifacts not found: {e}")
        raise


def extract_features(request: FraudPredictionRequest) -> np.ndarray:
    """Convert API request to feature vector."""
    category_map = {"online": 0, "retail": 1, "restaurant": 2, "travel": 3, "other": 4}
    features = np.array([[
        request.amount,
        request.hour,
        request.day_of_week,
        category_map.get(request.category, 4),
        request.user_tx_count,
        request.user_avg_amount,
    ]], dtype=np.float32)
    return scaler.transform(features)


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    return "high"


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(request: FraudPredictionRequest):
    """Single transaction fraud prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = extract_features(request)
        fraud_prob = float(model.predict_proba(features)[0, 1])

        return FraudPredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=fraud_prob,
            is_fraud=fraud_prob >= 0.5,
            risk_level=get_risk_level(fraud_prob),
            model_version=MODEL_VERSION,
        )

    except Exception as e:
        logger.error(f"Prediction failed for transaction {request.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", response_model=list[FraudPredictionResponse])
async def predict_fraud_batch(batch: BatchPredictionRequest):
    """Batch prediction — more efficient than multiple single requests."""
    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000")

    results = []
    for tx in batch.transactions:
        try:
            features = extract_features(tx)
            fraud_prob = float(model.predict_proba(features)[0, 1])
            results.append(FraudPredictionResponse(
                transaction_id=tx.transaction_id,
                fraud_probability=fraud_prob,
                is_fraud=fraud_prob >= 0.5,
                risk_level=get_risk_level(fraud_prob),
                model_version=MODEL_VERSION,
            ))
        except Exception as e:
            logger.warning(f"Skipping transaction {tx.transaction_id}: {e}")

    return results


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_version": MODEL_VERSION,
    }


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### g. Common Mistakes

- **Loading model on every request**: Pickle loading is slow. Load once at startup with `@app.on_event("startup")`.
- **Not validating inputs**: Without Pydantic validation, a malformed request (e.g., `amount = "abc"`) crashes the prediction function with a cryptic error.
- **No batch endpoint**: Individual requests have network overhead. Batch endpoints are 10-100x more efficient.
- **No health check endpoint**: Load balancers and orchestration systems (Kubernetes) require `/health` to know if the service is up.

---

## 14. Production-Ready Python Mindset

### a. Why This Matters for AI Engineering

There is a massive gap between "code that works on my laptop" and "code that runs reliably in production." ML models in production handle millions of requests, run for months without restarts, and are maintained by teams. Production-ready code is the difference between a prototype and a real system.

### b. Key Principles with ML Context

```python
# ============================================================
# 1. MODULAR DESIGN — each module does one thing
# ============================================================

# BAD — monolithic notebook-style code
def do_everything(data_path, target, lr, epochs):
    df = pd.read_csv(data_path)
    df = df.dropna()
    df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
    # ... 200 more lines ...

# GOOD — decomposed into clean, testable functions
def load_raw_data(path: str) -> pd.DataFrame: ...
def clean_data(df: pd.DataFrame) -> pd.DataFrame: ...
def engineer_features(df: pd.DataFrame) -> pd.DataFrame: ...
def split_dataset(df: pd.DataFrame) -> tuple: ...
def train(X, y, config: TrainingConfig) -> Model: ...
def evaluate(model: Model, X_test, y_test) -> dict: ...


# ============================================================
# 2. CONFIGURATION MANAGEMENT
# ============================================================

# BAD — hardcoded values scattered in code
model = train(X, y, lr=0.001, epochs=100, hidden=256)

# GOOD — all config in one place, version-controlled
from dataclasses import dataclass, asdict
import json

@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline_v1"
    learning_rate: float = 1e-3
    epochs: int = 100
    hidden_size: int = 256
    batch_size: int = 32
    seed: int = 42

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**json.load(f))


# ============================================================
# 3. LOGGING OVER PRINTING
# ============================================================

# BAD
print(f"Epoch 1 done, loss = {loss}")

# GOOD
logger = logging.getLogger(__name__)
logger.info(f"Epoch {epoch:03d}/{epochs} | loss={loss:.4f} | lr={lr:.2e}")


# ============================================================
# 4. TESTING ML CODE
# ============================================================

import pytest
import numpy as np

def test_standard_scaler_fit():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler.fit(X)
    assert np.allclose(scaler.mean_, [3, 4])
    assert np.allclose(scaler.std_, [1.63, 1.63], atol=0.01)


def test_standard_scaler_transform_zero_mean():
    scaler = StandardScaler()
    X = np.random.randn(100, 5)
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


def test_model_output_shape():
    """Always test that model output shape is correct."""
    model = MyModel(input_size=10, output_size=3)
    X = np.random.randn(32, 10)  # Batch of 32
    output = model.predict(X)
    assert output.shape == (32, 3), f"Expected (32, 3), got {output.shape}"


# ============================================================
# 5. READING OTHER PEOPLE'S CODE
# ============================================================

# Strategy for reading ML codebases:
# 1. Find the entry point (main.py, train.py, __main__.py)
# 2. Understand the data flow: What goes in? What comes out?
# 3. Read the model class __init__ and forward/predict methods
# 4. Check the training loop
# 5. Check the config/arguments
# Then read individual components

# Use Python introspection
import inspect
print(inspect.getsource(SomeMLClass.fit))  # Read source of any method
print(SomeMLClass.__mro__)                  # Understand inheritance chain
```

### g. Common Mistakes

- **No version control for experiments**: Each experiment should save its config, metrics, and model artifact. Without this, you can't reproduce results.
- **Hardcoding file paths**: Use `pathlib.Path` relative to project root or environment variables.
- **No tests for data pipelines**: A data preprocessing bug can silently corrupt your features. Test critical transformation functions.
- **Global mutable state**: Module-level mutable objects shared across functions cause hard-to-reproduce bugs in production.

---

## 15. Cross-Topic Connections

Understanding how these topics connect within ML systems is what separates a junior from a senior engineer.

```
PYTHON CORE (Syntax, Control Flow, Functions)
    ↓
    └── Powers ALL Python ML code

DATA STRUCTURES (Lists, Dicts, Sets)
    ↓
    ├── Vocabulary in NLP models (dict: word→index)
    ├── Batch management (list of arrays)
    └── Feature deduplication (set operations)

OOP (Classes, Inheritance, Dataclasses)
    ↓
    ├── Custom PyTorch nn.Module definitions
    ├── Custom Sklearn transformers
    ├── Dataset classes for DataLoaders
    └── Config management (dataclasses)

NUMPY (Vectorized math)
    ↓
    ├── Feature matrices and label vectors
    ├── Manual gradient computation
    ├── Statistical operations (mean, std, percentiles)
    └── Foundation of all ML framework tensors

PANDAS (Data analysis)
    ↓
    ├── Feature engineering (GroupBy, rolling)
    ├── Missing value imputation
    ├── Train/val/test splitting
    └── EDA before modeling

FILE HANDLING + SERIALIZATION
    ↓
    ├── Model checkpoint saving (Pickle, .pt, .h5)
    ├── Dataset loading (Parquet, CSV, NPZ)
    └── Config persistence (JSON)

GENERATORS + ITERTOOLS
    ↓
    ├── Memory-efficient data loading (too large for RAM)
    ├── Hyperparameter grid generation
    └── Batch iteration in training loops

ERROR HANDLING + LOGGING
    ↓
    ├── Robust data pipelines (handle corrupt data)
    ├── Training run monitoring
    └── Production inference error tracking

ASYNC + PARALLEL
    ↓
    ├── Parallel data loading (multiprocessing)
    ├── Concurrent inference (async)
    └── Distributed training coordination

FASTAPI + PYDANTIC
    ↓
    ├── Model serving endpoints
    ├── Input validation
    └── API documentation
```

---

## 16. End-to-End Practical System View

Here is how every topic in this document comes together in a real production ML system:

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL ML SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DATA INGESTION                                          │
│     ├── SQL queries → raw data (Feature pipeline)          │
│     ├── pandas.read_parquet/csv → DataFrame                │
│     └── pathlib → file management                          │
│                                                             │
│  2. DATA PROCESSING                                         │
│     ├── Pandas → clean, join, aggregate                    │
│     ├── Python OOP → pipeline classes                      │
│     ├── NumPy → numerical operations                       │
│     └── Generators → memory-efficient processing           │
│                                                             │
│  3. FEATURE ENGINEERING                                     │
│     ├── Pandas GroupBy → aggregate features                │
│     ├── NumPy → mathematical transformations               │
│     ├── Python functions → reusable feature logic          │
│     └── Serialization → save feature tables                │
│                                                             │
│  4. MODEL TRAINING                                          │
│     ├── NumPy arrays → feature matrix                      │
│     ├── OOP → custom Dataset, DataLoader, Model            │
│     ├── Control flow → training loop, early stopping       │
│     ├── Error handling → robust training                   │
│     └── Logging → experiment tracking                      │
│                                                             │
│  5. EVALUATION                                              │
│     ├── NumPy → metric computation                         │
│     ├── Matplotlib/Seaborn → result visualization          │
│     └── Pandas → prediction analysis                       │
│                                                             │
│  6. DEPLOYMENT                                              │
│     ├── FastAPI → model serving endpoint                   │
│     ├── Pydantic → input validation                        │
│     ├── Async → concurrent request handling                │
│     ├── Pickle/ONNX → model artifact serving               │
│     └── Logging → production monitoring                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Complete System Flow — Code

```python
# main.py — Complete ML system entry point

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MLPipelineConfig:
    data_path: str = "data/raw/transactions.csv"
    model_output_dir: str = "artifacts"
    experiment_name: str = "fraud_detection_v1"
    test_size: float = 0.2
    val_size: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    seed: int = 42


def run_pipeline(config: MLPipelineConfig):
    """
    End-to-end ML pipeline.
    Each step is a clean, testable function.
    """
    # Setup
    np.random.seed(config.seed)
    output_dir = Path(config.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"Starting experiment: {config.experiment_name}")

    # 1. Load data
    logger.info("Step 1: Loading data")
    df = pd.read_csv(config.data_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # 2. Clean data
    logger.info("Step 2: Cleaning data")
    df = clean_data(df)

    # 3. Feature engineering
    logger.info("Step 3: Engineering features")
    df = engineer_features(df)

    # 4. Prepare ML dataset
    logger.info("Step 4: Preparing ML dataset")
    dataset = prepare_ml_dataset(df, target_col="fraud")

    # 5. Train model
    logger.info("Step 5: Training model")
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        learning_rate=config.learning_rate,
        n_estimators=config.epochs,
        random_state=config.seed,
    )
    model.fit(dataset["X_train"], dataset["y_train"])

    # 6. Evaluate
    logger.info("Step 6: Evaluating model")
    from sklearn.metrics import roc_auc_score, classification_report
    y_pred_proba = model.predict_proba(dataset["X_test"])[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(dataset["y_test"], y_pred_proba)
    logger.info(f"Test AUC: {auc:.4f}")
    print(classification_report(dataset["y_test"], y_pred))

    # 7. Save artifacts
    logger.info("Step 7: Saving artifacts")
    import pickle
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {"test_auc": auc, "experiment": config.experiment_name}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Pipeline complete. Artifacts saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    config = MLPipelineConfig()
    results = run_pipeline(config)
    print(f"\nFinal Results: {results}")
```

---

## 17. Hands-On Projects

### Project 1: Fraud Detection Feature Pipeline

**Problem Statement**: Build a complete feature engineering pipeline for a credit card fraud detection system. The pipeline must handle raw transaction data, engineer meaningful features, and output a clean ML-ready dataset.

**Dataset Description**: Simulated credit card transactions with fields: `user_id`, `transaction_id`, `timestamp`, `amount`, `merchant`, `category`, `fraud` (0/1).

```python
# project_1/feature_pipeline.py

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("project_1/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: Generate realistic sample data
# ============================================================

def generate_sample_data(n_transactions: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fraud dataset for demonstration."""
    rng = np.random.default_rng(seed)

    n_users = 500
    categories = ["online", "retail", "restaurant", "travel", "other"]
    merchants = [f"merchant_{i}" for i in range(200)]

    data = {
        "user_id": rng.integers(0, n_users, n_transactions).astype(str),
        "transaction_id": [f"tx_{i:06d}" for i in range(n_transactions)],
        "timestamp": pd.date_range("2024-01-01", periods=n_transactions, freq="5min"),
        "amount": rng.exponential(scale=100, size=n_transactions).round(2),
        "merchant": rng.choice(merchants, n_transactions),
        "category": rng.choice(categories, n_transactions, p=[0.35, 0.25, 0.2, 0.1, 0.1]),
    }

    df = pd.DataFrame(data)

    # Create fraud signal with realistic patterns
    fraud_prob = (
        0.02  # Base rate 2%
        + 0.15 * (df["amount"] > 500)  # High-value transactions
        + 0.10 * (df["timestamp"].dt.hour.isin([0, 1, 2, 3]))  # Late night
        + 0.05 * (df["category"] == "online")  # Online transactions
    )
    df["fraud"] = (rng.random(n_transactions) < fraud_prob.clip(0, 1)).astype(int)

    # Add some missing values (realistic)
    mask = rng.random(n_transactions) < 0.05
    df.loc[mask, "amount"] = np.nan

    logger.info(f"Generated {n_transactions} transactions. Fraud rate: {df['fraud'].mean():.2%}")
    return df


# ============================================================
# STEP 2: Data cleaning
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction data."""
    df = df.copy()
    initial_size = len(df)

    # Handle missing amounts — use median per category
    df["amount"] = df.groupby("category")["amount"].transform(
        lambda x: x.fillna(x.median())
    )

    # Remove transactions with amount <= 0
    df = df[df["amount"] > 0]

    logger.info(
        f"Cleaned data: {initial_size} → {len(df)} rows "
        f"({initial_size - len(df)} removed)"
    )
    return df.reset_index(drop=True)


# ============================================================
# STEP 3: Feature engineering
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready features from raw transaction data."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # --- Temporal features ---
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(22, 6).astype(int)
    df["month"] = df["timestamp"].dt.month

    # --- User-level aggregate features ---
    user_stats = df.groupby("user_id").agg(
        user_tx_count=("amount", "count"),
        user_avg_amount=("amount", "mean"),
        user_std_amount=("amount", "std"),
        user_max_amount=("amount", "max"),
        user_fraud_history=("fraud", "mean"),
    ).reset_index()
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)
    df = df.merge(user_stats, on="user_id", how="left")

    # --- Amount-based features ---
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_vs_user_avg"] = df["amount"] / (df["user_avg_amount"] + 1)
    df["is_high_value"] = (df["amount"] > df["user_avg_amount"] * 3).astype(int)

    # --- Category frequency encoding ---
    cat_freq = df.groupby("category").size() / len(df)
    df["category_freq"] = df["category"].map(cat_freq)

    # --- Category target encoding (with 5-fold CV to prevent leakage) ---
    # Simple version: compute on full train, apply with smoothing
    global_fraud_rate = df["fraud"].mean()
    smoothing = 10
    category_stats = df.groupby("category")["fraud"].agg(["mean", "count"])
    category_stats["smoothed_rate"] = (
        (category_stats["mean"] * category_stats["count"] + global_fraud_rate * smoothing)
        / (category_stats["count"] + smoothing)
    )
    df["category_fraud_rate"] = df["category"].map(category_stats["smoothed_rate"])

    # --- Merchant frequency ---
    merchant_freq = df.groupby("merchant").size() / len(df)
    df["merchant_freq"] = df["merchant"].map(merchant_freq)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ============================================================
# STEP 4: Prepare ML dataset
# ============================================================

FEATURE_COLS = [
    "amount", "amount_log", "amount_vs_user_avg", "is_high_value",
    "hour", "day_of_week", "is_weekend", "is_night", "month",
    "user_tx_count", "user_avg_amount", "user_std_amount",
    "user_max_amount", "user_fraud_history",
    "category_freq", "category_fraud_rate", "merchant_freq",
]

TARGET_COL = "fraud"


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Create train/test split respecting temporal order.
    IMPORTANT: For time-series data, split chronologically, not randomly.
    """
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.int8)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.int8)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train fraud rate: {y_train.mean():.2%}, Test fraud rate: {y_test.mean():.2%}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLS,
        "scaler": scaler,
    }


# ============================================================
# STEP 5: Train and evaluate
# ============================================================

def train_and_evaluate(dataset: dict) -> dict:
    """Train model and compute evaluation metrics."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]

    logger.info("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "avg_precision": average_precision_score(y_test, y_pred_proba),
    }

    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(dataset["feature_names"], importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 Features:")
    for name, imp in feature_importance[:10]:
        print(f"  {name:35s} {imp:.4f}")

    # Save model
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ============================================================
# MAIN — Run complete pipeline
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION FEATURE PIPELINE")
    logger.info("=" * 60)

    df_raw = generate_sample_data(n_transactions=10000)
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    dataset = prepare_dataset(df_features, test_size=0.2)
    metrics = train_and_evaluate(dataset)

    logger.info("Pipeline complete!")
    logger.info(f"Final AUC-ROC: {metrics['auc_roc']:.4f}")
```

---

### Project 2: ML Model Serving API

**Problem Statement**: Take the trained fraud detection model from Project 1 and deploy it as a production-ready REST API with input validation, batch prediction, and health monitoring.

```python
# project_2/serve.py
# Run: uvicorn project_2.serve:app --reload

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import pickle
import numpy as np
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("project_1/outputs")
FEATURE_COLS = [
    "amount", "amount_log", "amount_vs_user_avg", "is_high_value",
    "hour", "day_of_week", "is_weekend", "is_night", "month",
    "user_tx_count", "user_avg_amount", "user_std_amount",
    "user_max_amount", "user_fraud_history",
    "category_freq", "category_fraud_rate", "merchant_freq",
]

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Model state
state = {"model": None, "scaler": None, "request_count": 0, "start_time": None}


@app.on_event("startup")
async def startup():
    logger.info("Loading model artifacts...")
    try:
        with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
            state["model"] = pickle.load(f)
        with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
            state["scaler"] = pickle.load(f)
        state["start_time"] = datetime.now()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model artifacts not found. Run project_1 first.")


class TransactionFeatures(BaseModel):
    amount: float = Field(..., gt=0)
    amount_log: float
    amount_vs_user_avg: float
    is_high_value: int = Field(..., ge=0, le=1)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_night: int = Field(..., ge=0, le=1)
    month: int = Field(..., ge=1, le=12)
    user_tx_count: int = Field(..., ge=0)
    user_avg_amount: float = Field(..., ge=0)
    user_std_amount: float = Field(..., ge=0)
    user_max_amount: float = Field(..., ge=0)
    user_fraud_history: float = Field(..., ge=0, le=1)
    category_freq: float = Field(..., ge=0, le=1)
    category_fraud_rate: float = Field(..., ge=0, le=1)
    merchant_freq: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    latency_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TransactionFeatures, background_tasks: BackgroundTasks):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    state["request_count"] += 1

    X = np.array([[getattr(features, col) for col in FEATURE_COLS]], dtype=np.float32)
    X_scaled = state["scaler"].transform(X)
    prob = float(state["model"].predict_proba(X_scaled)[0, 1])

    latency_ms = (time.perf_counter() - start) * 1000

    risk = "low" if prob < 0.3 else "medium" if prob < 0.7 else "high"

    background_tasks.add_task(
        logger.info,
        f"Request #{state['request_count']} | prob={prob:.3f} | latency={latency_ms:.1f}ms"
    )

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        is_fraud=prob >= 0.5,
        risk_level=risk,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health():
    uptime = (datetime.now() - state["start_time"]).seconds if state["start_time"] else 0
    return {
        "status": "healthy" if state["model"] is not None else "degraded",
        "uptime_seconds": uptime,
        "requests_served": state["request_count"],
    }
```

---

## 18. Cheat Sheets

### Math Formulas Used in ML

| Operation | Formula | NumPy |
|---|---|---|
| Z-score normalization | `(x - μ) / σ` | `(X - X.mean(0)) / X.std(0)` |
| Min-Max scaling | `(x - min) / (max - min)` | `(X - X.min(0)) / (X.max(0) - X.min(0))` |
| Dot product | `a · b = Σ aᵢbᵢ` | `np.dot(a, b)` |
| Matrix multiply | `C = A @ B` | `A @ B` |
| MSE | `(1/n) Σ (yᵢ - ŷᵢ)²` | `((y - y_pred)**2).mean()` |
| Binary cross-entropy | `-[y log(p) + (1-y) log(1-p)]` | `-y*np.log(p) - (1-y)*np.log(1-p)` |
| Softmax | `exp(xᵢ) / Σ exp(xⱼ)` | `np.exp(x) / np.exp(x).sum()` |
| Sigmoid | `1 / (1 + e^(-x))` | `1 / (1 + np.exp(-x))` |
| L2 norm | `√(Σ xᵢ²)` | `np.linalg.norm(x)` |
| Cosine similarity | `(a · b) / (‖a‖ ‖b‖)` | `np.dot(a,b) / (norm(a)*norm(b))` |

### Python Patterns for ML

| Pattern | Code |
|---|---|
| Shuffle dataset | `idx = np.random.permutation(n); X, y = X[idx], y[idx]` |
| Train/val/test split | `train, val, test = np.split(X, [int(.7*n), int(.85*n)])` |
| One-hot encode | `np.eye(n_classes)[y]` |
| Batch iterate | `for i in range(0, n, batch_size): X[i:i+batch_size]` |
| Top-k indices | `np.argsort(scores)[-k:][::-1]` |
| Clip values | `np.clip(X, a_min, a_max)` |
| Log + numerical stability | `np.log(X + 1e-8)` |
| Count classes | `np.bincount(y)` |
| Pairwise distances | `np.sqrt(((X[:, None] - X[None, :])**2).sum(-1))` |
| Apply function by group | `df.groupby("col")["feat"].transform(func)` |
| Load large CSV in chunks | `pd.read_csv(f, chunksize=10000)` |
| Save/load NumPy | `np.savez_compressed(f, X=X, y=y)` |
| Fast categorical encode | `pd.factorize(df["col"])` |

### Pandas Patterns for ML

| Task | Code |
|---|---|
| Check nulls | `df.isnull().sum()` |
| Fill nulls with median | `df["col"].fillna(df["col"].median())` |
| Drop high-null cols | `df.dropna(axis=1, thresh=int(0.5*len(df)))` |
| One-hot encode | `pd.get_dummies(df, columns=cat_cols)` |
| Select numeric cols | `df.select_dtypes(include="number")` |
| Feature aggregation | `df.groupby("id")["val"].agg(["mean","std","count"])` |
| Rolling mean | `df["col"].rolling(7).mean()` |
| Lag features | `df["col"].shift(1)` |
| Memory optimization | `df["col"].astype("float32")` |
| Parse dates | `pd.to_datetime(df["date"])` |
| Sort + reset index | `df.sort_values("timestamp").reset_index(drop=True)` |
| Merge DataFrames | `df1.merge(df2, on="id", how="left")` |

---

## 19. Interview Preparation

### Conceptual Questions

**Q: What is the difference between `iloc` and `loc` in Pandas?**
> `loc` is label-based (uses index values). `iloc` is position-based (uses integer positions). After filtering, `loc` references the original index, `iloc` always uses 0-based positions. In ML, always `reset_index(drop=True)` after filtering to avoid confusing behavior with `loc`.

**Q: Why is vectorization faster than Python loops in NumPy?**
> NumPy operations execute in C, bypassing Python's interpreter overhead and the GIL. They use SIMD (Single Instruction Multiple Data) CPU instructions to process multiple elements simultaneously. A Python loop over 1M elements has 1M Python bytecode executions; NumPy's `array.sum()` has one C function call.

**Q: What is broadcasting in NumPy and why does it matter?**
> Broadcasting allows operations on arrays of different shapes by automatically expanding dimensions. This enables you to subtract the per-feature mean `(20,)` from a feature matrix `(1000, 20)` without writing a loop or manually reshaping. Without broadcasting, normalization would require explicit `np.tile()` or loops — both slower.

**Q: When would you use a generator instead of a list in an ML pipeline?**
> When your dataset is larger than available RAM. A generator yields one item at a time, using O(1) memory instead of O(n). For a 100GB dataset, a list would crash; a generator processes it batch by batch. PyTorch's DataLoader uses this pattern internally.

**Q: What is the danger of fitting a scaler on the full dataset before splitting?**
> It causes **data leakage**. The scaler's mean and std are computed using information from the test set. At inference time, you only have the training statistics. Models trained this way appear better in evaluation than they actually are in production. Always fit on train, transform on val/test.

### Practical / Coding Questions

**Q: Write a function to detect and handle outliers in a feature column.**
```python
def clip_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Clip outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper)
```

**Q: How would you efficiently compute the dot product of two large matrices?**
```python
# Use @ operator or np.dot — both call optimized BLAS routines
result = A @ B         # Preferred syntax
result = np.dot(A, B)  # Equivalent

# For very large matrices, consider float32 (2x faster on modern hardware)
result = A.astype(np.float32) @ B.astype(np.float32)
```

**Q: How would you implement class-weighted loss for an imbalanced dataset?**
```python
from collections import Counter
import numpy as np

def compute_class_weights(y: np.ndarray) -> dict:
    """Compute inverse-frequency class weights."""
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    return {cls: total / (n_classes * count) for cls, count in counts.items()}

weights = compute_class_weights(y_train)
# {0: 0.571, 1: 2.5}  — minority class gets higher weight
```

**Q: How do you ensure reproducibility in an ML experiment?**
```python
import numpy as np
import random
import os

def set_global_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
```

### System Design Questions

**Q: You need to serve a fraud detection model that receives 10,000 requests per second. How do you architect this?**

Key points to cover:
1. **Async FastAPI** with uvicorn workers — non-blocking I/O
2. **Batch prediction endpoint** — group requests into batches for GPU efficiency
3. **Model loaded once at startup** — not per request
4. **Feature store** — precomputed user features cached in Redis
5. **Horizontal scaling** — multiple replicas behind a load balancer
6. **Monitoring** — latency, throughput, prediction distribution drift

**Q: Your data pipeline is slow and you have 50GB of CSV files to process. How do you fix it?**

1. Convert CSV to Parquet — 5-10x faster reads, 3-5x smaller
2. Use chunked loading: `pd.read_csv(f, chunksize=100000)`
3. Process chunks in parallel with `ProcessPoolExecutor`
4. Use `float32` instead of `float64` — halves memory
5. Consider Polars or DuckDB for very large datasets

---

## 20. Resources

### Python & NumPy

- **Python Docs** — https://docs.python.org/3/ (official reference)
- **NumPy User Guide** — https://numpy.org/doc/stable/user/ (official, practical)
- **NumPy for ML practitioners** — https://numpy.org/doc/stable/user/quickstart.html

### Pandas

- **Pandas User Guide** — https://pandas.pydata.org/docs/user_guide/
- **Effective Pandas** by Matt Harrison (book — highly practical)
- **Modern Pandas** tutorial series — https://tomaugspurger.net/posts/modern-1-intro/

### ML Engineering Foundations

- **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — Aurélien Géron (best practical book)
- **Full Stack Deep Learning** — https://fullstackdeeplearning.com (free, practical, production-focused)
- **Made With ML** — https://madewithml.com (MLOps and production ML)
- **Andrej Karpathy's Neural Networks: Zero to Hero** — https://karpathy.ai/zero-to-hero.html

### FastAPI & Deployment

- **FastAPI Docs** — https://fastapi.tiangolo.com (excellent official docs)
- **Pydantic Docs** — https://docs.pydantic.dev
- **Building ML APIs** — https://testdriven.io/blog/fastapi-machine-learning/

### Practice Datasets

- **Kaggle** — https://www.kaggle.com/datasets (real, messy, competition-grade data)
- **UCI ML Repository** — https://archive.ics.uci.edu/ml/index.php
- **HuggingFace Datasets** — https://huggingface.co/datasets

### Code Quality

- **Google Python Style Guide** — https://google.github.io/styleguide/pyguide.html
- **PEP8** — https://peps.python.org/pep-0008/
- **Clean Code in Python** by Mariano Anaya (book)

---

*Document generated for AI/ML Engineering education. Every concept is production-relevant.*
*Version 1.0 | Focus: Python Foundations for ML Engineers*
