# Foundations for AI Engineering: SQL — The Data Engineer's Backbone

> *"Every ML model is only as good as the data feeding it. SQL is how you get that data."*

---

## Overview

Before a single gradient is computed, before a model ever sees a feature vector, data must be extracted, cleaned, joined, transformed, and shaped into something a machine can learn from. That work — the unglamorous, critical, foundational work — is done in SQL.

SQL is not a "nice to have" for AI engineers. It is a core competency. Here is why:

- **80% of ML work is data work.** Feature engineering, data validation, cohort creation, label generation — all SQL.
- **Production ML systems read from databases.** Understanding how data is stored is understanding the source of truth.
- **ML pipelines are data pipelines.** The quality of your training data determines the ceiling of your model's performance.
- **SQL scales.** Pandas falls over at 10M rows. A well-indexed SQL query handles billions.
- **Cloud warehouses (BigQuery, Snowflake, Redshift) are SQL-first.** The entire modern data stack is built on SQL.

### How SQL Connects to Real ML Systems

```
Raw Events → Databases (PostgreSQL/BigQuery)
                  ↓
         SQL: Extract + Clean + Join
                  ↓
         Feature Tables (ML-ready)
                  ↓
         Python: Model Training
                  ↓
         Predictions → Back to DB
```

Every stage in this pipeline either reads from or writes to a database — via SQL.

---

## Table of Contents

1. [SQL Fundamentals](#1-sql-fundamentals)
2. [Data Manipulation (DML)](#2-data-manipulation-dml)
3. [Table Design & Schema Basics](#3-table-design--schema-basics)
4. [Core Querying](#4-core-querying-absolute-must)
5. [Filtering & Conditions](#5-filtering--conditions)
6. [Aggregations & Grouping](#6-aggregations--grouping)
7. [Joins — Critical for Real Data](#7-joins--critical-for-real-data)
8. [Subqueries & CTEs](#8-subqueries--ctes)
9. [Window Functions — AI/Analytics Superpower](#9-window-functions--aianalytics-superpower)
10. [Data Cleaning & Transformation](#10-data-cleaning--transformation)
11. [Performance & Optimization](#11-performance--optimization)
12. [Analytical Thinking with SQL](#12-analytical-thinking-with-sql)
13. [SQL Dialects Awareness](#13-sql-dialects-awareness)
14. [SQL + Python Integration](#14-sql--python-integration)
15. [Databases Every AI Engineer Should Touch](#15-databases-every-ai-engineer-should-touch)
16. [Real-World SQL Skills](#16-real-world-sql-skills)
17. [Cross-Topic Connections](#17-cross-topic-connections)
18. [End-to-End Practical System View](#18-end-to-end-practical-system-view)
19. [Hands-On Projects](#19-hands-on-projects)
20. [Cheat Sheets](#20-cheat-sheets)
21. [Interview Preparation](#21-interview-preparation)
22. [Resources](#22-resources)

---

## 1. SQL Fundamentals

### a. Why This Matters for AI Engineering

Every ML dataset starts as raw relational data: user events, transactions, sensor readings, logs. Relational databases are how organizations store this data. Without understanding the fundamentals of how data is organized (tables, keys, relationships), you cannot effectively extract features, trace data lineage, or debug why your model is receiving corrupted input.

SQL is the universal language of structured data. You will encounter it on Day 1 at any company doing ML.

### b. Intuition (AI-Focused)

Think of a relational database as a structured warehouse. Instead of one giant spreadsheet, data is organized into multiple purpose-built tables that link together. A `users` table holds who someone is. An `events` table holds what they did. A `products` table holds what they bought.

Your job as an AI engineer is to navigate this warehouse, pull the right shelves, combine them intelligently, and produce a clean, flat feature matrix your model can consume.

### c. Minimal Theory (Only What Matters)

**Relational Database Concepts:**

| Concept | What It Is | Why It Matters in ML |
|---|---|---|
| Table | A structured grid of rows and columns | Each table is a data source you'll query |
| Row (Record) | One observation | One training example |
| Column (Field) | One attribute | One feature (or label) |
| Primary Key | Unique row identifier | Join anchor — every ML feature table needs one |
| Foreign Key | Reference to another table's PK | How you enrich data — user_id links events to profiles |
| Schema | Blueprint of a database | Tells you what data exists and how it's shaped |

**Primary Key vs. Foreign Key — The ML Context:**

```
users table            events table
-----------            -------------------------
user_id (PK) ←──────── user_id (FK)
email                  event_type
age                    timestamp
country                product_id (FK) ──────→ products table
```

In feature engineering, you will join `users` ↔ `events` ↔ `products` constantly. Understanding PKs and FKs tells you which column is the reliable join key.

### d. Practical Usage in ML

- **Label generation:** Join user actions to outcomes using PKs
- **Feature enrichment:** Attach user demographics to event logs via FK joins
- **Data validation:** Check PK uniqueness to catch data pipeline bugs
- **Deduplication:** Detect when training data has duplicate rows using PK logic

### e. Python / SQL Implementation

```sql
-- Exploring a new database: first things an AI engineer does
-- 1. List all tables (PostgreSQL)
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';

-- 2. Understand the shape of a table
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'events';

-- 3. Check row count
SELECT COUNT(*) AS total_rows FROM events;

-- 4. Spot primary key violations (data quality check)
SELECT user_id, COUNT(*) AS cnt
FROM users
GROUP BY user_id
HAVING COUNT(*) > 1;
-- If this returns rows, your "primary key" is not actually unique — bad news for joins
```

### f. Mini Use Case

You're building a churn prediction model. Before writing any Python, you need to understand:
1. Where user data lives (`users` table)
2. Where activity data lives (`events` table)
3. How they link (`user_id` foreign key)
4. What the grain of each table is (one row per user? per event?)

This schema understanding is what separates engineers who build reliable pipelines from those who accidentally train on duplicated data.

### g. Common Mistakes

- **Assuming primary keys are enforced:** Many production databases have logical PKs without database-level constraints. Always verify uniqueness explicitly.
- **Ignoring table grain:** Not knowing whether a table has one row per user per day vs. one row per event leads to fan-out joins that explode row counts.
- **Treating all NULLs the same:** NULL means "unknown" — it is not zero, not empty string. This distinction matters heavily in feature engineering.

---

## 2. Data Manipulation (DML)

### a. Why This Matters for AI Engineering

AI engineers don't just read data — they write it back. Prediction results, generated labels, feature snapshots, experiment metadata — all get written to databases. Understanding DML is essential for building complete ML pipelines.

### b. Intuition (AI-Focused)

Think of DML as the write side of your ML system. Your model outputs predictions; those predictions need to go somewhere. Your feature pipeline computes features; those features need to be stored. DML is how you close that loop.

### c. Minimal Theory

| Command | Purpose | ML Use Case |
|---|---|---|
| `INSERT` | Add new rows | Store predictions, write feature snapshots |
| `UPDATE` | Modify existing rows | Update model scores, correct labels |
| `DELETE` | Remove rows | Remove stale features, purge test data |
| `UPSERT` | Insert or update | Idempotent feature writes |
| `BEGIN/COMMIT/ROLLBACK` | Transaction control | Atomic feature table updates |

### d. Practical Usage in ML

- **Storing predictions:** After inference, insert model outputs into a `predictions` table
- **Label writing:** Write human-annotated labels back to the database
- **Idempotent pipelines:** Use UPSERT so re-running a pipeline doesn't duplicate data
- **Atomic updates:** Use transactions when updating multiple related tables together

### e. Python / SQL Implementation

```sql
-- INSERT: Writing model predictions to a table
INSERT INTO predictions (user_id, model_version, score, predicted_at)
VALUES ('user_001', 'v2.1', 0.87, NOW());

-- Bulk INSERT (more efficient for batch predictions)
INSERT INTO predictions (user_id, model_version, score, predicted_at)
VALUES
  ('user_001', 'v2.1', 0.87, NOW()),
  ('user_002', 'v2.1', 0.23, NOW()),
  ('user_003', 'v2.1', 0.65, NOW());

-- UPDATE: Refresh a feature value
UPDATE user_features
SET avg_purchase_value = 142.50,
    feature_updated_at = NOW()
WHERE user_id = 'user_001';

-- DELETE: Remove stale feature rows older than 90 days
DELETE FROM feature_snapshots
WHERE snapshot_date < NOW() - INTERVAL '90 days';

-- UPSERT: Insert prediction, update if user already has one
-- PostgreSQL syntax
INSERT INTO predictions (user_id, model_version, score, predicted_at)
VALUES ('user_001', 'v2.1', 0.91, NOW())
ON CONFLICT (user_id)
DO UPDATE SET
  score = EXCLUDED.score,
  model_version = EXCLUDED.model_version,
  predicted_at = EXCLUDED.predicted_at;

-- TRANSACTION: Atomically update features + log the update
BEGIN;

UPDATE user_features
SET churn_score = 0.87,
    last_scored_at = NOW()
WHERE user_id = 'user_001';

INSERT INTO scoring_log (user_id, model_version, score, logged_at)
VALUES ('user_001', 'v2.1', 0.87, NOW());

COMMIT;
-- If either statement fails, ROLLBACK undoes both
```

```python
# Python: Batch insert predictions using psycopg2
import psycopg2
import psycopg2.extras

conn = psycopg2.connect("postgresql://localhost/ml_db")
cur = conn.cursor()

predictions = [
    ('user_001', 'v2.1', 0.87),
    ('user_002', 'v2.1', 0.23),
    ('user_003', 'v2.1', 0.65),
]

# Use execute_values for efficient bulk insert
psycopg2.extras.execute_values(
    cur,
    """
    INSERT INTO predictions (user_id, model_version, score, predicted_at)
    VALUES %s
    ON CONFLICT (user_id) DO UPDATE SET
        score = EXCLUDED.score,
        predicted_at = EXCLUDED.predicted_at
    """,
    [(uid, ver, score) + (None,) for uid, ver, score in predictions],
    template="(%s, %s, %s, NOW())"
)

conn.commit()
cur.close()
conn.close()
```

### f. Mini Use Case

Your batch inference pipeline runs nightly. It scores 500,000 users for churn risk. You need to:
1. Write all scores back to the database (`INSERT`)
2. Not duplicate rows if the pipeline re-runs (`ON CONFLICT`)
3. Ensure both the score table and the audit log update together (`TRANSACTION`)

### g. Common Mistakes

- **No transactions on multi-table writes:** If your pipeline crashes between two related writes, you get inconsistent state — half-written features are worse than no features.
- **INSERT without UPSERT on idempotent pipelines:** Re-running a pipeline doubles your data.
- **UPDATE without WHERE:** A `UPDATE predictions SET score = 0` without a WHERE clause zeros out every row. Always double-check.

---

## 3. Table Design & Schema Basics

### a. Why This Matters for AI Engineering

As an AI engineer, you will design feature tables, staging tables, and prediction stores. Poor schema design leads to slow queries, data inconsistencies, and pipelines that break under scale. Good schema design makes your feature engineering faster, more reliable, and easier to maintain.

### b. Intuition (AI-Focused)

A well-designed schema is like a well-organized training dataset. Each table has a clear purpose, each column has a defined type and meaning, and constraints prevent bad data from ever entering the system. Garbage in, garbage out — schema design is the first line of defense.

### c. Minimal Theory

**Constraints — The Data Quality Guardrails:**

| Constraint | Purpose | ML Relevance |
|---|---|---|
| `PRIMARY KEY` | Unique, non-null identifier | Reliable join key |
| `FOREIGN KEY` | Referential integrity | Prevents orphaned records in joins |
| `UNIQUE` | No duplicates on column | Prevent duplicate feature rows |
| `NOT NULL` | Mandatory field | Force complete records |
| `CHECK` | Custom validation rule | Enforce value ranges (e.g., age > 0) |

**Normalization — When to Normalize vs. Denormalize:**

| Level | What It Means | ML Implication |
|---|---|---|
| 1NF | No repeating groups, atomic values | Baseline: one value per cell |
| 2NF | No partial dependencies on composite key | Reduces redundancy |
| 3NF | No transitive dependencies | Clean source tables |
| Denormalized | Intentional redundancy for read speed | Feature tables are often denormalized |

**Key Insight for ML:** Source tables should be normalized (clean, no redundancy). Feature tables should often be denormalized (flat, one row per entity, all features in columns) because model training reads them sequentially and needs speed, not normalization.

**Indexes:**
An index is a lookup structure the database builds alongside a table to speed up searches. Without an index on `user_id`, a query `WHERE user_id = 'abc'` scans every row. With an index, it jumps directly to the match.

- B-tree index: Default. Good for equality and range queries.
- Partial index: Index only rows matching a condition. Great for ML where you often filter by status.

### d. Practical Usage in ML

- Design `user_features` table with a PK on `user_id` and indexed join columns
- Add `CHECK` constraints to enforce label ranges (0 or 1 for binary classification)
- Use `NOT NULL` on feature columns to catch pipeline bugs early
- Denormalize feature tables for fast batch reads during training

### e. Python / SQL Implementation

```sql
-- Creating a well-designed feature table for ML
CREATE TABLE user_features (
    user_id         VARCHAR(64)     PRIMARY KEY,
    -- Behavioral features
    total_purchases INTEGER         NOT NULL DEFAULT 0,
    avg_order_value NUMERIC(10, 2)  CHECK (avg_order_value >= 0),
    days_since_last_purchase INTEGER,
    purchase_frequency_30d NUMERIC(6, 2),
    -- Categorical features
    customer_segment VARCHAR(32),
    country          VARCHAR(64),
    -- Label
    is_churned       BOOLEAN,
    -- Metadata (ALWAYS include these)
    feature_date     DATE            NOT NULL,
    created_at       TIMESTAMP       NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP       NOT NULL DEFAULT NOW(),
    -- Constraints
    UNIQUE (user_id, feature_date)  -- One feature row per user per day
);

-- Index for fast lookups during training data extraction
CREATE INDEX idx_user_features_date
    ON user_features (feature_date);

-- Partial index: only index active users (saves space, speeds up common queries)
CREATE INDEX idx_user_features_active
    ON user_features (user_id)
    WHERE is_churned = FALSE;

-- Predictions table
CREATE TABLE predictions (
    prediction_id   SERIAL          PRIMARY KEY,
    user_id         VARCHAR(64)     NOT NULL REFERENCES user_features(user_id),
    model_version   VARCHAR(32)     NOT NULL,
    churn_score     NUMERIC(5, 4)   CHECK (churn_score BETWEEN 0 AND 1),
    predicted_at    TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- Check index usage with EXPLAIN
EXPLAIN SELECT * FROM user_features WHERE feature_date = '2024-01-01';
```

**Normalization example — source vs feature tables:**

```sql
-- NORMALIZED source tables (3NF) — good for transactional writes
CREATE TABLE users (
    user_id   VARCHAR(64) PRIMARY KEY,
    email     VARCHAR(255) UNIQUE NOT NULL,
    country   VARCHAR(64)
);

CREATE TABLE orders (
    order_id   SERIAL PRIMARY KEY,
    user_id    VARCHAR(64) NOT NULL REFERENCES users(user_id),
    amount     NUMERIC(10, 2) NOT NULL,
    ordered_at TIMESTAMP NOT NULL
);

-- DENORMALIZED feature table (for ML) — flat, fast, ready for training
-- This is derived from normalized sources via SQL transformation
CREATE TABLE user_features_denorm AS
SELECT
    u.user_id,
    u.country,
    COUNT(o.order_id)          AS total_orders,
    AVG(o.amount)              AS avg_order_value,
    MAX(o.ordered_at)          AS last_order_date,
    SUM(o.amount)              AS lifetime_value
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.country;
```

### f. Mini Use Case

You're building a daily feature pipeline. Without proper schema design:
- Missing `UNIQUE (user_id, feature_date)` → pipeline re-runs create duplicate rows → your training set has duplicates → inflated evaluation metrics.
- Missing `CHECK (churn_score BETWEEN 0 AND 1)` → a bug writes scores > 1 → your dashboard shows impossible values → hours of debugging.

Schema constraints are automated data quality checks that run on every write, for free.

### g. Common Mistakes

- **No `updated_at` timestamp:** You cannot know when a row was last refreshed, making incremental pipelines impossible.
- **VARCHAR without length limits:** Unbounded strings can corrupt data or create performance issues.
- **Normalizing feature tables:** Feature tables read by ML models should be flat and denormalized. Don't make your training loop do joins.
- **Skipping indexes on join keys:** Every FK column that participates in a JOIN should have an index.

---

## 4. Core Querying (Absolute Must)

### a. Why This Matters for AI Engineering

`SELECT` queries are how you inspect data, validate hypotheses, extract features, and debug pipelines. These are the most executed SQL statements in any ML workflow. Mastery here is non-negotiable.

### b. Intuition (AI-Focused)

A SELECT query is a question you ask the database. "Give me all users who signed up in January, show me their country and age, sorted by age, only the first 1000." Learning to compose these questions precisely is a core data engineering skill.

### c. Minimal Theory

**The Logical Order of SQL Execution (Critical for Understanding Behavior):**

```
1. FROM         → Which tables?
2. JOIN         → How to combine them?
3. WHERE        → Which rows to keep?
4. GROUP BY     → How to group?
5. HAVING       → Which groups to keep?
6. SELECT       → Which columns to return?
7. DISTINCT     → Remove duplicates?
8. ORDER BY     → Sort how?
9. LIMIT/OFFSET → How many rows?
```

This order is counterintuitive (SELECT is written first but executed 6th). Understanding this explains why you can't use column aliases defined in SELECT inside WHERE.

### d. Practical Usage in ML

- `SELECT` specific columns (not `SELECT *`) for efficient data extraction
- `DISTINCT` to find unique values in categorical features
- `ORDER BY` + `LIMIT` to inspect top/bottom records
- `LIMIT/OFFSET` for pagination when sampling large datasets

### e. Python / SQL Implementation

```sql
-- Basic feature extraction query
SELECT
    user_id,
    country,
    age,
    signup_date,
    is_premium
FROM users
WHERE signup_date >= '2024-01-01'
  AND age IS NOT NULL
ORDER BY signup_date DESC
LIMIT 1000;

-- DISTINCT: find all unique customer segments in your data
SELECT DISTINCT customer_segment
FROM user_features
ORDER BY customer_segment;

-- Aliases: make column names ML-friendly
SELECT
    user_id                                     AS entity_id,
    DATEDIFF(NOW(), last_login_date)            AS days_inactive,
    total_purchases * avg_order_value           AS estimated_ltv,
    CASE WHEN is_premium THEN 1 ELSE 0 END      AS is_premium_flag
FROM users;

-- LIMIT/OFFSET: page through a large dataset for sampling
-- Page 1 (rows 1-1000)
SELECT * FROM events ORDER BY event_id LIMIT 1000 OFFSET 0;
-- Page 2 (rows 1001-2000)
SELECT * FROM events ORDER BY event_id LIMIT 1000 OFFSET 1000;
```

```python
# Running a parameterized SELECT from Python
import psycopg2
import pandas as pd

conn = psycopg2.connect("postgresql://localhost/ml_db")

query = """
    SELECT
        user_id,
        country,
        age,
        total_purchases,
        is_churned
    FROM user_features
    WHERE feature_date = %s
      AND country = ANY(%s)
    ORDER BY user_id
"""

# Read directly into pandas DataFrame
df = pd.read_sql_query(
    query,
    conn,
    params=('2024-01-01', ['US', 'UK', 'CA'])
)

print(df.shape)  # (rows, cols)
print(df.dtypes)  # Always check dtypes when loading from SQL
conn.close()
```

### f. Mini Use Case

Before training a churn model, you need to understand your label distribution:

```sql
SELECT
    is_churned,
    COUNT(*)                                         AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM user_features
WHERE feature_date = '2024-01-01'
GROUP BY is_churned;
```

If 95% of users are `is_churned = FALSE`, you have class imbalance — a critical insight that shapes your entire modeling approach.

### g. Common Mistakes

- **`SELECT *` in production:** Selects all columns including large blobs, slows down queries, breaks when schema changes.
- **Using column aliases in WHERE:** `SELECT age * 2 AS double_age FROM users WHERE double_age > 40` fails — WHERE executes before SELECT.
- **Forgetting ORDER BY with LIMIT:** Without ORDER BY, LIMIT returns arbitrary rows — not necessarily the most recent or most relevant.

---

## 5. Filtering & Conditions

### a. Why This Matters for AI Engineering

Most ML datasets are slices of a larger database. You don't train on all data — you filter by date range, user segment, event type, quality criteria. Mastering WHERE clauses is mastering dataset construction.

### b. Intuition (AI-Focused)

Filtering is how you define your training population. Every WHERE condition is a decision about which observations to include. These decisions directly affect what your model learns. A wrong filter can introduce bias, leak future information, or exclude valid training examples.

### c. Minimal Theory

**Operator Reference:**

| Operator | Example | ML Use Case |
|---|---|---|
| `=` | `country = 'US'` | Filter to specific segment |
| `!=` or `<>` | `status != 'test'` | Exclude test accounts |
| `BETWEEN` | `age BETWEEN 18 AND 65` | Clip feature range |
| `IN` | `event_type IN ('purchase', 'add_to_cart')` | Filter to relevant events |
| `LIKE` | `email LIKE '%@company.com'` | Identify internal users |
| `IS NULL` | `churn_date IS NULL` | Filter active users |
| `CASE WHEN` | `CASE WHEN score > 0.5 THEN 'high' ELSE 'low' END` | Bucketing features |

**NULL behavior — critical trap:**

`NULL = NULL` evaluates to `NULL`, not `TRUE`. Always use `IS NULL` or `IS NOT NULL`, never `= NULL`.

### d. Practical Usage in ML

- Filter training data to a specific date range to prevent temporal leakage
- Exclude bot/test accounts from training data
- Create binary labels using `CASE WHEN`
- Validate feature ranges with `BETWEEN`

### e. Python / SQL Implementation

```sql
-- Building a clean training dataset with careful filtering
SELECT
    u.user_id,
    u.age,
    u.country,
    u.signup_date,
    -- Create binary label
    CASE
        WHEN u.churned_at IS NOT NULL
         AND u.churned_at <= '2024-01-01'
        THEN 1
        ELSE 0
    END AS label,
    -- Bucket a continuous feature
    CASE
        WHEN u.age < 25             THEN 'young'
        WHEN u.age BETWEEN 25 AND 45 THEN 'mid'
        WHEN u.age > 45             THEN 'senior'
        ELSE 'unknown'
    END AS age_group
FROM users u
WHERE
    -- Training window: users who signed up before observation date
    u.signup_date < '2023-07-01'
    -- Exclude test/internal accounts
    AND u.email NOT LIKE '%@internal.com'
    AND u.is_bot = FALSE
    -- Exclude users with missing critical features
    AND u.age IS NOT NULL
    AND u.country IS NOT NULL
    -- Filter to valid age range
    AND u.age BETWEEN 18 AND 100
    -- Only users in target markets
    AND u.country IN ('US', 'UK', 'CA', 'AU');

-- Using ILIKE for case-insensitive matching (PostgreSQL)
SELECT * FROM users
WHERE LOWER(email) LIKE '%test%'
   OR LOWER(email) ILIKE '%demo%';

-- NULL handling: find rows with missing features
SELECT
    user_id,
    COUNT(*) FILTER (WHERE age IS NULL)            AS missing_age,
    COUNT(*) FILTER (WHERE country IS NULL)        AS missing_country,
    COUNT(*) FILTER (WHERE avg_order_value IS NULL) AS missing_aov
FROM user_features
GROUP BY user_id
HAVING COUNT(*) FILTER (WHERE age IS NULL) > 0;
```

### f. Mini Use Case

Training a model on 2023 data to predict 2024 churn. Wrong filter:
```sql
-- WRONG: Includes users who churned after the training cutoff
WHERE churned_at IS NOT NULL
```

Right filter:
```sql
-- CORRECT: Only label as churned if it happened within the training window
WHERE churned_at < '2024-01-01'
```

The wrong filter leaks future information — your model will appear to work great in training but fail in production.

### g. Common Mistakes

- **`WHERE col = NULL`:** Never works. Use `IS NULL`.
- **Date filter mistakes:** `WHERE date >= '2024-01'` may not parse correctly. Always use full ISO dates: `'2024-01-01'`.
- **Implicit bias from filters:** Filtering to "users with at least 1 purchase" excludes never-purchasers. If your model then predicts on never-purchasers, you've trained on the wrong population.
- **OR vs AND logic errors:** `WHERE country = 'US' OR country = 'UK' AND age > 30` — AND has higher precedence than OR. Use parentheses: `WHERE (country = 'US' OR country = 'UK') AND age > 30`.

---

## 6. Aggregations & Grouping

### a. Why This Matters for AI Engineering

Feature engineering is almost entirely aggregation. "How many purchases did this user make in the last 30 days?" — that's an aggregation. "What is the average order value per customer?" — aggregation. "How many unique products has this user viewed?" — aggregation. `GROUP BY` + aggregate functions are the primary tools for turning raw event data into features.

### b. Intuition (AI-Focused)

Raw event data has one row per event — thousands of rows per user. ML models need one row per user — one feature vector. Aggregation is how you collapse many events into one summary per entity. This is called "entity-level aggregation" and it's the core transformation in feature engineering.

### c. Minimal Theory

**Aggregate Functions:**

| Function | Returns | ML Feature Example |
|---|---|---|
| `COUNT(*)` | Total rows | Number of sessions |
| `COUNT(DISTINCT col)` | Unique values | Unique products viewed |
| `SUM(col)` | Total value | Total spend |
| `AVG(col)` | Mean | Average session duration |
| `MIN(col)` | Minimum | First purchase date |
| `MAX(col)` | Maximum | Last login date |
| `STDDEV(col)` | Standard deviation | Spend variability |

**WHERE vs HAVING:**

```
WHERE  → filters rows BEFORE grouping   (operates on individual rows)
HAVING → filters groups AFTER grouping  (operates on aggregated results)
```

You cannot use aggregate functions in WHERE. You must use HAVING.

### d. Practical Usage in ML

- Build the feature table from raw events using GROUP BY
- Filter to users with enough data using HAVING (minimum activity threshold)
- Create count, sum, mean, max features from behavioral data

### e. Python / SQL Implementation

```sql
-- Core feature engineering query: events → user features
SELECT
    user_id,

    -- Count features
    COUNT(*)                                           AS total_events,
    COUNT(DISTINCT session_id)                         AS total_sessions,
    COUNT(DISTINCT product_id)                         AS unique_products_viewed,

    -- Purchase-specific counts
    COUNT(*) FILTER (WHERE event_type = 'purchase')    AS total_purchases,
    COUNT(*) FILTER (WHERE event_type = 'add_to_cart') AS total_add_to_carts,
    COUNT(*) FILTER (WHERE event_type = 'refund')      AS total_refunds,

    -- Monetary features
    SUM(CASE WHEN event_type = 'purchase' THEN amount ELSE 0 END)    AS total_revenue,
    AVG(CASE WHEN event_type = 'purchase' THEN amount END)           AS avg_order_value,
    MAX(CASE WHEN event_type = 'purchase' THEN amount END)           AS max_order_value,

    -- Temporal features
    MIN(event_timestamp)                               AS first_event_date,
    MAX(event_timestamp)                               AS last_event_date,
    EXTRACT(DAY FROM (MAX(event_timestamp) - MIN(event_timestamp))) AS customer_age_days,

    -- Derived rate features
    COUNT(*) FILTER (WHERE event_type = 'purchase')::FLOAT
        / NULLIF(COUNT(DISTINCT session_id), 0)        AS purchase_rate_per_session,

    -- Variability (spread of behavior)
    STDDEV(CASE WHEN event_type = 'purchase' THEN amount END) AS order_value_stddev

FROM events
WHERE event_timestamp >= '2023-01-01'
  AND event_timestamp <  '2024-01-01'   -- Training window
GROUP BY user_id
HAVING
    COUNT(*) >= 5                        -- Minimum 5 events (enough signal)
    AND COUNT(*) FILTER (WHERE event_type = 'purchase') > 0  -- At least 1 purchase
ORDER BY total_revenue DESC;
```

```sql
-- GROUP BY with ROLLUP: subtotals by segment (useful for validation)
SELECT
    country,
    customer_segment,
    COUNT(*)          AS user_count,
    AVG(churn_score)  AS avg_churn_score
FROM user_features
GROUP BY ROLLUP(country, customer_segment)
ORDER BY country, customer_segment;
```

### f. Mini Use Case

You have 100M raw events. You need a training dataset with one row per user with 20 features. This query transforms 100M rows → ~1M user rows:

```sql
-- The classic feature aggregation pipeline
CREATE TABLE user_features_v1 AS
SELECT
    user_id,
    COUNT(*)                                                        AS event_count_90d,
    COUNT(DISTINCT DATE(event_timestamp))                           AS active_days_90d,
    SUM(CASE WHEN event_type = 'purchase' THEN amount ELSE 0 END)  AS revenue_90d,
    AVG(CASE WHEN event_type = 'purchase' THEN amount END)         AS aov_90d,
    MAX(event_timestamp)                                            AS last_active_at
FROM events
WHERE event_timestamp >= NOW() - INTERVAL '90 days'
GROUP BY user_id;
```

### g. Common Mistakes

- **`COUNT(col)` vs `COUNT(*)`:** `COUNT(col)` ignores NULLs. `COUNT(*)` counts all rows. Use `COUNT(DISTINCT col)` for unique values.
- **Dividing by zero in rate calculations:** `SUM(purchases) / COUNT(sessions)` fails if sessions = 0. Use `NULLIF(COUNT(sessions), 0)`.
- **Forgetting HAVING to filter sparse users:** Training on users with 1 event produces noisy features. Add `HAVING COUNT(*) >= N`.
- **AVG including NULL:** If 30% of your amounts are NULL, `AVG(amount)` averages only the non-null ones — which may not be what you want.

---

## 7. Joins — Critical for Real Data

### a. Why This Matters for AI Engineering

No single table has all the features you need. User demographics are in `users`. Purchase history is in `orders`. Product metadata is in `products`. Session data is in `events`. Joining these tables together is how you build a rich feature set. Mastering joins is mastering feature enrichment.

### b. Intuition (AI-Focused)

Joins are how you answer multi-dimensional questions: "Show me all users FROM the users table, enriched WITH their order history, and the products they bought." Each join adds new dimensions (features) to your entity.

Think of INNER JOIN as "intersection" and LEFT JOIN as "left table plus whatever matches from the right."

### c. Minimal Theory

**Visual Join Reference:**

```
Table A: users           Table B: orders
user_id | name           user_id | amount
--------|------          --------|-------
001     | Alice          001     | $50
002     | Bob            001     | $80
003     | Carol          004     | $30   ← no matching user

INNER JOIN  → 001, 001 (only matches)
LEFT JOIN   → 001, 001, 002(NULL), 003(NULL)  (all A, matches from B)
RIGHT JOIN  → 001, 001, 004(NULL)              (all B, matches from A)
FULL OUTER  → 001, 001, 002(NULL), 003(NULL), 004(NULL)  (everything)
```

**Join Types for ML:**

| Join Type | When to Use in ML |
|---|---|
| INNER JOIN | Only want entities with data in both tables |
| LEFT JOIN | Keep all entities, add features where available |
| FULL OUTER | Data reconciliation, finding mismatches |
| CROSS JOIN | Generate combinations (feature interaction pairs) |
| SELF JOIN | Sequential comparisons (compare current to previous row) |

### d. Practical Usage in ML

- LEFT JOIN to enrich user features with optional metadata
- INNER JOIN to filter to users who appear in both training and feature tables
- SELF JOIN to compute session-to-session changes
- Multiple JOINs to build a flat feature matrix from normalized source tables

### e. Python / SQL Implementation

```sql
-- Building a feature-rich training dataset via multiple JOINs
SELECT
    u.user_id,
    u.age,
    u.country,
    u.signup_date,

    -- Features from orders table
    COALESCE(o.total_orders, 0)      AS total_orders,
    COALESCE(o.total_revenue, 0)     AS total_revenue,
    o.avg_order_value,
    o.last_order_date,

    -- Features from products table (via orders)
    COALESCE(p.unique_categories, 0) AS unique_product_categories,

    -- Label from churn table
    CASE WHEN c.churned_at IS NOT NULL THEN 1 ELSE 0 END AS is_churned

FROM users u

-- LEFT JOIN: keep all users, add order features where available
LEFT JOIN (
    SELECT
        user_id,
        COUNT(DISTINCT order_id)  AS total_orders,
        SUM(amount)               AS total_revenue,
        AVG(amount)               AS avg_order_value,
        MAX(ordered_at)           AS last_order_date
    FROM orders
    WHERE ordered_at < '2024-01-01'  -- No future data leakage
    GROUP BY user_id
) o ON u.user_id = o.user_id

-- LEFT JOIN to product diversity features
LEFT JOIN (
    SELECT
        o.user_id,
        COUNT(DISTINCT p.category) AS unique_categories
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    WHERE o.ordered_at < '2024-01-01'
    GROUP BY o.user_id
) p ON u.user_id = p.user_id

-- LEFT JOIN for churn label
LEFT JOIN churn_events c ON u.user_id = c.user_id

WHERE u.signup_date < '2023-07-01'  -- Established users only;

-- Self JOIN: compare a user's current behavior to their previous month
SELECT
    curr.user_id,
    curr.month,
    curr.revenue          AS current_revenue,
    prev.revenue          AS prev_revenue,
    curr.revenue - COALESCE(prev.revenue, 0) AS revenue_change
FROM monthly_user_stats curr
LEFT JOIN monthly_user_stats prev
    ON curr.user_id = prev.user_id
    AND prev.month = curr.month - INTERVAL '1 month';
```

**Detecting and preventing duplicate rows from joins:**

```sql
-- DANGER: Fan-out join (orders has multiple rows per user → cartesian explosion)
-- This WRONG query multiplies rows:
SELECT u.user_id, COUNT(*) as order_count
FROM users u
JOIN orders o ON u.user_id = o.user_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY u.user_id;
-- If an order has 3 items, the user appears 3x per order → wrong count

-- CORRECT: Aggregate first, then join
WITH order_summary AS (
    SELECT user_id, COUNT(DISTINCT order_id) AS order_count
    FROM orders
    GROUP BY user_id
),
item_summary AS (
    SELECT o.user_id, COUNT(*) AS item_count
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.user_id
)
SELECT u.user_id, os.order_count, its.item_count
FROM users u
LEFT JOIN order_summary os ON u.user_id = os.user_id
LEFT JOIN item_summary its ON u.user_id = its.user_id;
```

### f. Mini Use Case

You want to add product category diversity as a feature. Without joining `orders` to `products`, you can't compute it. With a well-structured join chain: `users → orders → products → COUNT(DISTINCT category)`, you get it in one query.

### g. Common Mistakes

- **Fan-out joins:** Joining a one-to-many table before aggregating inflates row counts. Always aggregate in a CTE or subquery before joining.
- **INNER JOIN when you need LEFT JOIN:** If a user has no orders, INNER JOIN drops them from the training set entirely — invisible data loss.
- **Missing join conditions:** A forgotten ON clause creates a CROSS JOIN (every row × every row) — your query runs for hours and returns billions of rows.
- **Joining on non-indexed columns:** Always ensure join keys are indexed.

---

## 8. Subqueries & CTEs

### a. Why This Matters for AI Engineering

Complex feature engineering queries can span dozens of steps. Without structure, they become unreadable and unmaintainable. CTEs (Common Table Expressions) let you break a complex transformation into named, logical steps — like functions in Python. This is how production-quality SQL is written.

### b. Intuition (AI-Focused)

A CTE is like a named intermediate result. Instead of nesting 5 subqueries inside each other (unreadable), you define each step with a name and build on it. This makes your feature pipeline self-documenting.

### c. Minimal Theory

**Subquery Types:**

| Type | Where Used | Example |
|---|---|---|
| In FROM | Virtual table | `SELECT * FROM (SELECT ...) sub` |
| In WHERE | Filter using derived value | `WHERE user_id IN (SELECT ...)` |
| In SELECT | Column-level computation | `SELECT (SELECT AVG(...)) AS avg_val` |
| Correlated | References outer query | Row-by-row, often slow |

**CTE vs Subquery — When to Use Which:**

| Use CTE | Use Subquery |
|---|---|
| Multi-step logic | Simple one-off filter |
| Reusing the same derived table | Used only once |
| Readability is priority | Performance is critical (optimizer may not materialize CTE) |
| Recursive logic | Non-recursive |

### d. Practical Usage in ML

- Break feature engineering into readable named steps
- Compute intermediate aggregations and reuse them
- Create cohort definitions as CTEs then join to features
- Use recursive CTEs for hierarchical data (category trees, org charts)

### e. Python / SQL Implementation

```sql
-- Multi-step feature pipeline using CTEs
-- Each CTE is one logical transformation step

WITH
-- Step 1: Define the user cohort (who are we analyzing?)
user_cohort AS (
    SELECT user_id, signup_date, country
    FROM users
    WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31'
      AND is_bot = FALSE
      AND country IN ('US', 'UK', 'CA')
),

-- Step 2: Compute purchase behavior in 90-day window
purchase_features AS (
    SELECT
        o.user_id,
        COUNT(DISTINCT o.order_id)    AS order_count_90d,
        SUM(o.amount)                 AS revenue_90d,
        AVG(o.amount)                 AS aov_90d,
        MAX(o.ordered_at)             AS last_order_date,
        MIN(o.ordered_at)             AS first_order_date
    FROM orders o
    INNER JOIN user_cohort uc ON o.user_id = uc.user_id
    WHERE o.ordered_at >= '2023-10-01'
      AND o.ordered_at <  '2024-01-01'
    GROUP BY o.user_id
),

-- Step 3: Compute engagement features (browsing)
engagement_features AS (
    SELECT
        e.user_id,
        COUNT(DISTINCT DATE(e.event_timestamp))  AS active_days_90d,
        COUNT(*)                                 AS total_events_90d,
        COUNT(*) FILTER (WHERE e.event_type = 'search') AS search_count
    FROM events e
    INNER JOIN user_cohort uc ON e.user_id = uc.user_id
    WHERE e.event_timestamp >= '2023-10-01'
      AND e.event_timestamp <  '2024-01-01'
    GROUP BY e.user_id
),

-- Step 4: Determine churn label
churn_labels AS (
    SELECT
        user_id,
        1 AS is_churned
    FROM churn_events
    WHERE churned_at >= '2024-01-01'
      AND churned_at <  '2024-04-01'  -- Predict churn in Q1 2024
),

-- Step 5: Assemble final feature matrix
feature_matrix AS (
    SELECT
        uc.user_id,
        uc.country,
        EXTRACT(DAYS FROM NOW() - uc.signup_date)    AS account_age_days,

        -- Purchase features (COALESCE handles users with no purchases)
        COALESCE(pf.order_count_90d, 0)              AS order_count_90d,
        COALESCE(pf.revenue_90d, 0)                  AS revenue_90d,
        pf.aov_90d,
        EXTRACT(DAYS FROM NOW() - pf.last_order_date) AS days_since_last_order,

        -- Engagement features
        COALESCE(ef.active_days_90d, 0)              AS active_days_90d,
        COALESCE(ef.total_events_90d, 0)             AS total_events_90d,

        -- Derived features
        COALESCE(pf.order_count_90d, 0)::FLOAT
            / NULLIF(ef.active_days_90d, 0)          AS orders_per_active_day,

        -- Label
        COALESCE(cl.is_churned, 0)                   AS is_churned

    FROM user_cohort uc
    LEFT JOIN purchase_features  pf ON uc.user_id = pf.user_id
    LEFT JOIN engagement_features ef ON uc.user_id = ef.user_id
    LEFT JOIN churn_labels        cl ON uc.user_id = cl.user_id
)

-- Final SELECT
SELECT * FROM feature_matrix
WHERE order_count_90d > 0  -- Only users with purchase history
ORDER BY user_id;
```

**Recursive CTE (conceptual — for hierarchical data like category trees):**

```sql
-- Recursive CTE: traverse a product category hierarchy
WITH RECURSIVE category_tree AS (
    -- Base case: top-level categories
    SELECT category_id, parent_id, name, 1 AS depth
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    -- Recursive case: children of current level
    SELECT c.category_id, c.parent_id, c.name, ct.depth + 1
    FROM categories c
    INNER JOIN category_tree ct ON c.parent_id = ct.category_id
)
SELECT * FROM category_tree ORDER BY depth, category_id;
-- Use case: flatten category hierarchy to assign top-level category as a feature
```

### f. Mini Use Case

Without CTEs, a 5-step feature pipeline is one giant nested query — 200 lines of SQL that no one can debug. With CTEs, each step is named and testable. You can `SELECT * FROM purchase_features` to inspect that step alone. This is the difference between maintainable and unmaintainable ML pipelines.

### g. Common Mistakes

- **Correlated subqueries in SELECT for large tables:** A correlated subquery runs once per row — on 1M rows, it runs 1M times. Rewrite as a JOIN.
- **CTEs not being optimized:** In PostgreSQL, CTEs are optimization fences (pre-12). In PG12+ they're inlined by default. In BigQuery, CTEs are always materialized — know your dialect.
- **Overcomplicating with subqueries when a JOIN works:** `WHERE user_id IN (SELECT user_id FROM ...)` is often slower than an equivalent JOIN.

---

## 9. Window Functions — AI/Analytics Superpower

### a. Why This Matters for AI Engineering

Window functions are the most powerful SQL tool for AI engineers. They let you compute features like "what was this user's rank by purchase amount last month?", "how much did this user spend in the rolling 7-day window?", "what was the previous session's duration?". These time-aware, rank-aware, window-aware features are impossible to express with GROUP BY alone and are critical for behavioral ML models.

### b. Intuition (AI-Focused)

Standard aggregation collapses many rows into one. Window functions compute an aggregation over a window of rows but return a result for each row individually. Think of it as "for each row, look at a window of related rows and compute something about that window."

```
Regular GROUP BY:
Row 1: user_001, Jan, $50  →  Collapsed to:  user_001, total=$200
Row 2: user_001, Feb, $80
Row 3: user_001, Mar, $70

Window function:
Row 1: user_001, Jan, $50,  running_total=$50,   rank=3
Row 2: user_001, Feb, $80,  running_total=$130,  rank=1
Row 3: user_001, Mar, $70,  running_total=$200,  rank=2
```

### c. Minimal Theory

**Window Function Anatomy:**

```sql
function_name(column) OVER (
    PARTITION BY partition_column  -- Group rows (like GROUP BY but rows preserved)
    ORDER BY order_column          -- Order within partition
    ROWS/RANGE frame_spec          -- Size of the window
)
```

**Key Window Functions:**

| Function | What It Computes | ML Feature Example |
|---|---|---|
| `ROW_NUMBER()` | Unique sequential rank | First/last event identifier |
| `RANK()` | Rank with gaps for ties | Customer rank by revenue |
| `DENSE_RANK()` | Rank without gaps | Percentile bucket |
| `LAG(col, n)` | Previous n-th row value | Last month's purchases |
| `LEAD(col, n)` | Next n-th row value | Next event type |
| `SUM() OVER(...)` | Running/window total | Cumulative revenue |
| `AVG() OVER(...)` | Moving average | 7-day rolling avg |
| `NTILE(n)` | Divide into n buckets | Quartile assignment |

### d. Practical Usage in ML

- **Time-series features:** Rolling averages, cumulative sums, lag features
- **Ranking features:** User's rank among peers
- **Sequence features:** Previous event type, time since last event
- **Percentile features:** Which decile a user falls in

### e. Python / SQL Implementation

```sql
-- Window functions for rich feature engineering

WITH user_monthly_activity AS (
    SELECT
        user_id,
        DATE_TRUNC('month', ordered_at)   AS month,
        COUNT(DISTINCT order_id)           AS order_count,
        SUM(amount)                        AS monthly_revenue
    FROM orders
    GROUP BY user_id, DATE_TRUNC('month', ordered_at)
)

SELECT
    user_id,
    month,
    order_count,
    monthly_revenue,

    -- Lag features (previous period values — critical for time-series ML)
    LAG(monthly_revenue, 1) OVER (
        PARTITION BY user_id ORDER BY month
    ) AS prev_month_revenue,

    LAG(monthly_revenue, 2) OVER (
        PARTITION BY user_id ORDER BY month
    ) AS prev_2month_revenue,

    -- Month-over-month change
    monthly_revenue - LAG(monthly_revenue, 1) OVER (
        PARTITION BY user_id ORDER BY month
    ) AS revenue_mom_change,

    -- Running total (cumulative revenue per user)
    SUM(monthly_revenue) OVER (
        PARTITION BY user_id
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue,

    -- 3-month rolling average (smoothed trend feature)
    AVG(monthly_revenue) OVER (
        PARTITION BY user_id
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS revenue_3mo_rolling_avg,

    -- Rank among all users for this month
    RANK() OVER (
        PARTITION BY month
        ORDER BY monthly_revenue DESC
    ) AS monthly_revenue_rank,

    -- Percentile within month (0-100)
    NTILE(10) OVER (
        PARTITION BY month
        ORDER BY monthly_revenue
    ) AS revenue_decile,

    -- Row number per user (identify first, second, nth event)
    ROW_NUMBER() OVER (
        PARTITION BY user_id
        ORDER BY month
    ) AS user_month_sequence

FROM user_monthly_activity
ORDER BY user_id, month;
```

```sql
-- Event sequence features: time between events (critical for session modeling)
SELECT
    user_id,
    event_type,
    event_timestamp,

    -- Time since previous event (seconds)
    EXTRACT(EPOCH FROM (
        event_timestamp - LAG(event_timestamp) OVER (
            PARTITION BY user_id ORDER BY event_timestamp
        )
    )) AS seconds_since_prev_event,

    -- Previous event type (sequence modeling feature)
    LAG(event_type) OVER (
        PARTITION BY user_id ORDER BY event_timestamp
    ) AS prev_event_type,

    -- Next event type (label for sequence prediction)
    LEAD(event_type) OVER (
        PARTITION BY user_id ORDER BY event_timestamp
    ) AS next_event_type,

    -- Session number (new session if gap > 30 minutes)
    SUM(
        CASE WHEN EXTRACT(EPOCH FROM (
            event_timestamp - LAG(event_timestamp) OVER (
                PARTITION BY user_id ORDER BY event_timestamp
            )
        )) > 1800 THEN 1 ELSE 0 END
    ) OVER (
        PARTITION BY user_id ORDER BY event_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) + 1 AS session_number

FROM events
WHERE user_id = 'user_001'
ORDER BY event_timestamp;
```

```sql
-- Cohort analysis using ROW_NUMBER (find first purchase month)
WITH first_purchase AS (
    SELECT
        user_id,
        MIN(DATE_TRUNC('month', ordered_at)) AS cohort_month
    FROM orders
    GROUP BY user_id
),
monthly_stats AS (
    SELECT
        o.user_id,
        DATE_TRUNC('month', o.ordered_at)        AS activity_month,
        fp.cohort_month,
        -- Months since first purchase
        EXTRACT(MONTH FROM AGE(
            DATE_TRUNC('month', o.ordered_at),
            fp.cohort_month
        ))                                        AS months_since_first_purchase,
        SUM(o.amount)                             AS revenue
    FROM orders o
    JOIN first_purchase fp ON o.user_id = fp.user_id
    GROUP BY o.user_id, DATE_TRUNC('month', o.ordered_at), fp.cohort_month
)
SELECT
    cohort_month,
    months_since_first_purchase,
    COUNT(DISTINCT user_id)  AS active_users,
    SUM(revenue)             AS total_revenue,
    AVG(revenue)             AS avg_revenue_per_user
FROM monthly_stats
GROUP BY cohort_month, months_since_first_purchase
ORDER BY cohort_month, months_since_first_purchase;
```

### f. Mini Use Case

Building a churn prediction model? One of the strongest features is "trend in purchasing behavior." With window functions:

```sql
-- Detect declining purchase trend — a strong churn signal
SELECT
    user_id,
    AVG(monthly_revenue) OVER (
        PARTITION BY user_id ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                              AS revenue_3mo_avg,
    AVG(monthly_revenue) OVER (
        PARTITION BY user_id ORDER BY month
        ROWS BETWEEN 5 PRECEDING AND 3 PRECEDING
    )                                              AS revenue_prev_3mo_avg,
    -- Trend: negative = declining
    AVG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    - AVG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month ROWS BETWEEN 5 PRECEDING AND 3 PRECEDING)
                                                   AS revenue_trend
FROM user_monthly_activity;
```

A negative `revenue_trend` is one of the strongest churn predictors. You can't compute this with GROUP BY.

### g. Common Mistakes

- **Forgetting PARTITION BY:** Without it, the window spans the entire table — ranks compare across all users instead of within each user.
- **Confusing ROW_NUMBER and RANK:** RANK gives ties the same number and skips the next rank (1, 1, 3). DENSE_RANK doesn't skip (1, 1, 2). ROW_NUMBER is always unique.
- **Using window functions in WHERE:** They can't appear in WHERE — use a CTE or subquery.
- **ROWS vs RANGE:** `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` is physical rows. `RANGE` is value-based. For time windows, ROWS is usually more predictable.

---

## 10. Data Cleaning & Transformation

### a. Why This Matters for AI Engineering

Real-world data is messy. Strings have inconsistent casing. Dates are stored as text. Numbers are stored as strings. IDs have leading/trailing spaces. Missing values are represented as empty strings, "N/A", "null", or actual NULL. Data cleaning in SQL is your first line of defense — if you clean at source, every downstream system benefits.

### b. Intuition (AI-Focused)

Cleaning data before it reaches your model is always better than cleaning it in Python. SQL transformations run at database scale, are version-controllable, and can be scheduled. Think of SQL cleaning as building a reliable data preprocessing layer that every ML pipeline shares.

### c. Minimal Theory

**Key Transformation Functions:**

| Category | Function | Example |
|---|---|---|
| String | `LOWER(col)` | Normalize email |
| String | `TRIM(col)` | Remove leading/trailing spaces |
| String | `SUBSTRING(col, start, len)` | Extract area code |
| String | `REPLACE(col, old, new)` | Clean special characters |
| Date | `DATE_TRUNC('month', ts)` | Month-level grouping |
| Date | `EXTRACT(DOW FROM ts)` | Day-of-week feature |
| Date | `AGE(ts1, ts2)` | Time difference |
| Type | `CAST(col AS INTEGER)` | Convert string to number |
| Type | `col::FLOAT` | PostgreSQL cast shorthand |
| Null | `COALESCE(col, default)` | Fill missing values |
| Null | `NULLIF(col, value)` | Replace value with NULL |

### d. Practical Usage in ML

- Standardize categorical values before encoding
- Extract date features (day of week, month, hour) from timestamps
- Cast string numbers to numeric types
- Fill missing values with sensible defaults
- Detect and handle sentinel values (e.g., -999 used as "missing")

### e. Python / SQL Implementation

```sql
-- Comprehensive data cleaning query for ML preparation
SELECT
    -- String normalization
    LOWER(TRIM(email))                           AS email_clean,
    INITCAP(TRIM(first_name))                    AS first_name_clean,
    UPPER(TRIM(country_code))                    AS country_code_clean,

    -- Handle inconsistent category values
    CASE
        WHEN LOWER(TRIM(gender)) IN ('m', 'male', 'man')     THEN 'M'
        WHEN LOWER(TRIM(gender)) IN ('f', 'female', 'woman') THEN 'F'
        ELSE 'Unknown'
    END AS gender_clean,

    -- Type casting (safe cast with fallback)
    CASE
        WHEN age ~ '^\d+$' THEN age::INTEGER
        ELSE NULL
    END AS age_int,

    -- Numeric cleaning: replace sentinel values with NULL
    NULLIF(credit_score, -999)                   AS credit_score_clean,
    NULLIF(income, 0)                            AS income_clean,

    -- Null filling with defaults
    COALESCE(country, 'Unknown')                 AS country,
    COALESCE(age::INTEGER, 0)                    AS age_filled,

    -- Date feature extraction
    EXTRACT(YEAR  FROM created_at)               AS signup_year,
    EXTRACT(MONTH FROM created_at)               AS signup_month,
    EXTRACT(DOW   FROM created_at)               AS signup_day_of_week,  -- 0=Sunday
    EXTRACT(HOUR  FROM created_at)               AS signup_hour,

    -- Date truncation for grouping
    DATE_TRUNC('week',  created_at)              AS signup_week,
    DATE_TRUNC('month', created_at)              AS signup_month_trunc,

    -- Age in days (temporal feature)
    EXTRACT(DAYS FROM NOW() - created_at)        AS account_age_days,

    -- String extraction
    SUBSTRING(phone FROM 1 FOR 3)                AS area_code,
    SPLIT_PART(email, '@', 2)                    AS email_domain,

    -- Normalization: min-max scaling using subquery
    (amount - min_amount) / NULLIF(max_amount - min_amount, 0) AS amount_normalized

FROM users
CROSS JOIN (
    SELECT MIN(amount) AS min_amount, MAX(amount) AS max_amount
    FROM users
) stats
WHERE created_at IS NOT NULL;

-- Detecting data quality issues before cleaning
SELECT
    COUNT(*)                                                          AS total_rows,
    COUNT(*) FILTER (WHERE email IS NULL OR email = '')              AS missing_email,
    COUNT(*) FILTER (WHERE age IS NULL OR age < 0 OR age > 120)     AS invalid_age,
    COUNT(*) FILTER (WHERE country IS NULL)                          AS missing_country,
    COUNT(*) FILTER (WHERE created_at > NOW())                       AS future_dates,
    COUNT(DISTINCT user_id) - COUNT(*)                               AS duplicate_rows
FROM users;
```

### f. Mini Use Case

Your `country` column has values: `'US'`, `'United States'`, `'USA'`, `'united states'`, `'us'`. Before encoding this as a categorical feature, you must standardize it:

```sql
UPDATE users
SET country = CASE
    WHEN LOWER(TRIM(country)) IN ('us', 'usa', 'united states', 'u.s.a', 'u.s.')
        THEN 'US'
    WHEN LOWER(TRIM(country)) IN ('uk', 'united kingdom', 'great britain', 'gb')
        THEN 'UK'
    ELSE UPPER(TRIM(country))
END;
```

Without this, your model treats 'US' and 'USA' as different categories — splitting what should be one feature value into five.

### g. Common Mistakes

- **Not handling NULL vs empty string:** `WHERE col IS NULL` misses `''`. Check both: `WHERE col IS NULL OR col = ''`.
- **Timezone issues:** `NOW()` returns database timezone. If users are global, store timestamps in UTC and convert explicitly.
- **Regex in SQL without testing:** `col ~ '^\d+$'` can be slow on large tables. Test on a sample first.
- **Coalescing to 0 for averages:** Replacing NULL spend with 0 before averaging will lower the mean. In some features, NULL should stay NULL (missing = unknown, not zero).

---

## 11. Performance & Optimization

### a. Why This Matters for AI Engineering

Feature engineering queries run over billions of rows. A slow query means slow pipelines, late model retraining, and frustrated teams. Understanding how to write efficient SQL is the difference between a pipeline that completes in 5 minutes and one that times out after 8 hours.

### b. Intuition (AI-Focused)

The database has to read rows from disk. The fewer rows it reads, the faster your query. Indexes tell the database "jump to these rows, don't read everything." EXPLAIN shows you the database's execution plan — where it's doing full scans vs. index lookups. Optimization is about eliminating unnecessary work.

### c. Minimal Theory

**Index Types:**

| Type | Best For | ML Use Case |
|---|---|---|
| B-tree | Equality, range, ORDER BY | `WHERE user_id = 'x'`, `WHERE date > '2024-01-01'` |
| Hash | Equality only | Hash joins |
| Partial | Subset of rows | `WHERE is_active = TRUE` |
| Composite | Multi-column filters | `WHERE user_id = x AND date = y` |
| GIN | Arrays, JSONB | Searching array features |

**EXPLAIN Output Keywords:**

| Keyword | Meaning | Good or Bad? |
|---|---|---|
| `Seq Scan` | Full table scan | Bad on large tables |
| `Index Scan` | Uses an index | Good |
| `Index Only Scan` | Only reads index, not table | Best |
| `Hash Join` | Join via hash table | Good for large tables |
| `Nested Loop` | Row-by-row join | Bad for large tables |
| `cost=X..Y` | Estimated cost | Lower is better |

### d. Practical Usage in ML

- Add indexes to all columns used in WHERE, JOIN, and ORDER BY
- Run EXPLAIN before deploying a new feature query
- Filter early (push WHERE conditions as early as possible)
- Pre-aggregate before joining

### e. Python / SQL Implementation

```sql
-- EXPLAIN ANALYZE: see the actual execution plan and timing
EXPLAIN ANALYZE
SELECT u.user_id, SUM(o.amount) AS total_revenue
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE o.ordered_at >= '2024-01-01'
GROUP BY u.user_id;

-- Add index to speed up the query above
CREATE INDEX idx_orders_user_date
    ON orders (user_id, ordered_at DESC);

-- Partial index: only index recent orders (much smaller, often sufficient)
CREATE INDEX idx_orders_recent
    ON orders (user_id, ordered_at)
    WHERE ordered_at >= '2023-01-01';

-- SLOW: Computing features inside a join (database must compute for every row)
SELECT u.user_id,
       (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.user_id) AS order_count
FROM users u;
-- This is a correlated subquery: runs once per user row = O(n) queries

-- FAST: Pre-aggregate, then join
WITH order_counts AS (
    SELECT user_id, COUNT(*) AS order_count
    FROM orders
    GROUP BY user_id
)
SELECT u.user_id, COALESCE(oc.order_count, 0)
FROM users u
LEFT JOIN order_counts oc ON u.user_id = oc.user_id;
-- Single aggregation pass, single join = much faster

-- SLOW: SELECT * with large tables
SELECT * FROM events WHERE user_id = 'abc';

-- FAST: Select only needed columns
SELECT event_type, event_timestamp, amount
FROM events
WHERE user_id = 'abc';

-- SLOW: Function on indexed column (index unusable)
SELECT * FROM orders WHERE EXTRACT(YEAR FROM ordered_at) = 2024;

-- FAST: Range condition on indexed timestamp column
SELECT * FROM orders
WHERE ordered_at >= '2024-01-01' AND ordered_at < '2025-01-01';

-- When SQL beats Pandas (at scale):
-- Pandas: loads entire table into memory, then filters
-- SQL: filters at source, only sends matching rows
-- Rule of thumb: > 10M rows → SQL. < 1M rows → Pandas fine.
```

**Partitioning for large feature tables:**

```sql
-- Partition by date for efficient time-range queries on large tables
CREATE TABLE events_partitioned (
    user_id         VARCHAR(64),
    event_type      VARCHAR(64),
    amount          NUMERIC(10,2),
    event_timestamp TIMESTAMP NOT NULL
) PARTITION BY RANGE (event_timestamp);

-- Monthly partitions
CREATE TABLE events_2024_01 PARTITION OF events_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE events_2024_02 PARTITION OF events_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Queries filtered to a date range only scan relevant partitions
SELECT COUNT(*) FROM events_partitioned
WHERE event_timestamp >= '2024-01-01' AND event_timestamp < '2024-02-01';
-- Only scans events_2024_01 partition
```

### f. Mini Use Case

Your nightly feature pipeline queries 2 years of events for 5M users. Without optimization: 4-hour runtime. With optimizations:
1. Add composite index on `(user_id, event_timestamp)`
2. Partition table by month
3. Pre-aggregate in CTEs before joining
4. Select only needed columns

Result: 12-minute runtime. This matters because slow pipelines delay model retraining and delay deployments.

### g. Common Mistakes

- **Applying functions to indexed columns:** `WHERE LOWER(email) = 'abc'` cannot use an index on `email`. Create a functional index instead: `CREATE INDEX ON users (LOWER(email))`.
- **Too many indexes:** Indexes speed up reads but slow down writes. On a heavily written feature table, too many indexes hurt pipeline performance.
- **Not running EXPLAIN:** Writing SQL without ever looking at the execution plan is flying blind.
- **Fetching entire tables into Python then filtering:** `pd.read_sql("SELECT * FROM events")` and then filtering in Python — this transfers terabytes of data. Filter in SQL.

---

## 12. Analytical Thinking with SQL

### a. Why This Matters for AI Engineering

Before you train a model, you need to understand your data deeply: how users behave, what patterns exist, what the label distribution is, whether there's temporal drift. SQL analytics is how you explore and validate your dataset at scale. This is also how you extract ML-ready features from raw behavioral data.

### b. Intuition (AI-Focused)

Think of analytical SQL as "exploratory data analysis at database scale." Instead of loading data into a Jupyter notebook, you answer questions directly in SQL where the data lives — faster, more scalable, and no memory constraints.

### c. Minimal Theory

**Key Analytical Patterns:**

| Pattern | What It Answers | ML Relevance |
|---|---|---|
| Funnel Analysis | Conversion at each step | Identify drop-off for labels |
| Cohort Analysis | Retention over time | Train on similar cohorts |
| Time-based Analysis | Trends and seasonality | Detect data drift |
| User Behavior Analysis | What users do | Feature generation |
| Feature Extraction | Derive ML features | Direct pipeline output |

### d. Practical Usage in ML

- Funnel analysis reveals which pipeline step most affects conversion (label quality)
- Cohort analysis ensures training and test sets come from similar time periods
- Time-based analysis identifies seasonality features
- User behavior analysis drives feature ideation

### e. Python / SQL Implementation

```sql
-- FUNNEL ANALYSIS: track conversion through a purchase funnel
-- Each step is a filter, count unique users at each stage
SELECT
    COUNT(DISTINCT user_id)                                                  AS step_0_total,
    COUNT(DISTINCT CASE WHEN did_search THEN user_id END)                   AS step_1_searched,
    COUNT(DISTINCT CASE WHEN did_view_product THEN user_id END)             AS step_2_viewed,
    COUNT(DISTINCT CASE WHEN did_add_to_cart THEN user_id END)              AS step_3_carted,
    COUNT(DISTINCT CASE WHEN did_checkout THEN user_id END)                 AS step_4_checkout,
    COUNT(DISTINCT CASE WHEN did_purchase THEN user_id END)                 AS step_5_purchased,
    -- Conversion rates
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN did_purchase THEN user_id END)
        / NULLIF(COUNT(DISTINCT user_id), 0), 2)                            AS overall_conversion_pct
FROM (
    SELECT
        user_id,
        BOOL_OR(event_type = 'search')        AS did_search,
        BOOL_OR(event_type = 'view_product')  AS did_view_product,
        BOOL_OR(event_type = 'add_to_cart')   AS did_add_to_cart,
        BOOL_OR(event_type = 'checkout')      AS did_checkout,
        BOOL_OR(event_type = 'purchase')      AS did_purchase
    FROM events
    WHERE event_timestamp >= '2024-01-01'
    GROUP BY user_id
) user_funnel;

-- TIME-BASED ANALYSIS: Weekly revenue trend
SELECT
    DATE_TRUNC('week', ordered_at)     AS week,
    COUNT(DISTINCT user_id)            AS active_buyers,
    COUNT(DISTINCT order_id)           AS order_count,
    SUM(amount)                        AS weekly_revenue,
    AVG(amount)                        AS avg_order_value,
    -- Week-over-week change using LAG
    SUM(amount) - LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('week', ordered_at))
                                       AS wow_revenue_change
FROM orders
WHERE ordered_at >= '2023-01-01'
GROUP BY DATE_TRUNC('week', ordered_at)
ORDER BY week;

-- USER BEHAVIOR ANALYSIS: RFM scoring (Recency, Frequency, Monetary)
-- Classic ML feature set for customer segmentation
WITH rfm_raw AS (
    SELECT
        user_id,
        EXTRACT(DAYS FROM NOW() - MAX(ordered_at))   AS recency_days,
        COUNT(DISTINCT order_id)                     AS frequency,
        SUM(amount)                                  AS monetary
    FROM orders
    GROUP BY user_id
),
rfm_scored AS (
    SELECT
        user_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days ASC)    AS r_score,  -- lower recency = higher score
        NTILE(5) OVER (ORDER BY frequency DESC)      AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC)       AS m_score
    FROM rfm_raw
)
SELECT
    user_id,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score                     AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4           THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3           THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2           THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 3           THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2           THEN 'Lost'
        ELSE 'Potential'
    END AS rfm_segment
FROM rfm_scored
ORDER BY rfm_total DESC;
```

### f. Mini Use Case

RFM analysis (above) is a complete ML-ready feature table. R, F, M scores are features; RFM segment is a label for clustering or classification. This comes entirely from SQL — no Python needed until model training.

### g. Common Mistakes

- **Confusing funnel steps:** If a user can do step 3 without step 2, your funnel counts won't be monotonically decreasing. Define funnels carefully.
- **Not controlling for time in cohort analysis:** Comparing Jan cohort to Dec cohort without controlling for holiday effects introduces spurious patterns.
- **Over-indexing on summary statistics:** GROUP BY averages hide distributions. Always look at percentiles too.

---

## 13. SQL Dialects Awareness

### a. Why This Matters for AI Engineering

You will work across multiple databases in your career. Your local dev uses SQLite. Your production data is in PostgreSQL. Your ML feature store queries BigQuery. Your company analytics runs on Snowflake. Understanding dialect differences prevents broken queries and helps you choose the right tool.

### b. Dialect Comparison Table

| Feature | PostgreSQL | MySQL | SQLite | BigQuery | Snowflake |
|---|---|---|---|---|---|
| Auto-increment | `SERIAL` / `IDENTITY` | `AUTO_INCREMENT` | `INTEGER PRIMARY KEY` | `INT64` (no auto) | `AUTOINCREMENT` |
| String concat | `||` or `CONCAT()` | `CONCAT()` | `||` | `CONCAT()` | `||` or `CONCAT()` |
| Date functions | `DATE_TRUNC`, `EXTRACT` | `DATE_FORMAT`, `YEAR()` | `strftime()` | `DATE_TRUNC`, `EXTRACT` | `DATE_TRUNC`, `YEAR()` |
| Regex | `~` / `~*` | `REGEXP` | `GLOB` / `LIKE` | `REGEXP_CONTAINS` | `RLIKE` |
| Upsert | `ON CONFLICT DO UPDATE` | `ON DUPLICATE KEY UPDATE` | `ON CONFLICT` | `MERGE` | `MERGE` |
| JSON | `JSONB`, `->>` | `JSON_EXTRACT()` | Limited | `JSON_EXTRACT_SCALAR` | `GET_PATH()` |
| Window frames | Full support | Full support | Limited | Full support | Full support |

### c. Practical Notes for AI Engineers

```sql
-- PostgreSQL (most feature-rich, most used in ML backends)
SELECT EXTRACT(EPOCH FROM NOW() - created_at) / 86400 AS days_old;
SELECT array_agg(product_id) AS product_list FROM orders GROUP BY user_id;

-- BigQuery (for large-scale analytics)
SELECT DATE_DIFF(CURRENT_DATE(), DATE(created_at), DAY) AS days_old;
SELECT ARRAY_AGG(product_id) AS product_list FROM orders GROUP BY user_id;

-- SQLite (for local development and testing)
SELECT CAST((julianday('now') - julianday(created_at)) AS INTEGER) AS days_old;

-- Snowflake (cloud warehouse)
SELECT DATEDIFF('day', created_at::DATE, CURRENT_DATE()) AS days_old;
```

**ANSI SQL (works everywhere):**
```sql
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total,
    AVG(amount) AS avg_amount
FROM orders
WHERE ordered_at IS NOT NULL
GROUP BY user_id
HAVING COUNT(*) > 0
ORDER BY total DESC;
```

### g. Common Mistakes

- **Developing on SQLite, deploying on PostgreSQL:** SQLite is very permissive (dynamic types, no strict mode). PostgreSQL is strict. Queries that work on SQLite break on PostgreSQL.
- **Using dialect-specific syntax without knowing it:** `ILIKE` is PostgreSQL only. `LIMIT x OFFSET y` isn't supported in all dialects.
- **Assuming all window functions work identically:** RANGE frame behavior varies across databases.

---

## 14. SQL + Python Integration

### a. Why This Matters for AI Engineering

ML pipelines are Python. But data lives in databases. The connection between your Python code and your database is where most production ML pipelines live. Understanding this integration — efficiently, securely, and correctly — is essential.

### b. Tools Overview

| Tool | Best For | Notes |
|---|---|---|
| `sqlite3` | Local development, testing | Built into Python stdlib |
| `psycopg2` | PostgreSQL production | Low-level, fast |
| `SQLAlchemy` | ORM + dialect abstraction | Use for multi-DB apps |
| `pandas.read_sql` | Data extraction to DataFrame | Easy but no write optimization |
| `connectorx` | Fast bulk reads | Much faster than pandas for large data |

### c. Python / SQL Implementation

```python
# ============================================================
# 1. SQLite: local development and testing
# ============================================================
import sqlite3
import pandas as pd

conn = sqlite3.connect('ml_dev.db')
cursor = conn.cursor()

# Create and populate test table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_features (
        user_id TEXT PRIMARY KEY,
        age INTEGER,
        total_purchases INTEGER,
        is_churned INTEGER
    )
""")

cursor.executemany(
    "INSERT OR REPLACE INTO user_features VALUES (?, ?, ?, ?)",
    [('u001', 28, 5, 0), ('u002', 35, 1, 1), ('u003', 42, 12, 0)]
)
conn.commit()

# Read into DataFrame
df = pd.read_sql_query(
    "SELECT * FROM user_features WHERE total_purchases > 0",
    conn
)
conn.close()

# ============================================================
# 2. PostgreSQL with psycopg2: production
# ============================================================
import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "host": "localhost",
    "dbname": "ml_prod",
    "user": "ml_user",
    "password": "your_password",
    "port": 5432
}

def get_training_data(feature_date: str, min_purchases: int = 1) -> pd.DataFrame:
    """Extract training data from PostgreSQL — parameterized, safe."""
    query = """
        SELECT
            user_id,
            age,
            country,
            total_purchases,
            avg_order_value,
            days_since_last_purchase,
            is_churned
        FROM user_features
        WHERE feature_date = %s
          AND total_purchases >= %s
          AND age IS NOT NULL
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql_query(query, conn, params=(feature_date, min_purchases))
        return df
    finally:
        conn.close()  # Always close, even on error

df_train = get_training_data('2024-01-01', min_purchases=2)

# ============================================================
# 3. SQLAlchemy: recommended for production apps
# ============================================================
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://ml_user:password@localhost:5432/ml_prod",
    pool_size=5,           # Connection pooling
    max_overflow=10,
    echo=False             # Set True for SQL debugging
)

# Read
with engine.connect() as conn:
    df = pd.read_sql(
        text("SELECT * FROM user_features WHERE feature_date = :dt"),
        conn,
        params={"dt": "2024-01-01"}
    )

# Write predictions back
predictions_df = pd.DataFrame({
    'user_id': ['u001', 'u002'],
    'churn_score': [0.87, 0.23],
    'model_version': ['v2.1', 'v2.1']
})

predictions_df.to_sql(
    'predictions',
    engine,
    if_exists='append',    # 'replace' drops and recreates, 'append' adds rows
    index=False,
    method='multi'         # Faster bulk insert
)

# ============================================================
# 4. Parameterized queries — ALWAYS use these (prevent SQL injection)
# ============================================================
import psycopg2

# WRONG (SQL injection risk):
user_input = "'; DROP TABLE users; --"
query = f"SELECT * FROM users WHERE email = '{user_input}'"  # NEVER do this

# CORRECT (parameterized):
query = "SELECT * FROM users WHERE email = %s"
cursor.execute(query, (user_input,))  # Safely handles any input

# CORRECT with SQLAlchemy:
with engine.connect() as conn:
    result = conn.execute(
        text("SELECT * FROM users WHERE email = :email"),
        {"email": user_input}  # Parameterized, safe
    )

# ============================================================
# 5. Efficient bulk operations with connectorx (10-100x faster than pandas.read_sql)
# ============================================================
import connectorx as cx

df = cx.read_sql(
    "postgresql://ml_user:password@localhost:5432/ml_prod",
    "SELECT * FROM user_features WHERE feature_date = '2024-01-01'",
    return_type="pandas"
)
# connectorx reads in parallel and is significantly faster for large datasets
```

### g. Common Mistakes

- **Not closing connections:** Connection pools get exhausted. Always use `try/finally` or context managers.
- **String formatting SQL queries:** Never `f"SELECT * WHERE id = '{user_input}'"`. Always use parameterized queries.
- **`pd.read_sql("SELECT * FROM events")`:** Loads the entire table into memory. Add WHERE clauses to filter in SQL first.
- **Not using connection pooling in services:** Creating a new connection per request is slow. Use SQLAlchemy's connection pool.

---

## 15. Databases Every AI Engineer Should Touch

### a. PostgreSQL — The Gold Standard

**Why:** Full-featured, ACID-compliant, excellent window functions, JSONB support, strong Python ecosystem. This is the most common database you'll encounter in ML backends.

```bash
# Installation
brew install postgresql  # macOS
apt-get install postgresql  # Ubuntu

# Connect
psql -h localhost -U postgres -d mydb
```

**Key PostgreSQL features for ML:**
- `JSONB` columns for flexible schema (store raw features as JSON)
- `COPY` command for bulk data loading (10-100x faster than INSERT)
- `EXPLAIN ANALYZE` for query tuning
- `pg_stat_statements` for identifying slow queries

### b. SQLite — The ML Dev Environment

**Why:** Zero-config, file-based, perfect for local development, testing pipelines, and small ML projects. Ships with Python.

```python
import sqlite3
conn = sqlite3.connect('features.db')  # Creates file if not exists
```

**Use cases:** Unit testing SQL queries, rapid prototyping, embedding a database in a Python ML application.

### c. MySQL/MariaDB

**Why:** Common in web application backends. You'll often extract training data from MySQL databases.

**Key differences from PostgreSQL:**
- `LIMIT x OFFSET y` same syntax
- `DATETIME` vs PostgreSQL's `TIMESTAMP WITH TIME ZONE`
- No `RETURNING` clause
- `GROUP BY` is more permissive (dangerous — allows non-aggregated columns)

### d. Cloud Warehouses (Awareness Level)

| Warehouse | Owner | Best For | Unique Feature |
|---|---|---|---|
| BigQuery | Google | Petabyte analytics, ML at scale | `ML.PREDICT()` in SQL |
| Snowflake | Snowflake Inc. | Multi-cloud, flexible | Time-travel queries |
| Redshift | AWS | AWS-native data warehousing | Spectrum for S3 |
| Databricks | Databricks | ML + data engineering | Delta Lake + Spark SQL |

**BigQuery ML — SQL-native model training (awareness):**

```sql
-- You can train a logistic regression model directly in BigQuery SQL
CREATE OR REPLACE MODEL `my_project.my_dataset.churn_model`
OPTIONS(model_type='logistic_reg', input_label_cols=['is_churned'])
AS
SELECT
    age,
    total_purchases,
    avg_order_value,
    days_since_last_purchase,
    is_churned
FROM `my_project.my_dataset.user_features`
WHERE feature_date = '2024-01-01';

-- Then predict
SELECT * FROM ML.PREDICT(
    MODEL `my_project.my_dataset.churn_model`,
    (SELECT * FROM `my_project.my_dataset.user_features_new`)
);
```

---

## 16. Real-World SQL Skills

### a. Writing Readable SQL

Good SQL is written for humans first, machines second. Your colleagues will read, maintain, and extend your queries. Make it easy for them.

```sql
-- BAD: Unreadable, no structure
SELECT u.user_id,u.email,COUNT(o.order_id) cnt,SUM(o.amount) rev,CASE WHEN SUM(o.amount)>1000 THEN 'high' WHEN SUM(o.amount)>100 THEN 'med' ELSE 'low' END tier FROM users u LEFT JOIN orders o ON u.user_id=o.user_id WHERE u.created_at>'2023-01-01' AND u.is_bot=FALSE GROUP BY u.user_id,u.email HAVING COUNT(o.order_id)>0 ORDER BY rev DESC;

-- GOOD: Readable, structured, commented
/*
 * Feature extraction: User revenue tier for churn model training
 * Author: Data Eng Team
 * Date: 2024-01-15
 * Ticket: ML-1234
 */
SELECT
    u.user_id,
    u.email,

    -- Purchase features
    COUNT(o.order_id)    AS total_orders,
    SUM(o.amount)        AS total_revenue,

    -- Derived label / feature
    CASE
        WHEN SUM(o.amount) > 1000  THEN 'high_value'
        WHEN SUM(o.amount) > 100   THEN 'mid_value'
        ELSE                            'low_value'
    END                  AS revenue_tier

FROM users u

LEFT JOIN orders o
    ON u.user_id = o.user_id

WHERE
    u.created_at > '2023-01-01'
    AND u.is_bot = FALSE

GROUP BY
    u.user_id,
    u.email

HAVING
    COUNT(o.order_id) > 0  -- Only users with at least one order

ORDER BY
    total_revenue DESC;
```

### b. Debugging Wrong Results

```sql
-- Step 1: Count rows at each stage
SELECT COUNT(*) FROM users;                          -- Source count
SELECT COUNT(*) FROM users WHERE is_bot = FALSE;    -- After filter
SELECT COUNT(DISTINCT user_id) FROM orders;         -- Users with orders

-- Step 2: Inspect before aggregating (check for duplicates)
SELECT user_id, COUNT(*) AS cnt
FROM orders
GROUP BY user_id
HAVING COUNT(*) > 1
LIMIT 10;

-- Step 3: Run the query on a single entity first
SELECT *
FROM user_features
WHERE user_id = 'user_001';

-- Step 4: Check for NULLs affecting calculations
SELECT
    user_id,
    AVG(amount),                          -- Ignores NULLs
    AVG(COALESCE(amount, 0))              -- Treats NULL as 0
FROM orders
WHERE user_id = 'user_001'
GROUP BY user_id;

-- Step 5: Break complex query into CTEs and inspect each
WITH step1 AS (SELECT ... FROM ...),
     step2 AS (SELECT ... FROM step1 ...)
SELECT * FROM step1 LIMIT 10;  -- Inspect intermediate result
```

### c. Versioning SQL

Best practices for version-controlled SQL in ML pipelines:

```
project/
├── sql/
│   ├── features/
│   │   ├── user_purchase_features_v1.sql
│   │   ├── user_purchase_features_v2.sql   ← incremental improvements
│   │   └── user_engagement_features_v1.sql
│   ├── labels/
│   │   └── churn_labels_90d.sql
│   └── training_sets/
│       └── churn_training_set_v3.sql
```

Use `-- Version: 2.1`, `-- Author:`, `-- Last modified:`, `-- Depends on:` comments in SQL files.

---

## 17. Cross-Topic Connections

| SQL Concept | Connects To | How They Interact |
|---|---|---|
| `GROUP BY` + aggregations | Feature Engineering (Python/ML) | SQL aggregates raw events → Python receives feature matrix |
| Window functions | Time-series ML | LAG/LEAD create temporal features that LSTMs and gradient boosters need |
| JOIN | Data pipelines | Multi-table joins build the enriched datasets that feed ML training |
| CTEs | MLOps / Pipeline design | CTE structure mirrors DAG structure in Airflow/dbt pipelines |
| Indexes | ML pipeline performance | Faster SQL = faster feature refresh = faster model retraining cycles |
| Transactions | Data integrity | Atomic writes prevent corrupted training data from reaching models |
| Normalization | Data modeling | Normalized sources → denormalized feature tables (SQL as ETL) |
| Filtering | Label engineering | WHERE clauses define training population and prevent data leakage |
| CASE WHEN | Feature encoding | Binary/ordinal encoding in SQL before Python processing |
| NULL handling | Missing value strategy | SQL-level imputation strategy feeds into model preprocessing decisions |
| SQL + Python | End-to-end pipelines | `pd.read_sql()` → pandas → scikit-learn → `to_sql()` → predictions back to DB |

---

## 18. End-to-End Practical System View

Here is how SQL fits into a complete ML system:

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL-WORLD ML SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. RAW DATA INGESTION                                      │
│     Web events → Kafka → PostgreSQL (raw_events table)     │
│     Transactions → ETL → BigQuery (orders table)           │
│                                                             │
│  2. SQL: DATA EXTRACTION & CLEANING         ← YOU ARE HERE │
│     ┌─────────────────────────────────────┐               │
│     │ SELECT, WHERE, JOIN, CLEAN          │               │
│     │ Remove bots, handle NULLs           │               │
│     │ Standardize categories              │               │
│     └─────────────────────────────────────┘               │
│                         ↓                                   │
│  3. SQL: FEATURE ENGINEERING               ← YOU ARE HERE │
│     ┌─────────────────────────────────────┐               │
│     │ GROUP BY + aggregations             │               │
│     │ Window functions (LAG, rolling avg) │               │
│     │ CTEs for multi-step transforms      │               │
│     │ Output: user_features table         │               │
│     └─────────────────────────────────────┘               │
│                         ↓                                   │
│  4. PYTHON: LOAD & PREPROCESS                              │
│     ┌─────────────────────────────────────┐               │
│     │ pd.read_sql() → DataFrame           │               │
│     │ Scale, encode, split train/test     │               │
│     │ sklearn Pipeline                    │               │
│     └─────────────────────────────────────┘               │
│                         ↓                                   │
│  5. PYTHON: MODEL TRAINING                                  │
│     ┌─────────────────────────────────────┐               │
│     │ XGBoost / LightGBM / Neural Net     │               │
│     │ Cross-validation                    │               │
│     │ Hyperparameter tuning               │               │
│     └─────────────────────────────────────┘               │
│                         ↓                                   │
│  6. SQL: STORE PREDICTIONS                 ← YOU ARE HERE │
│     ┌─────────────────────────────────────┐               │
│     │ INSERT INTO predictions             │               │
│     │ UPSERT with ON CONFLICT             │               │
│     │ Transaction for atomic updates      │               │
│     └─────────────────────────────────────┘               │
│                         ↓                                   │
│  7. SQL: MONITORING & ANALYSIS             ← YOU ARE HERE │
│     ┌─────────────────────────────────────┐               │
│     │ Track prediction distribution       │               │
│     │ Compare against actuals (labels)    │               │
│     │ Cohort analysis for drift detection │               │
│     └─────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**SQL appears in at least 4 of 7 stages.** It is not a supporting tool — it is a primary tool.

---

## 19. Hands-On Projects

### Project 1: Build a Complete ML Feature Pipeline Using SQL + Python

**Problem Statement:**
Build a customer churn prediction feature pipeline. Given raw event and transaction data, produce a clean, ML-ready feature table using SQL, then train a simple classifier using Python.

**Dataset Description:**
We'll create synthetic data representing an e-commerce platform:
- `users`: 10,000 users with demographics
- `orders`: 50,000 orders over 2 years
- `events`: 500,000 behavioral events

**Step-by-Step Implementation:**

**Step 1: Set up the database and generate data**

```python
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Create database
conn = sqlite3.connect('churn_project.db')

# Generate synthetic data
np.random.seed(42)
n_users = 10000

users_df = pd.DataFrame({
    'user_id': [f'u{i:05d}' for i in range(n_users)],
    'age': np.random.randint(18, 70, n_users),
    'country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_users, p=[0.5, 0.25, 0.15, 0.1]),
    'signup_date': [
        (datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d')
        for _ in range(n_users)
    ],
    'is_bot': np.random.choice([0, 1], n_users, p=[0.97, 0.03])
})

# Save to SQLite
users_df.to_sql('users', conn, if_exists='replace', index=False)

# Generate orders (some users churn and have no recent orders)
n_orders = 50000
order_users = np.random.choice(users_df['user_id'].values, n_orders)

orders_df = pd.DataFrame({
    'order_id': [f'o{i:06d}' for i in range(n_orders)],
    'user_id': order_users,
    'amount': np.round(np.random.exponential(scale=75, size=n_orders).clip(5, 1000), 2),
    'ordered_at': [
        (datetime(2022, 6, 1) + timedelta(days=np.random.randint(0, 548))).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(n_orders)
    ]
})
orders_df.to_sql('orders', conn, if_exists='replace', index=False)

# Generate churn events (30% of users churn)
churned_users = np.random.choice(users_df['user_id'].values, int(n_users * 0.3), replace=False)
churn_df = pd.DataFrame({
    'user_id': churned_users,
    'churned_at': [
        (datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))).strftime('%Y-%m-%d')
        for _ in churned_users
    ]
})
churn_df.to_sql('churn_events', conn, if_exists='replace', index=False)

print("Database created with:")
print(f"  Users:  {len(users_df):,}")
print(f"  Orders: {len(orders_df):,}")
print(f"  Churn:  {len(churn_df):,}")
```

**Step 2: Build the feature engineering SQL**

```sql
-- feature_pipeline.sql
-- Complete feature engineering pipeline for churn prediction

WITH
-- Step 1: Clean user cohort (remove bots, define observation window)
clean_users AS (
    SELECT
        user_id,
        age,
        country,
        signup_date
    FROM users
    WHERE is_bot = 0
      AND signup_date < '2023-10-01'  -- Users established before observation window
),

-- Step 2: Purchase features (last 90 days before Jan 1, 2024)
purchase_features AS (
    SELECT
        o.user_id,
        COUNT(DISTINCT o.order_id)                           AS order_count_90d,
        SUM(o.amount)                                        AS revenue_90d,
        AVG(o.amount)                                        AS avg_order_value_90d,
        MAX(o.amount)                                        AS max_order_90d,
        MIN(o.ordered_at)                                    AS first_order_date,
        MAX(o.ordered_at)                                    AS last_order_date,
        CAST(
            (julianday('2024-01-01') - julianday(MAX(o.ordered_at)))
        AS INTEGER)                                          AS days_since_last_order
    FROM orders o
    INNER JOIN clean_users cu ON o.user_id = cu.user_id
    WHERE o.ordered_at >= '2023-10-01'
      AND o.ordered_at <  '2024-01-01'
    GROUP BY o.user_id
),

-- Step 3: Historical features (all-time)
historical_features AS (
    SELECT
        o.user_id,
        COUNT(DISTINCT o.order_id)  AS total_orders_alltime,
        SUM(o.amount)               AS revenue_alltime
    FROM orders o
    INNER JOIN clean_users cu ON o.user_id = cu.user_id
    WHERE o.ordered_at < '2024-01-01'
    GROUP BY o.user_id
),

-- Step 4: Churn label (did they churn in Q1 2024?)
churn_labels AS (
    SELECT
        user_id,
        1 AS is_churned
    FROM churn_events
    WHERE churned_at >= '2024-01-01'
      AND churned_at <  '2024-04-01'
),

-- Step 5: Assemble feature matrix
feature_matrix AS (
    SELECT
        cu.user_id,

        -- Demographic features
        cu.age,
        cu.country,
        CAST(
            (julianday('2024-01-01') - julianday(cu.signup_date))
        AS INTEGER)                                          AS account_age_days,

        -- Recent purchase features (90-day window)
        COALESCE(pf.order_count_90d, 0)                     AS order_count_90d,
        COALESCE(pf.revenue_90d, 0)                         AS revenue_90d,
        pf.avg_order_value_90d,
        pf.days_since_last_order,

        -- Historical features
        COALESCE(hf.total_orders_alltime, 0)                AS total_orders_alltime,
        COALESCE(hf.revenue_alltime, 0)                     AS revenue_alltime,

        -- Derived features
        CASE
            WHEN COALESCE(hf.total_orders_alltime, 0) = 0 THEN 0
            ELSE CAST(COALESCE(pf.order_count_90d, 0) AS FLOAT)
                 / hf.total_orders_alltime
        END AS recent_order_ratio,   -- Are recent orders a high % of total?

        -- Label
        COALESCE(cl.is_churned, 0)                          AS is_churned

    FROM clean_users cu
    LEFT JOIN purchase_features  pf ON cu.user_id = pf.user_id
    LEFT JOIN historical_features hf ON cu.user_id = hf.user_id
    LEFT JOIN churn_labels        cl ON cu.user_id = cl.user_id
)

SELECT * FROM feature_matrix
ORDER BY user_id;
```

**Step 3: Execute pipeline and train model in Python**

```python
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- Execute SQL feature pipeline ---
conn = sqlite3.connect('churn_project.db')

with open('feature_pipeline.sql', 'r') as f:
    sql = f.read()

df = pd.read_sql_query(sql, conn)
conn.close()

print(f"Feature matrix shape: {df.shape}")
print(f"\nLabel distribution:\n{df['is_churned'].value_counts(normalize=True).round(3)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# --- Preprocessing ---
# Encode country
le = LabelEncoder()
df['country_encoded'] = le.fit_transform(df['country'].fillna('Unknown'))

# Fill remaining NULLs with median
numeric_cols = ['avg_order_value_90d', 'days_since_last_order']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Define features
feature_cols = [
    'age', 'account_age_days', 'country_encoded',
    'order_count_90d', 'revenue_90d', 'avg_order_value_90d',
    'days_since_last_order', 'total_orders_alltime',
    'revenue_alltime', 'recent_order_ratio'
]

X = df[feature_cols]
y = df['is_churned']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Model ---
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# --- Feature Importance ---
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(importance_df.to_string(index=False))

# --- Write predictions back to database ---
conn = sqlite3.connect('churn_project.db')
predictions_df = pd.DataFrame({
    'user_id': df.loc[X_test.index, 'user_id'],
    'churn_score': y_prob,
    'is_churned_predicted': y_pred,
    'model_version': 'v1.0'
})
predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
conn.close()

print(f"\nPredictions written for {len(predictions_df):,} users")
```

---

### Project 2: SQL-Based Analytical Feature Store

**Problem Statement:**
Build a reusable analytical feature store using SQL views and materialized queries. This mimics what real ML platforms like Feast or Tecton do, but built entirely in SQL.

**Implementation:**

```sql
-- feature_store.sql
-- A SQL-based feature store: reusable feature definitions

-- VIEW 1: RFM Features (reusable across models)
CREATE VIEW v_rfm_features AS
SELECT
    user_id,
    EXTRACT(DAYS FROM NOW() - MAX(ordered_at))   AS recency_days,
    COUNT(DISTINCT order_id)                     AS frequency,
    SUM(amount)                                  AS monetary,
    NTILE(5) OVER (ORDER BY COUNT(DISTINCT order_id) DESC) AS frequency_quintile,
    NTILE(5) OVER (ORDER BY SUM(amount) DESC)    AS monetary_quintile
FROM orders
GROUP BY user_id;

-- VIEW 2: Engagement Features
CREATE VIEW v_engagement_features AS
SELECT
    user_id,
    COUNT(DISTINCT DATE(event_timestamp))                       AS active_days_90d,
    COUNT(*) FILTER (WHERE event_type = 'search')              AS search_count_90d,
    COUNT(*) FILTER (WHERE event_type = 'view_product')        AS product_view_count_90d,
    COUNT(*) FILTER (WHERE event_type = 'add_to_cart')         AS cart_add_count_90d,
    -- Conversion rate: cart-to-purchase
    COUNT(*) FILTER (WHERE event_type = 'purchase')::FLOAT
        / NULLIF(COUNT(*) FILTER (WHERE event_type = 'add_to_cart'), 0)
                                                               AS cart_conversion_rate
FROM events
WHERE event_timestamp >= NOW() - INTERVAL '90 days'
GROUP BY user_id;

-- VIEW 3: Master Feature View (assembles all features for training)
CREATE VIEW v_training_features AS
SELECT
    u.user_id,
    u.age,
    u.country,
    -- RFM features
    r.recency_days,
    r.frequency,
    r.monetary,
    r.frequency_quintile,
    r.monetary_quintile,
    -- Engagement features
    COALESCE(e.active_days_90d, 0)        AS active_days_90d,
    COALESCE(e.search_count_90d, 0)       AS search_count_90d,
    COALESCE(e.cart_conversion_rate, 0)   AS cart_conversion_rate
FROM users u
LEFT JOIN v_rfm_features r      ON u.user_id = r.user_id
LEFT JOIN v_engagement_features e ON u.user_id = e.user_id
WHERE u.is_bot = FALSE;
```

```python
# Using the feature store view in Python
import pandas as pd
import psycopg2

conn = psycopg2.connect("postgresql://localhost/ml_db")

# Single query to get all features — views handle the complexity
df = pd.read_sql_query(
    "SELECT * FROM v_training_features WHERE country = 'US'",
    conn
)

print(f"Training set shape: {df.shape}")
conn.close()
```

---

## 20. Cheat Sheets

### SQL Cheat Sheet for ML Engineers

**Core Aggregation Patterns:**

| Goal | SQL Pattern |
|---|---|
| Count per group | `SELECT col, COUNT(*) FROM t GROUP BY col` |
| Unique count | `COUNT(DISTINCT col)` |
| Conditional count | `COUNT(*) FILTER (WHERE condition)` or `SUM(CASE WHEN condition THEN 1 ELSE 0 END)` |
| Rate (avoid div/0) | `col1::FLOAT / NULLIF(col2, 0)` |
| Running total | `SUM(col) OVER (PARTITION BY id ORDER BY date)` |
| Moving average | `AVG(col) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| Previous value | `LAG(col, 1) OVER (PARTITION BY id ORDER BY date)` |
| Percentile bucket | `NTILE(10) OVER (ORDER BY col DESC)` |

**Data Cleaning Patterns:**

| Goal | SQL Pattern |
|---|---|
| Fill NULL with default | `COALESCE(col, 0)` |
| Replace value with NULL | `NULLIF(col, -999)` |
| Safe type cast | `CASE WHEN col ~ '^\d+$' THEN col::INT ELSE NULL END` |
| Normalize string | `LOWER(TRIM(col))` |
| Extract email domain | `SPLIT_PART(email, '@', 2)` |
| Day of week | `EXTRACT(DOW FROM timestamp_col)` |
| Days between dates | `EXTRACT(DAYS FROM date2 - date1)` |

**Join Patterns:**

| Goal | SQL Pattern |
|---|---|
| Enrich all entities | `LEFT JOIN` |
| Only entities in both | `INNER JOIN` |
| Find missing records | `LEFT JOIN ... WHERE right_table.id IS NULL` |
| Prevent fan-out | Aggregate in subquery BEFORE joining |

**Performance Patterns:**

| Goal | SQL Pattern |
|---|---|
| Check execution plan | `EXPLAIN ANALYZE SELECT ...` |
| Add index | `CREATE INDEX idx_name ON table(col)` |
| Partial index | `CREATE INDEX ... WHERE condition` |
| Avoid function on index | `WHERE col >= '2024-01-01'` not `WHERE YEAR(col) = 2024` |
| Bulk insert | `INSERT INTO ... VALUES (...), (...), (...)` |

**Python + SQL Patterns:**

```python
# Read to DataFrame
df = pd.read_sql_query(query, conn, params=(param1, param2))

# Parameterized (PostgreSQL)
cursor.execute("SELECT * FROM t WHERE id = %s", (user_id,))

# Parameterized (SQLite)
cursor.execute("SELECT * FROM t WHERE id = ?", (user_id,))

# Bulk insert
pd.DataFrame(data).to_sql('table', engine, if_exists='append', index=False, method='multi')

# Fast bulk read
import connectorx as cx
df = cx.read_sql(connection_string, query, return_type="pandas")
```

---

## 21. Interview Preparation

### SQL Interview Questions for AI/ML Engineers

**Conceptual Questions:**

1. **"Why would you use SQL instead of pandas for feature engineering?"**
   - SQL scales to billions of rows without memory constraints. SQL filtering happens at the database — only matching rows transfer to Python. SQL transformations are reproducible, versionable, and shareable.

2. **"What is the difference between WHERE and HAVING?"**
   - WHERE filters individual rows before grouping. HAVING filters groups after aggregation. You cannot use aggregate functions in WHERE.

3. **"Explain the difference between INNER JOIN and LEFT JOIN. When would you use each in an ML pipeline?"**
   - INNER JOIN: returns only rows with matches in both tables. Use when you only want entities present in both datasets.
   - LEFT JOIN: returns all rows from the left table, with NULLs for unmatched right rows. Use when you want to keep all entities and add features where available (don't drop users with no orders).

4. **"What are window functions and how would you use them for feature engineering?"**
   - Window functions compute calculations over a window of related rows without collapsing them. LAG/LEAD create time-series features. ROW_NUMBER identifies sequence position. Rolling AVG smooths noisy metrics.

5. **"How do you prevent data leakage in SQL feature engineering?"**
   - Strict date filters in WHERE: `WHERE event_date < prediction_date`. Never use future data in features. Use `<` (strict) not `<=` for boundary dates.

**Practical Questions (Coding):**

6. **"Write SQL to find the top 3 products by revenue in each country."**
```sql
WITH ranked AS (
    SELECT
        p.country,
        p.product_name,
        SUM(o.amount) AS revenue,
        ROW_NUMBER() OVER (
            PARTITION BY p.country
            ORDER BY SUM(o.amount) DESC
        ) AS rank
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY p.country, p.product_name
)
SELECT country, product_name, revenue
FROM ranked
WHERE rank <= 3
ORDER BY country, rank;
```

7. **"Calculate a 7-day rolling average of daily revenue."**
```sql
SELECT
    order_date,
    daily_revenue,
    AVG(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS revenue_7day_avg
FROM (
    SELECT
        DATE(ordered_at)  AS order_date,
        SUM(amount)        AS daily_revenue
    FROM orders
    GROUP BY DATE(ordered_at)
) daily
ORDER BY order_date;
```

8. **"Identify users who were active in month 1 but not in month 2 (churned users)."**
```sql
SELECT DISTINCT user_id
FROM events
WHERE DATE_TRUNC('month', event_timestamp) = '2024-01-01'
  AND user_id NOT IN (
    SELECT DISTINCT user_id
    FROM events
    WHERE DATE_TRUNC('month', event_timestamp) = '2024-02-01'
  );
```

9. **"Given a table of user sessions, compute the average session duration per user and identify the top 10% by duration."**
```sql
WITH session_durations AS (
    SELECT
        user_id,
        session_id,
        EXTRACT(EPOCH FROM (MAX(event_timestamp) - MIN(event_timestamp))) AS duration_seconds
    FROM events
    GROUP BY user_id, session_id
),
user_avg_duration AS (
    SELECT
        user_id,
        AVG(duration_seconds) AS avg_duration_seconds,
        NTILE(10) OVER (ORDER BY AVG(duration_seconds) DESC) AS duration_decile
    FROM session_durations
    GROUP BY user_id
)
SELECT *
FROM user_avg_duration
WHERE duration_decile = 1
ORDER BY avg_duration_seconds DESC;
```

10. **"How would you design a SQL schema for storing ML model predictions at scale?"**
```sql
-- Answer: structured, indexed, with audit trail
CREATE TABLE model_predictions (
    prediction_id   BIGSERIAL       PRIMARY KEY,
    entity_id       VARCHAR(64)     NOT NULL,        -- user_id, item_id, etc.
    model_name      VARCHAR(128)    NOT NULL,
    model_version   VARCHAR(32)     NOT NULL,
    score           NUMERIC(8, 6)   NOT NULL CHECK (score BETWEEN 0 AND 1),
    label           VARCHAR(64),                     -- predicted class
    features_hash   VARCHAR(64),                     -- hash of input features for reproducibility
    predicted_at    TIMESTAMP       NOT NULL DEFAULT NOW(),
    batch_id        VARCHAR(64)                      -- group predictions by run
);

CREATE INDEX idx_preds_entity     ON model_predictions (entity_id, model_name);
CREATE INDEX idx_preds_predicted  ON model_predictions (predicted_at DESC);
CREATE INDEX idx_preds_batch      ON model_predictions (batch_id);
```

---

## 22. Resources

### Documentation
- [PostgreSQL Official Docs](https://www.postgresql.org/docs/) — most complete SQL reference, window functions section is excellent
- [SQLite Documentation](https://www.sqlite.org/docs.html) — for local dev understanding
- [BigQuery SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax) — for cloud-scale ML
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference) — increasingly common in enterprise ML

### Practical Learning
- [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial/) — hands-on, analytics-focused
- [SQLZoo](https://sqlzoo.net/) — interactive exercises
- [pgexercises.com](https://pgexercises.com/) — PostgreSQL-specific practice problems
- [Leetcode Database Problems](https://leetcode.com/problemset/database/) — interview prep

### Books
- *"Learning SQL" by Alan Beaulieu* — solid fundamentals
- *"SQL for Data Analysis" by Cathy Tanimura (O'Reilly)* — analytics and ML focus, highly recommended
- *"High Performance MySQL"* — for optimization depth

### Tools
- [DBeaver](https://dbeaver.io/) — free universal database IDE
- [DataGrip](https://www.jetbrains.com/datagrip/) — JetBrains SQL IDE (paid, excellent)
- [TablePlus](https://tableplus.com/) — clean macOS/Windows DB client
- [dbt (data build tool)](https://www.getdbt.com/) — SQL-based transformation pipelines, essential for ML feature engineering at scale
- [sqlfluff](https://github.com/sqlfluff/sqlfluff) — SQL linter for enforcing style and quality

### Applied Resources
- [The Analytics Engineering Guide (dbt docs)](https://docs.getdbt.com/guides) — modern SQL-based data engineering
- [Locally Optimistic Blog](https://locallyoptimistic.com/) — analytics engineering in practice
- [Count.co SQL Patterns](https://count.co/sql-resources/) — analytical SQL patterns

---

*Document generated for AI Engineering curriculum. All SQL examples use PostgreSQL syntax unless otherwise noted. Examples target Python 3.10+ and pandas 2.x.*
