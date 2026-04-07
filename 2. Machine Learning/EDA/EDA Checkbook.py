"""
================================================================================
 EXHAUSTIVE EDA GUIDEBOOK FOR ML ENGINEERS
 A complete reference for Exploratory Data Analysis on industrial-scale datasets
================================================================================

DATASET USED IN EXAMPLES:
  NYC Taxi Trip Records (Yellow Cab) - Jan 2023
  URL: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
  Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

  Titanic Dataset (for classification EDA)
  URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

# ================================================================================
# 0. ENVIRONMENT SETUP
# ================================================================================

# Install dependencies (run once):
# pip install polars plotly kaleido ydata-profiling scipy scikit-learn
# pip install statsmodels missingno pyarrow fastparquet umap-learn

import warnings
warnings.filterwarnings('ignore')

import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_white"

from scipy import stats
from scipy.stats import kstest, shapiro, normaltest, chi2_contingency, pointbiserialr
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans

import missingno as msno
import matplotlib.pyplot as plt

from pathlib import Path
import time
import gc

# ================================================================================
# 1. DATA INGESTION & INITIAL INSPECTION
# ================================================================================

class DataIngestion:
    """
    Step 1: Load data efficiently using Polars (LazyFrame first for large files)
    Polars is 5-20x faster than pandas for large datasets and uses Apache Arrow
    memory format under the hood.
    """

    @staticmethod
    def load_parquet_lazy(path: str) -> pl.LazyFrame:
        """
        Use LazyFrame for large files - queries are optimized before execution.
        The query plan is optimized (predicate pushdown, projection pushdown).
        """
        return pl.scan_parquet(path)

    @staticmethod
    def load_csv_lazy(path: str) -> pl.LazyFrame:
        return pl.scan_csv(path, infer_schema_length=10000)

    @staticmethod
    def load_json(path: str) -> pl.DataFrame:
        return pl.read_json(path)

    @staticmethod
    def load_from_url_pandas(url: str) -> pd.DataFrame:
        """Fallback to pandas for HTTP URLs not directly supported by polars"""
        return pd.read_parquet(url, engine='pyarrow')

    @staticmethod
    def sample_for_profiling(df: pl.DataFrame, n: int = 100_000) -> pl.DataFrame:
        """
        For datasets > 1M rows, sample before profiling.
        Use stratified sampling if target column exists.
        """
        if df.height > n:
            return df.sample(n=n, seed=42)
        return df

    @staticmethod
    def schema_report(df: pl.DataFrame) -> None:
        """Print a rich schema overview."""
        print("=" * 60)
        print(f"SCHEMA REPORT  |  {df.height:,} rows × {df.width} columns")
        print("=" * 60)
        for col in df.schema:
            dtype = df.schema[col]
            null_count = df[col].null_count()
            null_pct = null_count / df.height * 100
            print(f"  {col:<35} {str(dtype):<20} nulls={null_pct:.1f}%")
        print()


# ================================================================================
# EXAMPLE: Load NYC Taxi Data
# ================================================================================

# --- Load via pandas (HTTP URL) then convert to Polars ---
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

print("Loading NYC Taxi dataset...")
pdf = pd.read_parquet(URL)
df = pl.from_pandas(pdf)

DataIngestion.schema_report(df)


# ================================================================================
# 2. BASIC DATASET STATISTICS (The "Quick Vitals")
# ================================================================================

class BasicStats:
    """
    Before any visualization, get the raw numbers.
    These are the vital signs of your dataset.
    """

    @staticmethod
    def describe_all(df: pl.DataFrame) -> pl.DataFrame:
        """
        Polars describe() covers: count, null_count, mean, std, min, max,
        and percentiles (25th, 50th, 75th) for ALL columns.
        """
        return df.describe()

    @staticmethod
    def memory_usage(df: pl.DataFrame) -> str:
        """Report memory usage - critical for large datasets."""
        estimated_bytes = df.estimated_size()
        return f"Memory: {estimated_bytes / 1024**2:.1f} MB"

    @staticmethod
    def cardinality_report(df: pl.DataFrame) -> pl.DataFrame:
        """
        Cardinality = number of unique values per column.
        High cardinality categoricals → bad for one-hot encoding
        Low cardinality numerics → might actually be ordinal/categorical
        """
        rows = []
        for col in df.columns:
            n_unique = df[col].n_unique()
            null_count = df[col].null_count()
            rows.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": n_unique,
                "null_count": null_count,
                "null_pct": round(null_count / df.height * 100, 2),
                "cardinality_ratio": round(n_unique / df.height, 4)
            })
        return pl.DataFrame(rows).sort("n_unique", descending=True)

    @staticmethod
    def detect_column_types(df: pl.DataFrame) -> dict:
        """
        Intelligent type detection beyond dtypes:
        - Binary columns (only 2 unique values)
        - High-cardinality strings (likely IDs)
        - Disguised categoricals (int with few unique values)
        - Date-like strings
        """
        report = {"binary": [], "id_like": [], "categorical": [],
                  "continuous": [], "datetime": [], "text": []}

        for col in df.columns:
            n_unique = df[col].n_unique()
            dtype = df[col].dtype

            if n_unique == 2:
                report["binary"].append(col)
            elif dtype == pl.Utf8:
                if n_unique > df.height * 0.5:
                    report["id_like"].append(col)
                elif n_unique <= 50:
                    report["categorical"].append(col)
                else:
                    report["text"].append(col)
            elif dtype in (pl.Date, pl.Datetime):
                report["datetime"].append(col)
            elif n_unique <= 20 and dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                report["categorical"].append(col)
            else:
                report["continuous"].append(col)

        return report


# --- Run on Taxi Data ---
print(BasicStats.memory_usage(df))
cardinality = BasicStats.cardinality_report(df)
print(cardinality)

col_types = BasicStats.detect_column_types(df)
print("Column types:", col_types)


# ================================================================================
# 3. MISSING VALUE ANALYSIS
# ================================================================================

class MissingValueAnalysis:
    """
    Missing data is never random. Patterns in missingness are features.
    Three types: MCAR (Missing Completely At Random), MAR (At Random),
    MNAR (Not At Random). Each requires a different imputation strategy.
    """

    @staticmethod
    def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
        """Comprehensive missing value summary with Polars."""
        rows = []
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                rows.append({
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": null_count,
                    "missing_pct": round(null_count / df.height * 100, 3),
                    "present_count": df.height - null_count
                })
        if not rows:
            return pl.DataFrame({"message": ["No missing values found!"]})
        return (pl.DataFrame(rows)
                .sort("missing_pct", descending=True))

    @staticmethod
    def missing_heatmap(df_pd: pd.DataFrame, sample_n: int = 5000) -> None:
        """
        missingno matrix: white = present, dark = missing.
        Vertical alignment of dark patches → correlated missingness (MAR/MNAR)
        """
        sample = df_pd.sample(min(sample_n, len(df_pd)), random_state=42)
        cols_with_missing = [c for c in sample.columns if sample[c].isna().any()]
        if cols_with_missing:
            msno.matrix(sample[cols_with_missing], figsize=(12, 6), fontsize=10)
            plt.title("Missing Value Matrix (white=present, dark=missing)")
            plt.tight_layout()
            plt.savefig("missing_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()

    @staticmethod
    def missing_correlation(df_pd: pd.DataFrame) -> go.Figure:
        """
        Compute correlation of missingness indicators.
        If col A and col B both missing together → MNAR/MAR signal.
        """
        miss_indicators = df_pd.isna().astype(int)
        cols_with_missing = miss_indicators.columns[miss_indicators.sum() > 0].tolist()
        if len(cols_with_missing) < 2:
            print("Not enough columns with missing values for correlation.")
            return

        corr = miss_indicators[cols_with_missing].corr()
        fig = px.imshow(
            corr,
            title="Missing Value Correlation (red=correlated missingness)",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        return fig

    @staticmethod
    def littles_mcar_test(df_pd: pd.DataFrame, cols: list) -> dict:
        """
        Little's MCAR test (simplified version).
        H0: Data is MCAR. p > 0.05 → MCAR (random).
        p < 0.05 → MAR or MNAR (non-random, needs careful imputation).
        """
        # Approximate using chi2 on missingness patterns
        miss_matrix = df_pd[cols].isna().astype(int)
        n, k = miss_matrix.shape
        chi2_stat = 0
        for col in cols:
            obs = miss_matrix[col].values
            expected = obs.mean()
            chi2_stat += ((obs - expected) ** 2 / (expected + 1e-10)).sum()
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=k)
        return {
            "chi2_statistic": round(chi2_stat, 4),
            "p_value": round(p_value, 6),
            "is_mcar": p_value > 0.05,
            "interpretation": "MCAR (safe for listwise deletion)" if p_value > 0.05
                              else "NOT MCAR - use MAR/MNAR imputation strategies"
        }

    @staticmethod
    def plotly_missing_bar(df: pl.DataFrame) -> go.Figure:
        """Interactive bar chart of missing percentages."""
        missing = []
        for col in df.columns:
            null_pct = df[col].null_count() / df.height * 100
            if null_pct > 0:
                missing.append({"column": col, "missing_pct": round(null_pct, 2)})

        if not missing:
            print("No missing values!")
            return

        miss_df = pd.DataFrame(missing).sort_values("missing_pct", ascending=False)
        fig = px.bar(
            miss_df,
            x="column", y="missing_pct",
            title="Missing Value Percentage by Column",
            labels={"missing_pct": "Missing %"},
            color="missing_pct",
            color_continuous_scale="Reds",
            text="missing_pct"
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False)
        return fig


# --- Missing Analysis on Taxi Data ---
missing_summary = MissingValueAnalysis.missing_summary(df)
print(missing_summary)
fig_missing = MissingValueAnalysis.plotly_missing_bar(df)
if fig_missing:
    fig_missing.write_html("missing_bar.html")
    fig_missing.write_image("missing_bar.png", scale=2)


# ================================================================================
# 4. UNIVARIATE ANALYSIS
# ================================================================================

class UnivariateAnalysis:
    """
    Analyze each variable in isolation.
    Goal: understand the distribution, shape, spread, and outliers.
    """

    # --- Numerical ---

    @staticmethod
    def distribution_panel(df: pl.DataFrame, num_cols: list, ncols: int = 3) -> go.Figure:
        """
        Grid of histograms + KDE for all numeric columns.
        Reveals: skewness, bimodality, bounded distributions, spikes.
        """
        nrows = (len(num_cols) + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=num_cols)

        for i, col in enumerate(num_cols):
            row, c = i // ncols + 1, i % ncols + 1
            data = df[col].drop_nulls().to_numpy()
            # Remove extreme outliers for visualization (99.9th percentile cap)
            p999 = np.percentile(data, 99.9)
            p001 = np.percentile(data, 0.1)
            data_clipped = data[(data >= p001) & (data <= p999)]

            fig.add_trace(
                go.Histogram(x=data_clipped, name=col, nbinsx=50,
                             showlegend=False, marker_color='#378ADD',
                             opacity=0.75),
                row=row, col=c
            )

        fig.update_layout(
            height=350 * nrows,
            title_text="Distribution of Numerical Features",
            showlegend=False
        )
        return fig

    @staticmethod
    def five_number_summary(df: pl.DataFrame, col: str) -> dict:
        """
        The classic: min, Q1, median, Q3, max + extras.
        Also: skewness, kurtosis, CV (coefficient of variation).
        """
        series = df[col].drop_nulls()
        vals = series.to_numpy()
        return {
            "column": col,
            "count": len(vals),
            "min": float(np.min(vals)),
            "Q1": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "Q3": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "IQR": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "skewness": float(stats.skew(vals)),
            "kurtosis": float(stats.kurtosis(vals)),
            "CV": float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else None,
        }

    @staticmethod
    def normality_tests(series: np.ndarray, col_name: str) -> dict:
        """
        Multiple normality tests since each has different strengths:
        - Shapiro-Wilk: best for n < 5000
        - D'Agostino-Pearson: works for larger n, tests skew+kurtosis
        - Kolmogorov-Smirnov: compares with theoretical normal
        """
        sample = series[~np.isnan(series)]
        if len(sample) > 5000:
            sample = np.random.choice(sample, 5000, replace=False)

        results = {"column": col_name, "n_sampled": len(sample)}

        # Shapiro-Wilk
        stat, p = shapiro(sample[:min(len(sample), 5000)])
        results["shapiro_wilk"] = {"statistic": round(stat, 4), "p_value": round(p, 6),
                                    "is_normal": p > 0.05}
        # D'Agostino-Pearson
        stat, p = normaltest(sample)
        results["dagostino_pearson"] = {"statistic": round(stat, 4), "p_value": round(p, 6),
                                         "is_normal": p > 0.05}
        # KS Test
        standardized = (sample - sample.mean()) / sample.std()
        stat, p = kstest(standardized, 'norm')
        results["ks_test"] = {"statistic": round(stat, 4), "p_value": round(p, 6),
                               "is_normal": p > 0.05}
        return results

    @staticmethod
    def qq_plot(series: np.ndarray, col_name: str) -> go.Figure:
        """
        QQ plot: compare sample quantiles to theoretical normal quantiles.
        Points on diagonal → normal. S-shape → heavy tails. Curve → skewed.
        """
        sample = series[~np.isnan(series)]
        if len(sample) > 10000:
            sample = np.random.choice(sample, 10000, replace=False)

        (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                                  name='Sample Quantiles',
                                  marker=dict(size=3, color='#378ADD', opacity=0.5)))
        # Reference line
        x_line = np.array([osm.min(), osm.max()])
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept,
                                  mode='lines', name='Normal Reference',
                                  line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"QQ Plot: {col_name}",
                          xaxis_title="Theoretical Quantiles",
                          yaxis_title="Sample Quantiles")
        return fig

    @staticmethod
    def violin_box_plot(df: pl.DataFrame, col: str, group_col: str = None) -> go.Figure:
        """
        Violin + box overlay: best of both worlds.
        Violin → distribution shape, Box → quartiles & outliers.
        """
        vals = df[col].drop_nulls().to_numpy()
        p99 = np.percentile(vals, 99)
        p1 = np.percentile(vals, 1)

        pdf_temp = df.select([col, group_col] if group_col else [col]).to_pandas()
        pdf_temp = pdf_temp[(pdf_temp[col] >= p1) & (pdf_temp[col] <= p99)]

        if group_col:
            fig = px.violin(pdf_temp, y=col, x=group_col, box=True,
                            title=f"Distribution of {col} by {group_col}",
                            color=group_col)
        else:
            fig = px.violin(pdf_temp, y=col, box=True,
                            title=f"Distribution of {col}")
        return fig

    # --- Categorical ---

    @staticmethod
    def frequency_table(df: pl.DataFrame, col: str, top_n: int = 20) -> pl.DataFrame:
        """Frequency table with count, percentage, cumulative percentage."""
        freq = (df
                .group_by(col)
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(top_n)
                .with_columns([
                    (pl.col("count") / df.height * 100).round(2).alias("pct"),
                    (pl.col("count").cum_sum() / df.height * 100).round(2).alias("cum_pct")
                ]))
        return freq

    @staticmethod
    def pareto_chart(df: pl.DataFrame, col: str, top_n: int = 15) -> go.Figure:
        """
        Pareto chart: bars (frequency) + line (cumulative %).
        The 80/20 rule visualized. Identifies the vital few categories.
        """
        freq = UnivariateAnalysis.frequency_table(df, col, top_n).to_pandas()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=freq[col].astype(str), y=freq["count"],
                   name="Frequency", marker_color='#378ADD'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=freq[col].astype(str), y=freq["cum_pct"],
                       name="Cumulative %", mode='lines+markers',
                       line=dict(color='#D85A30', width=2)),
            secondary_y=True
        )
        fig.add_hline(y=80, line_dash="dash", line_color="gray",
                      annotation_text="80%", secondary_y=True)
        fig.update_layout(title=f"Pareto Chart: {col}")
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
        return fig

    @staticmethod
    def treemap_categorical(df: pl.DataFrame, col: str) -> go.Figure:
        """Treemap: area-proportional visualization of category frequencies."""
        freq = UnivariateAnalysis.frequency_table(df, col, top_n=30).to_pandas()
        fig = px.treemap(freq, path=[col], values="count",
                         title=f"Treemap: {col} Frequency",
                         color="count", color_continuous_scale="Blues")
        return fig


# --- Run on Taxi Data ---
# Numeric columns to analyze
num_cols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'passenger_count']
num_cols_exist = [c for c in num_cols if c in df.columns]

# Distribution panel
fig_dist = UnivariateAnalysis.distribution_panel(df, num_cols_exist, ncols=3)
fig_dist.write_image("distributions.png", scale=2, width=1400, height=800)

# Five number summary for fare_amount
summary = UnivariateAnalysis.five_number_summary(df, 'fare_amount')
print("Fare Amount Summary:", summary)

# Normality test
fare_vals = df['fare_amount'].drop_nulls().to_numpy()
normality = UnivariateAnalysis.normality_tests(fare_vals, 'fare_amount')
print("Normality Tests:", normality)

# QQ plot
fig_qq = UnivariateAnalysis.qq_plot(fare_vals, 'fare_amount')
fig_qq.write_image("qq_plot.png", scale=2)

# Violin/box
fig_violin = UnivariateAnalysis.violin_box_plot(df, 'fare_amount')
fig_violin.write_image("violin_fare.png", scale=2)


# ================================================================================
# 5. BIVARIATE ANALYSIS
# ================================================================================

class BivariateAnalysis:
    """
    Analyze relationships between pairs of variables.
    Numerical×Numerical, Numerical×Categorical, Categorical×Categorical.
    """

    # --- Numerical × Numerical ---

    @staticmethod
    def correlation_matrix(df: pl.DataFrame, num_cols: list,
                           method: str = "pearson") -> go.Figure:
        """
        Correlation heatmap. Methods:
        - Pearson: linear relationships (sensitive to outliers)
        - Spearman: monotonic relationships (rank-based, robust)
        - Kendall: concordance (best for small n, ordinal data)
        """
        pdf = df.select(num_cols).to_pandas()
        corr = pdf.corr(method=method)

        # Create mask for upper triangle to avoid redundancy
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.copy()
        corr_masked[mask] = None

        fig = px.imshow(
            corr_masked,
            title=f"Correlation Matrix ({method.capitalize()})",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=".2f"
        )
        fig.update_traces(textfont=dict(size=9))
        return fig

    @staticmethod
    def scatter_matrix(df: pl.DataFrame, cols: list, color_col: str = None,
                       sample_n: int = 5000) -> go.Figure:
        """
        Scatter plot matrix (SPLOM): all pairwise scatter plots.
        Critical for detecting: linear/non-linear relationships, clusters,
        outliers, interaction effects.
        """
        pdf = df.select(cols + ([color_col] if color_col else [])).to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        fig = px.scatter_matrix(pdf, dimensions=cols,
                                color=color_col,
                                title="Scatter Plot Matrix",
                                opacity=0.4)
        fig.update_traces(diagonal_visible=False, marker=dict(size=2))
        return fig

    @staticmethod
    def hexbin_plot(df: pl.DataFrame, x: str, y: str,
                    sample_n: int = 50000) -> go.Figure:
        """
        Hexbin: better than scatter for >50K points (avoids overplotting).
        Color intensity = point density in each hexagon.
        """
        pdf = df.select([x, y]).drop_nulls().to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        # Remove extreme outliers
        for col in [x, y]:
            q99, q01 = pdf[col].quantile(0.99), pdf[col].quantile(0.01)
            pdf = pdf[(pdf[col] <= q99) & (pdf[col] >= q01)]

        fig = go.Figure(go.Histogram2dContour(
            x=pdf[x], y=pdf[y],
            colorscale='Blues',
            contours=dict(showlabels=True),
            name=f"{x} vs {y}"
        ))
        fig.update_layout(title=f"Density: {x} vs {y}",
                          xaxis_title=x, yaxis_title=y)
        return fig

    @staticmethod
    def regression_plot(df: pl.DataFrame, x: str, y: str,
                        degree: int = 1) -> go.Figure:
        """
        Scatter + regression line (linear or polynomial).
        Shows trend + confidence interval.
        """
        pdf = df.select([x, y]).drop_nulls().to_pandas()
        pdf = pdf.sample(min(5000, len(pdf)), random_state=42)

        for col in [x, y]:
            q99, q01 = pdf[col].quantile(0.99), pdf[col].quantile(0.01)
            pdf = pdf[(pdf[col] <= q99) & (pdf[col] >= q01)]

        fig = px.scatter(pdf, x=x, y=y, opacity=0.3,
                         trendline="ols" if degree == 1 else "lowess",
                         title=f"{y} vs {x} (degree={degree})")
        return fig

    # --- Numerical × Categorical ---

    @staticmethod
    def group_statistics(df: pl.DataFrame, num_col: str,
                         cat_col: str) -> pl.DataFrame:
        """
        Descriptive stats of a numerical column grouped by a categorical column.
        Key for understanding group-level differences.
        """
        return (df
                .group_by(cat_col)
                .agg([
                    pl.col(num_col).count().alias("count"),
                    pl.col(num_col).mean().round(3).alias("mean"),
                    pl.col(num_col).median().alias("median"),
                    pl.col(num_col).std().round(3).alias("std"),
                    pl.col(num_col).min().alias("min"),
                    pl.col(num_col).max().alias("max"),
                    pl.col(num_col).quantile(0.25).alias("Q1"),
                    pl.col(num_col).quantile(0.75).alias("Q3"),
                ])
                .sort("mean", descending=True))

    @staticmethod
    def grouped_violin(df: pl.DataFrame, num_col: str,
                       cat_col: str, top_n: int = 10) -> go.Figure:
        """Violin plots for each category. Best for distribution comparison."""
        top_cats = (df.group_by(cat_col)
                    .agg(pl.len().alias("n"))
                    .sort("n", descending=True)
                    .head(top_n)[cat_col].to_list())

        pdf = (df.filter(pl.col(cat_col).is_in(top_cats))
                 .select([num_col, cat_col]).to_pandas())

        q99 = pdf[num_col].quantile(0.99)
        q01 = pdf[num_col].quantile(0.01)
        pdf = pdf[(pdf[num_col] >= q01) & (pdf[num_col] <= q99)]

        fig = px.violin(pdf, x=cat_col, y=num_col, box=True,
                        color=cat_col,
                        title=f"{num_col} Distribution by {cat_col}",
                        points=False)
        return fig

    @staticmethod
    def anova_test(df: pl.DataFrame, num_col: str, cat_col: str) -> dict:
        """
        One-way ANOVA: are means significantly different across groups?
        H0: all group means equal.
        p < 0.05: at least one group is significantly different.
        Also compute Eta-squared (effect size): small=0.01, medium=0.06, large=0.14
        """
        groups = []
        for cat, group_df in df.group_by(cat_col):
            vals = group_df[num_col].drop_nulls().to_numpy()
            if len(vals) >= 5:
                groups.append(vals)

        f_stat, p_val = stats.f_oneway(*groups)
        # Eta-squared
        grand_mean = df[num_col].drop_nulls().mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((v - grand_mean)**2 for g in groups for v in g)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        return {
            "test": "One-way ANOVA",
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_val, 8),
            "significant": p_val < 0.05,
            "eta_squared": round(eta_sq, 4),
            "effect_size": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
        }

    @staticmethod
    def kruskal_wallis_test(df: pl.DataFrame, num_col: str,
                            cat_col: str) -> dict:
        """
        Non-parametric alternative to ANOVA.
        Use when data is NOT normally distributed.
        """
        groups = []
        for cat, group_df in df.group_by(cat_col):
            vals = group_df[num_col].drop_nulls().to_numpy()
            if len(vals) >= 5:
                groups.append(vals)

        stat, p = stats.kruskal(*groups)
        return {
            "test": "Kruskal-Wallis",
            "statistic": round(stat, 4),
            "p_value": round(p, 8),
            "significant": p < 0.05
        }

    # --- Categorical × Categorical ---

    @staticmethod
    def contingency_heatmap(df: pl.DataFrame, col1: str, col2: str,
                            top_n: int = 10, normalize: bool = True) -> go.Figure:
        """
        Heatmap of cross-tabulation. normalize=True shows row percentages.
        Reveals: associations, conditional distributions.
        """
        top1 = (df.group_by(col1).agg(pl.len())
                  .sort("len", descending=True).head(top_n)[col1].to_list())
        top2 = (df.group_by(col2).agg(pl.len())
                  .sort("len", descending=True).head(top_n)[col2].to_list())

        pdf = df.filter(
            pl.col(col1).is_in(top1) & pl.col(col2).is_in(top2)
        ).select([col1, col2]).to_pandas()

        ct = pd.crosstab(pdf[col1], pdf[col2], normalize='index' if normalize else False)
        fig = px.imshow(ct, title=f"Cross-tab: {col1} × {col2}",
                        color_continuous_scale="Blues",
                        text_auto=".1%" if normalize else ".0f",
                        aspect="auto")
        return fig

    @staticmethod
    def chi_square_test(df: pl.DataFrame, col1: str, col2: str) -> dict:
        """
        Chi-square test of independence.
        H0: the two categorical variables are independent.
        Also compute Cramér's V (effect size): 0=no assoc, 1=perfect assoc.
        """
        pdf = df.select([col1, col2]).drop_nulls().to_pandas()
        ct = pd.crosstab(pdf[col1], pdf[col2])
        chi2, p, dof, expected = chi2_contingency(ct)

        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        return {
            "test": "Chi-square",
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p, 8),
            "degrees_of_freedom": dof,
            "significant": p < 0.05,
            "cramers_v": round(cramers_v, 4),
            "association_strength": "strong" if cramers_v > 0.3 else
                                    "moderate" if cramers_v > 0.1 else "weak"
        }


# --- Bivariate Analysis on Taxi Data ---
num_cols_exist = [c for c in ['trip_distance', 'fare_amount', 'tip_amount',
                               'total_amount'] if c in df.columns]

# Correlation matrix (Spearman for robustness)
fig_corr = BivariateAnalysis.correlation_matrix(df, num_cols_exist, method='spearman')
fig_corr.write_image("correlation_matrix.png", scale=2)

# Hexbin density plot
if 'trip_distance' in df.columns and 'fare_amount' in df.columns:
    fig_hex = BivariateAnalysis.hexbin_plot(df, 'trip_distance', 'fare_amount')
    fig_hex.write_image("hexbin_dist_fare.png", scale=2)

# Scatter matrix
fig_splom = BivariateAnalysis.scatter_matrix(df, num_cols_exist)
fig_splom.write_image("scatter_matrix.png", scale=2, width=1200, height=1000)


# ================================================================================
# 6. MULTIVARIATE ANALYSIS
# ================================================================================

class MultivariateAnalysis:
    """
    Analyze relationships involving 3+ variables simultaneously.
    Dimensionality reduction, clustering patterns, feature importance.
    """

    @staticmethod
    def pca_analysis(df: pl.DataFrame, num_cols: list,
                     n_components: int = 10) -> dict:
        """
        PCA: reduce dimensionality, find axes of maximum variance.
        Key outputs:
        - Explained variance ratio (how much info each PC captures)
        - Loadings (which features contribute to each PC)
        - Scree plot (elbow to choose n_components)
        """
        pdf = df.select(num_cols).drop_nulls().to_pandas()
        pdf = pdf.sample(min(50000, len(pdf)), random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pdf)

        pca = PCA(n_components=min(n_components, len(num_cols)))
        X_pca = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(explained_var)

        # Scree plot
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained_var))],
                                    y=explained_var * 100, name="Explained Variance %"))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cum_var))],
                                        y=cum_var * 100, name="Cumulative %",
                                        line=dict(color='red')))
        fig_scree.update_layout(title="PCA Scree Plot", yaxis_title="Variance Explained (%)")

        # Loadings heatmap
        loadings = pd.DataFrame(pca.components_.T, index=num_cols,
                                 columns=[f"PC{i+1}" for i in range(pca.n_components_)])
        fig_loadings = px.imshow(loadings, title="PCA Loadings",
                                  color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, text_auto=".2f",
                                  aspect="auto")

        # Biplot (PC1 vs PC2)
        fig_biplot = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                 opacity=0.2,
                                 labels={"x": f"PC1 ({explained_var[0]*100:.1f}%)",
                                         "y": f"PC2 ({explained_var[1]*100:.1f}%)"},
                                 title="PCA Biplot (PC1 vs PC2)")

        return {
            "pca_object": pca,
            "X_pca": X_pca,
            "explained_variance": explained_var,
            "n_components_90pct": int(np.argmax(cum_var >= 0.90)) + 1,
            "fig_scree": fig_scree,
            "fig_loadings": fig_loadings,
            "fig_biplot": fig_biplot,
        }

    @staticmethod
    def tsne_plot(df: pl.DataFrame, num_cols: list,
                  color_col: str = None, sample_n: int = 5000,
                  perplexity: int = 30) -> go.Figure:
        """
        t-SNE: non-linear dimensionality reduction for visualization.
        Reveals: clusters, manifold structure, class separability.
        NOTE: t-SNE is slow. Use UMAP for >50K samples.
        Perplexity: 5-50 (higher = more global structure preserved).
        """
        cols = num_cols + ([color_col] if color_col else [])
        pdf = df.select(cols).drop_nulls().to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pdf[num_cols])

        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=42, n_iter=1000)
        X_2d = tsne.fit_transform(X_scaled)

        plot_df = pd.DataFrame({"tsne_1": X_2d[:, 0], "tsne_2": X_2d[:, 1]})
        if color_col:
            plot_df[color_col] = pdf[color_col].values

        fig = px.scatter(plot_df, x="tsne_1", y="tsne_2",
                          color=color_col if color_col else None,
                          title=f"t-SNE Visualization (perplexity={perplexity})",
                          opacity=0.5, size_max=4)
        return fig

    @staticmethod
    def parallel_coordinates(df: pl.DataFrame, num_cols: list,
                              color_col: str = None,
                              sample_n: int = 5000) -> go.Figure:
        """
        Parallel coordinates: visualize all dimensions simultaneously.
        Brush any axis to filter. Great for finding multi-dimensional patterns.
        Each line = one observation.
        """
        cols = num_cols + ([color_col] if color_col else [])
        pdf = df.select(cols).drop_nulls().to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        # Cap outliers for better visualization
        for col in num_cols:
            pdf[col] = pdf[col].clip(pdf[col].quantile(0.01), pdf[col].quantile(0.99))

        fig = px.parallel_coordinates(pdf, dimensions=num_cols,
                                       color=color_col,
                                       color_continuous_scale="Viridis",
                                       title="Parallel Coordinates Plot")
        return fig

    @staticmethod
    def vif_analysis(df: pl.DataFrame, num_cols: list) -> pd.DataFrame:
        """
        Variance Inflation Factor (VIF): detect multicollinearity.
        VIF = 1: no correlation with other features.
        VIF 1-5: moderate correlation (acceptable).
        VIF > 5: high multicollinearity (problematic).
        VIF > 10: severe multicollinearity (must address).
        Action: remove one of the correlated pair or use PCA.
        """
        pdf = df.select(num_cols).drop_nulls().to_pandas()
        pdf = pdf.sample(min(10000, len(pdf)), random_state=42)
        pdf = pdf.dropna()

        X = sm.add_constant(pdf)
        vif_data = pd.DataFrame()
        vif_data["feature"] = num_cols
        vif_data["VIF"] = [variance_inflation_factor(X.values, i+1)
                            for i in range(len(num_cols))]
        vif_data["severity"] = vif_data["VIF"].apply(
            lambda v: "severe" if v > 10 else "high" if v > 5 else
                      "moderate" if v > 2 else "low"
        )
        return vif_data.sort_values("VIF", descending=True) if hasattr(
            vif_data, 'sort_values') else vif_data.sort_values("VIF", ascending=False)

    @staticmethod
    def radar_chart(group_stats: pd.DataFrame, groups: list,
                    metrics: list) -> go.Figure:
        """
        Radar/Spider chart: compare multiple groups across multiple metrics.
        Great for: customer segments, store performance, model comparison.
        """
        fig = go.Figure()
        for group in groups:
            vals = group_stats[group_stats.iloc[:, 0] == group][metrics].values.flatten()
            fig.add_trace(go.Scatterpolar(
                r=np.append(vals, vals[0]),
                theta=metrics + [metrics[0]],
                fill='toself', name=str(group), opacity=0.6
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Radar Chart: Multi-metric Group Comparison"
        )
        return fig


# --- PCA on Taxi Data ---
num_cols_pca = [c for c in ['trip_distance', 'fare_amount', 'tip_amount',
                              'total_amount', 'tolls_amount'] if c in df.columns]
pca_results = MultivariateAnalysis.pca_analysis(df, num_cols_pca)
pca_results["fig_scree"].write_image("pca_scree.png", scale=2)
pca_results["fig_loadings"].write_image("pca_loadings.png", scale=2)
pca_results["fig_biplot"].write_image("pca_biplot.png", scale=2, width=800)
print(f"Components for 90% variance: {pca_results['n_components_90pct']}")

# VIF
vif = MultivariateAnalysis.vif_analysis(df, num_cols_pca)
print("VIF Analysis:\n", vif)

# Parallel coordinates
fig_pc = MultivariateAnalysis.parallel_coordinates(df, num_cols_pca)
fig_pc.write_image("parallel_coords.png", scale=2, width=1200, height=600)


# ================================================================================
# 7. OUTLIER DETECTION
# ================================================================================

class OutlierDetection:
    """
    Outliers affect models differently:
    - Linear models: severely (high leverage points)
    - Tree models: less affected
    - Neural networks: moderate impact

    Types:
    - Point outliers: single anomalous value
    - Contextual outliers: normal globally, abnormal in context
    - Collective outliers: group of anomalous values
    """

    @staticmethod
    def iqr_method(df: pl.DataFrame, col: str) -> dict:
        """
        IQR method (Tukey fences).
        Lower fence: Q1 - 1.5*IQR
        Upper fence: Q3 + 1.5*IQR
        Works well for: roughly symmetric distributions.
        Extended fence (3*IQR) for extreme outliers.
        """
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        lower_ext = q1 - 3 * iqr
        upper_ext = q3 + 3 * iqr

        n_outliers = df.filter(
            (pl.col(col) < lower) | (pl.col(col) > upper)
        ).height
        n_extreme = df.filter(
            (pl.col(col) < lower_ext) | (pl.col(col) > upper_ext)
        ).height

        return {
            "method": "IQR",
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_fence": lower, "upper_fence": upper,
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / df.height * 100, 3),
            "n_extreme_outliers": n_extreme
        }

    @staticmethod
    def zscore_method(df: pl.DataFrame, col: str,
                      threshold: float = 3.0) -> dict:
        """
        Z-score method: |z| > threshold = outlier.
        Assumes normality - use Modified Z-score (MAD) for skewed data.
        threshold=2: 5% false positive | threshold=3: 0.3% | threshold=3.5: 0.05%
        """
        mean = df[col].mean()
        std = df[col].std()
        z_scores = ((df[col] - mean) / std).abs()
        n_outliers = (z_scores > threshold).sum()

        # Modified Z-score (MAD-based, robust)
        median = df[col].median()
        mad = (df[col] - median).abs().median()
        modified_z = (0.6745 * (df[col] - median) / mad).abs()
        n_modified = (modified_z > threshold).sum()

        return {
            "method": "Z-Score",
            "threshold": threshold,
            "n_outliers_zscore": int(n_outliers),
            "n_outliers_modified_zscore": int(n_modified),
            "pct_outliers": round(n_outliers / df.height * 100, 3)
        }

    @staticmethod
    def isolation_forest(df: pl.DataFrame, num_cols: list,
                         contamination: float = 0.05,
                         sample_n: int = 50000) -> pl.DataFrame:
        """
        Isolation Forest: tree-based anomaly detection.
        Isolates anomalies (few splits needed) from normal points.
        contamination: expected % of outliers (0.01 to 0.5).
        Returns: DataFrame with anomaly_score and is_outlier columns.
        """
        pdf = df.select(num_cols).drop_nulls().to_pandas()
        idx = pdf.index
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        iforest = IsolationForest(n_estimators=100,
                                   contamination=contamination,
                                   random_state=42, n_jobs=-1)
        scores = iforest.fit_predict(pdf)
        anomaly_scores = iforest.decision_function(pdf)

        result = pdf.copy()
        result['is_outlier'] = (scores == -1)
        result['anomaly_score'] = anomaly_scores
        return pl.from_pandas(result)

    @staticmethod
    def outlier_boxplot(df: pl.DataFrame, num_cols: list) -> go.Figure:
        """
        Box plots for multiple columns side-by-side.
        Each point beyond whiskers is a potential outlier.
        Standardize values for fair comparison.
        """
        pdf = df.select(num_cols).drop_nulls().to_pandas()
        pdf = pdf.sample(min(10000, len(pdf)), random_state=42)

        # Standardize for comparison
        scaler = StandardScaler()
        pdf_scaled = pd.DataFrame(
            scaler.fit_transform(pdf), columns=num_cols
        )
        pdf_melted = pdf_scaled.melt(var_name="Feature", value_name="Standardized Value")

        fig = px.box(pdf_melted, x="Feature", y="Standardized Value",
                     title="Outlier Detection: Standardized Box Plots",
                     color="Feature", points="outliers")
        return fig

    @staticmethod
    def outlier_summary_report(df: pl.DataFrame, num_cols: list) -> pd.DataFrame:
        """Consolidated outlier report across all methods."""
        rows = []
        for col in num_cols:
            iqr_res = OutlierDetection.iqr_method(df, col)
            z_res = OutlierDetection.zscore_method(df, col)
            rows.append({
                "column": col,
                "iqr_outliers_pct": iqr_res["pct_outliers"],
                "iqr_lower_fence": round(iqr_res["lower_fence"], 2),
                "iqr_upper_fence": round(iqr_res["upper_fence"], 2),
                "zscore_outliers_pct": z_res["pct_outliers"],
                "extreme_outliers": iqr_res["n_extreme_outliers"]
            })
        return pd.DataFrame(rows).sort_values("iqr_outliers_pct", ascending=False)


# --- Outlier Detection on Taxi Data ---
iqr_fare = OutlierDetection.iqr_method(df, 'fare_amount')
print("IQR Outliers in fare_amount:", iqr_fare)

outlier_report = OutlierDetection.outlier_summary_report(df, num_cols_pca)
print("Outlier Report:\n", outlier_report)

fig_box = OutlierDetection.outlier_boxplot(df, num_cols_pca)
fig_box.write_image("outlier_boxplot.png", scale=2)


# ================================================================================
# 8. TIME SERIES EDA
# ================================================================================

class TimeSeriesEDA:
    """
    Temporal patterns analysis. Essential for any dataset with timestamps.
    Components: Trend + Seasonality + Cyclicality + Residual.
    """

    @staticmethod
    def temporal_features(df: pl.DataFrame, dt_col: str) -> pl.DataFrame:
        """
        Extract temporal features from a datetime column.
        These are often the most predictive features!
        """
        return df.with_columns([
            pl.col(dt_col).dt.year().alias("year"),
            pl.col(dt_col).dt.month().alias("month"),
            pl.col(dt_col).dt.day().alias("day"),
            pl.col(dt_col).dt.weekday().alias("weekday"),  # 0=Mon, 6=Sun
            pl.col(dt_col).dt.hour().alias("hour"),
            pl.col(dt_col).dt.minute().alias("minute"),
            (pl.col(dt_col).dt.weekday() >= 5).alias("is_weekend"),
            pl.col(dt_col).dt.quarter().alias("quarter"),
        ])

    @staticmethod
    def time_aggregation_plot(df: pl.DataFrame, dt_col: str,
                               value_col: str, freq: str = "1h") -> go.Figure:
        """
        Aggregate metrics over time intervals.
        freq options: "1m", "1h", "1d", "1w", "1mo"
        """
        agg = (df
               .with_columns(pl.col(dt_col).dt.truncate(freq).alias("time_bucket"))
               .group_by("time_bucket")
               .agg([
                   pl.col(value_col).count().alias("count"),
                   pl.col(value_col).mean().alias("mean"),
                   pl.col(value_col).sum().alias("sum")
               ])
               .sort("time_bucket")
               .to_pandas())

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["Count", "Mean Value"])
        fig.add_trace(go.Bar(x=agg["time_bucket"], y=agg["count"],
                              name="Count"), row=1, col=1)
        fig.add_trace(go.Scatter(x=agg["time_bucket"], y=agg["mean"],
                                  name="Mean", line=dict(color='#D85A30')),
                      row=2, col=1)
        fig.update_layout(title=f"Time Series: {value_col} by {freq}")
        return fig

    @staticmethod
    def heatmap_calendar(df: pl.DataFrame, dt_col: str,
                          value_col: str) -> go.Figure:
        """
        Calendar heatmap: hour × day-of-week.
        Reveals: rush hours, weekly seasonality, business patterns.
        """
        agg = (df
               .with_columns([
                   pl.col(dt_col).dt.hour().alias("hour"),
                   pl.col(dt_col).dt.weekday().alias("weekday")
               ])
               .group_by(["hour", "weekday"])
               .agg(pl.col(value_col).mean().alias("mean_value"))
               .sort(["weekday", "hour"])
               .to_pandas())

        pivot = agg.pivot_table(index="hour", columns="weekday",
                                 values="mean_value", aggfunc="mean")
        pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig = px.imshow(pivot, title=f"{value_col}: Hour × Day-of-Week Heatmap",
                         color_continuous_scale="Viridis",
                         labels=dict(x="Day of Week", y="Hour of Day",
                                     color=f"Mean {value_col}"),
                         aspect="auto")
        return fig

    @staticmethod
    def seasonal_decomposition(series: pd.Series, period: int = 24,
                                model: str = "additive") -> go.Figure:
        """
        Decompose time series into Trend + Seasonal + Residual.
        model='additive': value = trend + seasonal + residual
        model='multiplicative': value = trend × seasonal × residual
        Use multiplicative when variance increases with level.
        """
        result = seasonal_decompose(series.dropna(), model=model, period=period)

        fig = make_subplots(rows=4, cols=1,
                             subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
                             shared_xaxes=True)

        for i, (name, component) in enumerate([
            ("Original", series),
            ("Trend", result.trend),
            ("Seasonal", result.seasonal),
            ("Residual", result.resid)
        ], 1):
            fig.add_trace(go.Scatter(y=component.values, name=name,
                                      line=dict(width=1)), row=i, col=1)

        fig.update_layout(height=800,
                           title=f"Seasonal Decomposition ({model})")
        return fig

    @staticmethod
    def autocorrelation_plot(series: pd.Series, n_lags: int = 48) -> go.Figure:
        """
        ACF & PACF plots for time series.
        ACF: correlation with lagged versions of itself.
        PACF: partial correlation (removes intermediate lag effects).
        Use for: AR/MA order selection in ARIMA models.
        Significant lag at k: correlation at lag k is real, not noise.
        """
        from statsmodels.tsa.stattools import acf, pacf

        acf_vals, acf_conf = acf(series.dropna(), nlags=n_lags,
                                   alpha=0.05, fft=True)
        pacf_vals, pacf_conf = pacf(series.dropna(), nlags=n_lags, alpha=0.05)

        lags = list(range(n_lags + 1))
        fig = make_subplots(rows=2, cols=1,
                             subplot_titles=["ACF (Autocorrelation)",
                                              "PACF (Partial Autocorrelation)"])

        for row, (vals, conf, name) in enumerate([
            (acf_vals, acf_conf, "ACF"),
            (pacf_vals, pacf_conf, "PACF")
        ], 1):
            # Confidence bounds
            upper = conf[:, 1] - vals
            lower = vals - conf[:, 0]

            fig.add_trace(go.Bar(x=lags, y=vals, name=name,
                                  error_y=dict(type='data', array=upper,
                                               arrayminus=lower)),
                           row=row, col=1)
            fig.add_hline(y=0, row=row, col=1)

        fig.update_layout(height=600, title="ACF and PACF Analysis")
        return fig


# --- Time Series EDA on Taxi Data ---
dt_col = 'tpep_pickup_datetime'
if dt_col in df.columns:
    # Add temporal features
    df = TimeSeriesEDA.temporal_features(df, dt_col)

    # Trip volume by hour
    fig_ts = TimeSeriesEDA.time_aggregation_plot(df, dt_col, 'trip_distance', freq='1h')
    fig_ts.write_image("time_series_hourly.png", scale=2, width=1400, height=600)

    # Heatmap: hour × day
    fig_cal = TimeSeriesEDA.heatmap_calendar(df, dt_col, 'fare_amount')
    fig_cal.write_image("calendar_heatmap.png", scale=2)


# ================================================================================
# 9. GEOSPATIAL EDA
# ================================================================================

class GeospatialEDA:
    """
    For datasets with lat/lon coordinates.
    NYC Taxi is a perfect example - pickup/dropoff locations.
    """

    @staticmethod
    def density_map(df: pl.DataFrame, lat_col: str, lon_col: str,
                    value_col: str = None, sample_n: int = 50000) -> go.Figure:
        """
        Mapbox density heatmap of coordinates.
        Reveals: hotspots, corridors, spatial clusters.
        """
        pdf = df.select([lat_col, lon_col] + ([value_col] if value_col else []))
        # Filter valid coordinates
        pdf = pdf.filter(
            pl.col(lat_col).is_between(40.4, 41.0) &
            pl.col(lon_col).is_between(-74.3, -73.6)
        ).to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        fig = px.density_mapbox(
            pdf, lat=lat_col, lon=lon_col,
            z=value_col,
            radius=5, zoom=10,
            center={"lat": 40.7128, "lon": -74.0060},
            mapbox_style="carto-positron",
            title=f"Density Map: {value_col or 'Trip Count'}"
        )
        return fig

    @staticmethod
    def scatter_map(df: pl.DataFrame, lat_col: str, lon_col: str,
                    color_col: str = None, sample_n: int = 5000) -> go.Figure:
        """Point map colored by a metric. Good for sparse data."""
        pdf = df.select([lat_col, lon_col] + ([color_col] if color_col else []))
        pdf = pdf.filter(
            pl.col(lat_col).is_between(40.4, 41.0) &
            pl.col(lon_col).is_between(-74.3, -73.6)
        ).to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)

        fig = px.scatter_mapbox(
            pdf, lat=lat_col, lon=lon_col,
            color=color_col,
            zoom=10, size_max=5,
            center={"lat": 40.7128, "lon": -74.0060},
            mapbox_style="carto-positron",
            opacity=0.5,
            title=f"Scatter Map: {color_col}"
        )
        return fig

    @staticmethod
    def spatial_stats(df: pl.DataFrame, lat_col: str, lon_col: str) -> dict:
        """Compute center of mass, spread, and bounding box."""
        lat = df[lat_col].drop_nulls().to_numpy()
        lon = df[lon_col].drop_nulls().to_numpy()

        # Filter valid
        mask = (lat > 40.4) & (lat < 41.0) & (lon > -74.3) & (lon < -73.6)
        lat, lon = lat[mask], lon[mask]

        return {
            "centroid": (round(lat.mean(), 4), round(lon.mean(), 4)),
            "lat_range": (round(lat.min(), 4), round(lat.max(), 4)),
            "lon_range": (round(lon.min(), 4), round(lon.max(), 4)),
            "lat_std": round(lat.std(), 4),
            "lon_std": round(lon.std(), 4),
            "n_valid_points": len(lat)
        }


# --- Geospatial on Taxi Data ---
if 'pickup_latitude' in df.columns:
    fig_map = GeospatialEDA.density_map(df, 'pickup_latitude', 'pickup_longitude')
    fig_map.write_html("pickup_density_map.html")

# For newer taxi data: PULocationID instead of lat/lon
if 'PULocationID' in df.columns:
    pickup_freq = (df.group_by('PULocationID')
                     .agg(pl.len().alias("pickup_count"))
                     .sort("pickup_count", descending=True))
    print("Top pickup zones:\n", pickup_freq.head(10))


# ================================================================================
# 10. FEATURE IMPORTANCE & TARGET RELATIONSHIP
# ================================================================================

class TargetAnalysis:
    """
    When a target variable exists, analyze each feature's relationship with it.
    This is the bridge between EDA and feature engineering.
    """

    @staticmethod
    def mutual_information_scores(df: pl.DataFrame, feature_cols: list,
                                   target_col: str,
                                   task: str = "regression") -> pd.DataFrame:
        """
        Mutual Information: measures any kind of dependency (not just linear).
        Works for: continuous, discrete, mixed features.
        MI = 0: independent. Higher = stronger dependency.
        """
        pdf = df.select(feature_cols + [target_col]).drop_nulls().to_pandas()
        X = pdf[feature_cols]
        y = pdf[target_col]

        if task == "regression":
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42)

        mi_df = pd.DataFrame({"feature": feature_cols, "mi_score": mi_scores})
        mi_df = mi_df.sort_values("mi_score", ascending=False)

        fig = px.bar(mi_df, x="mi_score", y="feature", orientation='h',
                     title="Mutual Information with Target",
                     color="mi_score", color_continuous_scale="Blues")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})

        return mi_df, fig

    @staticmethod
    def random_forest_importance(df: pl.DataFrame, feature_cols: list,
                                  target_col: str, task: str = "regression",
                                  sample_n: int = 50000) -> pd.DataFrame:
        """
        Random Forest feature importance.
        - Mean Decrease Impurity (MDI): fast but biased toward high-cardinality
        - Permutation Importance: slower but unbiased (preferred)
        """
        pdf = df.select(feature_cols + [target_col]).drop_nulls().to_pandas()
        pdf = pdf.sample(min(sample_n, len(pdf)), random_state=42)
        X = pdf[feature_cols]
        y = pdf[target_col]

        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                     random_state=42, max_depth=5)
        rf.fit(X, y)

        mdi_imp = pd.DataFrame({
            "feature": feature_cols,
            "mdi_importance": rf.feature_importances_
        }).sort_values("mdi_importance", ascending=False)

        # Permutation importance (more reliable)
        perm_imp = permutation_importance(rf, X, y, n_repeats=5,
                                           random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({
            "feature": feature_cols,
            "perm_mean": perm_imp.importances_mean,
            "perm_std": perm_imp.importances_std
        }).sort_values("perm_mean", ascending=False)

        return mdi_imp, perm_df

    @staticmethod
    def target_distribution_by_feature(df: pl.DataFrame,
                                        feature_col: str,
                                        target_col: str,
                                        bins: int = 10) -> go.Figure:
        """
        For regression: target mean/median per feature bin.
        For classification: target class distribution per feature value.
        """
        pdf = df.select([feature_col, target_col]).drop_nulls().to_pandas()

        if pdf[feature_col].dtype in ['int64', 'float64']:
            pdf['bin'] = pd.cut(pdf[feature_col], bins=bins)
            grouped = pdf.groupby('bin', observed=True)[target_col].agg(['mean', 'std', 'count'])
            grouped = grouped.reset_index()
            grouped['bin'] = grouped['bin'].astype(str)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=grouped['bin'], y=grouped['count'],
                                  name='Count', opacity=0.4,
                                  marker_color='lightblue'), secondary_y=True)
            fig.add_trace(go.Scatter(x=grouped['bin'], y=grouped['mean'],
                                      name=f'Mean {target_col}',
                                      mode='lines+markers',
                                      line=dict(color='red', width=2)),
                           secondary_y=False)
            fig.update_layout(title=f"Target '{target_col}' vs '{feature_col}'")
        else:
            grouped = pdf.groupby(feature_col)[target_col].mean().reset_index()
            fig = px.bar(grouped, x=feature_col, y=target_col,
                          title=f"Mean {target_col} by {feature_col}")

        return fig

    @staticmethod
    def point_biserial_correlation(df: pl.DataFrame,
                                    num_col: str, binary_col: str) -> dict:
        """
        Point-biserial correlation: numerical × binary.
        Equivalent to Pearson but one variable is binary.
        """
        pdf = df.select([num_col, binary_col]).drop_nulls().to_pandas()
        r, p = pointbiserialr(pdf[binary_col], pdf[num_col])
        return {
            "column": num_col, "binary_col": binary_col,
            "rpb": round(r, 4), "p_value": round(p, 6),
            "significant": p < 0.05
        }


# --- Target Analysis ---
# Create target: high-value trip (tip_pct > 20%)
if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
    df = df.with_columns([
        (pl.col('tip_amount') / (pl.col('fare_amount') + 1e-6) * 100).alias('tip_pct'),
    ])
    # Filter reasonable values
    df_clean = df.filter(
        (pl.col('fare_amount') > 2) & (pl.col('fare_amount') < 200) &
        (pl.col('trip_distance') > 0) & (pl.col('trip_distance') < 50)
    )

    feature_cols = [c for c in ['trip_distance', 'passenger_count',
                                  'hour', 'weekday', 'is_weekend']
                    if c in df_clean.columns]

    if feature_cols:
        mi_df, fig_mi = TargetAnalysis.mutual_information_scores(
            df_clean, feature_cols, 'tip_pct', task='regression')
        print("Mutual Information:\n", mi_df)
        fig_mi.write_image("mutual_information.png", scale=2)

        # Target distribution plot
        fig_td = TargetAnalysis.target_distribution_by_feature(
            df_clean, 'trip_distance', 'tip_pct')
        fig_td.write_image("target_vs_distance.png", scale=2)


# ================================================================================
# 11. DATA QUALITY AUDIT
# ================================================================================

class DataQualityAudit:
    """
    Industrial-scale datasets have messy quality issues.
    This module finds them all.
    """

    @staticmethod
    def detect_duplicates(df: pl.DataFrame,
                          subset: list = None) -> dict:
        """
        Full duplicates vs near-duplicates.
        Near-duplicates: same across key columns only.
        """
        total = df.height
        full_dups = total - df.unique().height
        result = {"total_rows": total, "full_duplicates": full_dups,
                  "full_dup_pct": round(full_dups / total * 100, 3)}

        if subset:
            key_dups = total - df.unique(subset=subset).height
            result["key_duplicates"] = key_dups
            result["key_dup_pct"] = round(key_dups / total * 100, 3)
            result["key_columns_used"] = subset

        return result

    @staticmethod
    def detect_impossible_values(df: pl.DataFrame) -> list:
        """
        Domain-specific impossible value checks.
        Extend with your domain knowledge!
        """
        issues = []

        # Negative values in non-negative columns
        for col in df.columns:
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                n_neg = (df[col] < 0).sum()
                if n_neg > 0:
                    col_name = col.lower()
                    if any(k in col_name for k in ['amount', 'distance',
                                                    'count', 'price', 'qty']):
                        issues.append({
                            "column": col,
                            "issue": "negative_values_in_non_negative_col",
                            "count": int(n_neg),
                            "pct": round(n_neg / df.height * 100, 3)
                        })

        # Future timestamps
        for col in df.columns:
            if df[col].dtype in (pl.Datetime, pl.Date):
                n_future = (df[col] > pl.lit(
                    "2025-01-01").str.strptime(pl.Date)).sum()
                if n_future > 0:
                    issues.append({"column": col, "issue": "future_dates",
                                    "count": int(n_future)})

        # Zero-variance columns
        for col in df.columns:
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                if df[col].n_unique() == 1:
                    issues.append({"column": col, "issue": "zero_variance",
                                    "count": df.height})

        return issues

    @staticmethod
    def value_range_validation(df: pl.DataFrame,
                                rules: dict) -> list:
        """
        Validate columns against expected ranges.
        rules = {'fare_amount': (0, 1000), 'passenger_count': (1, 8)}
        """
        violations = []
        for col, (lo, hi) in rules.items():
            if col not in df.columns:
                continue
            n_violations = df.filter(
                (pl.col(col) < lo) | (pl.col(col) > hi)
            ).height
            if n_violations > 0:
                violations.append({
                    "column": col,
                    "expected_range": (lo, hi),
                    "n_violations": n_violations,
                    "pct_violations": round(n_violations / df.height * 100, 3)
                })
        return violations

    @staticmethod
    def string_quality_check(df: pl.DataFrame,
                              str_cols: list) -> pd.DataFrame:
        """
        Check string columns for:
        - Trailing/leading whitespace
        - Mixed case issues
        - Empty strings (vs null)
        - Suspicious values
        """
        rows = []
        for col in str_cols:
            if col not in df.columns:
                continue
            series = df[col]
            n_empty = (series == "").sum()
            n_whitespace = series.filter(series.str.strip_chars() != series).len()
            n_null = series.null_count()
            rows.append({
                "column": col,
                "n_null": n_null,
                "n_empty_string": n_empty,
                "n_leading_trailing_ws": n_whitespace,
                "n_unique": series.n_unique(),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def generate_full_report(df: pl.DataFrame) -> dict:
        """
        One-call comprehensive quality report.
        """
        print("Running full data quality audit...")

        # Duplicates
        dups = DataQualityAudit.detect_duplicates(df)

        # Missing values
        missing = MissingValueAnalysis.missing_summary(df)

        # Impossible values
        impossible = DataQualityAudit.detect_impossible_values(df)

        # Value ranges (taxi-specific)
        range_rules = {}
        if 'fare_amount' in df.columns:
            range_rules['fare_amount'] = (-1, 1000)
        if 'trip_distance' in df.columns:
            range_rules['trip_distance'] = (0, 200)
        if 'passenger_count' in df.columns:
            range_rules['passenger_count'] = (0, 9)

        range_violations = DataQualityAudit.value_range_validation(df, range_rules)

        report = {
            "duplicates": dups,
            "missing_values": missing.to_dicts() if hasattr(missing, 'to_dicts') else [],
            "impossible_values": impossible,
            "range_violations": range_violations
        }

        print("=== DATA QUALITY REPORT ===")
        for k, v in report.items():
            print(f"\n{k.upper()}:")
            print(v)

        return report


# --- Data Quality Audit ---
quality_report = DataQualityAudit.generate_full_report(df)


# ================================================================================
# 12. AUTOMATED PROFILING (ydata-profiling)
# ================================================================================

def run_ydata_profiling(df_pd: pd.DataFrame, output_file: str = "profile_report.html",
                         minimal: bool = True) -> None:
    """
    ydata-profiling (formerly pandas-profiling): generates a full HTML report
    with correlations, distributions, missing values, and interactions.

    minimal=True for large datasets (skips expensive computations).
    minimal=False for full analysis (slow on large datasets).

    For datasets > 1M rows, always use minimal=True or sample first.
    """
    try:
        from ydata_profiling import ProfileReport

        sample = df_pd.sample(min(50000, len(df_pd)), random_state=42)
        profile = ProfileReport(
            sample,
            title="Automated EDA Report",
            minimal=minimal,
            explorative=not minimal,
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": False},
            },
            missing_diagrams={
                "heatmap": True,
                "dendrogram": True,
            }
        )
        profile.to_file(output_file)
        print(f"Profiling report saved: {output_file}")
    except ImportError:
        print("Install ydata-profiling: pip install ydata-profiling")


# run_ydata_profiling(pdf, "nyc_taxi_profile.html", minimal=True)


# ================================================================================
# 13. EDA FOR IMBALANCED DATASETS
# ================================================================================

class ImbalancedDataEDA:
    """
    For classification tasks with imbalanced targets.
    Imbalance ratio: majority/minority class size.
    """

    @staticmethod
    def class_distribution(df: pl.DataFrame, target_col: str) -> go.Figure:
        """Visualize class balance. Critical for classification tasks."""
        freq = (df.group_by(target_col)
                  .agg(pl.len().alias("count"))
                  .sort("count", descending=True)
                  .to_pandas())
        freq["pct"] = freq["count"] / freq["count"].sum() * 100

        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Count", "Percentage"])
        fig.add_trace(go.Bar(x=freq[target_col].astype(str), y=freq["count"],
                              name="Count", marker_color='#378ADD'), row=1, col=1)
        fig.add_trace(go.Pie(labels=freq[target_col].astype(str),
                              values=freq["pct"],
                              name="Distribution"), row=1, col=2)
        fig.update_layout(title=f"Class Distribution: {target_col}")
        return fig

    @staticmethod
    def stratified_stats(df: pl.DataFrame, target_col: str,
                          feature_cols: list) -> dict:
        """
        Compute feature statistics separately for each class.
        Reveals: which features differ most between classes.
        """
        results = {}
        for cls_val, cls_df in df.group_by(target_col):
            results[str(cls_val)] = {
                col: {
                    "mean": cls_df[col].mean(),
                    "median": cls_df[col].median(),
                    "std": cls_df[col].std()
                }
                for col in feature_cols
                if cls_df[col].dtype in (pl.Float32, pl.Float64,
                                          pl.Int32, pl.Int64)
            }
        return results

    @staticmethod
    def imbalance_ratio(df: pl.DataFrame, target_col: str) -> dict:
        freq = (df.group_by(target_col)
                  .agg(pl.len().alias("count"))
                  .sort("count", descending=True))
        counts = freq["count"].to_list()
        ratio = counts[0] / counts[-1] if counts[-1] > 0 else float('inf')
        return {
            "n_classes": len(counts),
            "majority_count": counts[0],
            "minority_count": counts[-1],
            "imbalance_ratio": round(ratio, 2),
            "severity": "severe" if ratio > 10 else
                        "moderate" if ratio > 3 else "mild"
        }


# ================================================================================
# 14. COMPREHENSIVE EDA PIPELINE (Put It All Together)
# ================================================================================

def run_full_eda_pipeline(df: pl.DataFrame,
                           target_col: str = None,
                           output_dir: str = "eda_outputs") -> None:
    """
    One-function EDA runner. Call this on any new dataset.
    Runs all analyses and saves outputs to disk.
    """
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"FULL EDA PIPELINE - {df.height:,} rows × {df.width} cols")
    print(f"{'='*60}\n")

    # 1. Basic stats
    print("[1/8] Basic Statistics...")
    DataIngestion.schema_report(df)
    print(BasicStats.memory_usage(df))
    cardinality = BasicStats.cardinality_report(df)
    print(cardinality.head(20))
    col_types = BasicStats.detect_column_types(df)
    print("Column types:", col_types)

    # 2. Missing values
    print("\n[2/8] Missing Value Analysis...")
    missing = MissingValueAnalysis.missing_summary(df)
    print(missing)

    # 3. Outliers
    print("\n[3/8] Outlier Detection...")
    num_cols = col_types.get("continuous", [])[:6]
    if num_cols:
        outlier_report = OutlierDetection.outlier_summary_report(df, num_cols)
        print(outlier_report)

    # 4. Univariate
    print("\n[4/8] Univariate Analysis...")
    if num_cols:
        for col in num_cols[:3]:
            summary = UnivariateAnalysis.five_number_summary(df, col)
            print(f"  {col}: mean={summary['mean']:.2f}, skew={summary['skewness']:.2f}")

    # 5. Bivariate
    print("\n[5/8] Bivariate Analysis...")
    if len(num_cols) >= 2:
        corr_fig = BivariateAnalysis.correlation_matrix(df, num_cols, "spearman")
        corr_fig.write_image(f"{output_dir}/correlation.png", scale=2)

    # 6. Data Quality
    print("\n[6/8] Data Quality Audit...")
    DataQualityAudit.generate_full_report(df)

    # 7. Target analysis (if target provided)
    if target_col and target_col in df.columns:
        print(f"\n[7/8] Target Analysis (target={target_col})...")
        feature_cols = [c for c in num_cols if c != target_col]
        if feature_cols:
            mi_df, mi_fig = TargetAnalysis.mutual_information_scores(
                df, feature_cols, target_col)
            mi_fig.write_image(f"{output_dir}/mutual_info.png", scale=2)
            print(mi_df.head(10))

    # 8. Summary
    print(f"\n[8/8] EDA Complete. Outputs in: {output_dir}/")
    print(f"      Rows: {df.height:,} | Cols: {df.width}")
    print(f"      Nulls: {sum(df[c].null_count() for c in df.columns):,}")
    print(f"      Dtypes: {dict(df.dtypes)}")


# ================================================================================
# 15. EDA CHECKLIST (Reference)
# ================================================================================

EDA_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════╗
║              INDUSTRIAL EDA CHECKLIST                            ║
╠══════════════════════════════════════════════════════════════════╣
║ □ INGESTION & SHAPE                                              ║
║   □ Rows, columns, memory usage                                  ║
║   □ Schema (dtypes, names)                                       ║
║   □ Sample first rows (df.head())                                ║
║   □ Column cardinality                                           ║
║   □ Detect column semantic types                                 ║
║                                                                  ║
║ □ DATA QUALITY                                                   ║
║   □ Missing values (count, %, pattern)                           ║
║   □ Duplicates (full + key-based)                                ║
║   □ Impossible values (negatives, future dates)                  ║
║   □ Value range validation                                       ║
║   □ String quality (whitespace, empty, case)                     ║
║   □ Data type mismatches                                         ║
║                                                                  ║
║ □ UNIVARIATE                                                     ║
║   □ Distribution histograms (all numeric)                        ║
║   □ 5-number summary (all numeric)                               ║
║   □ Skewness & kurtosis                                          ║
║   □ Normality tests (Shapiro/D'Agostino/KS)                     ║
║   □ Frequency tables (all categorical)                           ║
║   □ Pareto charts (categorical)                                  ║
║   □ Box plots / violin plots                                     ║
║   □ QQ plots (numeric)                                           ║
║                                                                  ║
║ □ BIVARIATE                                                      ║
║   □ Correlation matrix (Pearson + Spearman)                      ║
║   □ Scatter matrix / SPLOM                                       ║
║   □ Hex bin plots (dense numeric pairs)                          ║
║   □ Group statistics (numeric × categorical)                     ║
║   □ ANOVA / Kruskal-Wallis (numeric × categorical)              ║
║   □ Chi-square + Cramér's V (categorical × categorical)         ║
║   □ Contingency heatmaps                                         ║
║                                                                  ║
║ □ MULTIVARIATE                                                   ║
║   □ PCA (scree plot, loadings, biplot)                           ║
║   □ t-SNE / UMAP (cluster structure)                             ║
║   □ Parallel coordinates                                         ║
║   □ VIF (multicollinearity)                                      ║
║   □ Radar charts (group comparison)                              ║
║                                                                  ║
║ □ OUTLIER DETECTION                                              ║
║   □ IQR fences (all numeric)                                     ║
║   □ Z-score / Modified Z-score                                   ║
║   □ Isolation Forest (multivariate)                              ║
║   □ Visual inspection (box plots, scatter)                       ║
║                                                                  ║
║ □ TEMPORAL (if datetime cols exist)                              ║
║   □ Extract temporal features (hour, day, month)                 ║
║   □ Time series plot (rolling stats)                             ║
║   □ Hour × day heatmap                                           ║
║   □ Seasonal decomposition                                       ║
║   □ ACF/PACF plots                                               ║
║                                                                  ║
║ □ GEOSPATIAL (if lat/lon cols exist)                             ║
║   □ Scatter map                                                  ║
║   □ Density/heatmap                                              ║
║   □ Spatial statistics                                           ║
║                                                                  ║
║ □ TARGET ANALYSIS (if supervised task)                           ║
║   □ Class balance (classification)                               ║
║   □ Target distribution (regression)                             ║
║   □ Mutual information (all features)                            ║
║   □ Random Forest importance                                     ║
║   □ Feature vs target plots                                      ║
║   □ Point-biserial (binary features)                             ║
║                                                                  ║
║ □ DOCUMENTATION                                                  ║
║   □ Note every anomaly found                                     ║
║   □ List data quality issues for team                            ║
║   □ Suggest feature engineering ideas                            ║
║   □ Flag columns to drop (high missing, zero variance)           ║
║   □ Write EDA summary report                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

print(EDA_CHECKLIST)

print("\nAll outputs saved. EDA complete.")
