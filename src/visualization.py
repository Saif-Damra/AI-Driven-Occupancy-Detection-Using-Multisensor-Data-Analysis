"""
Visualization Module for Occupancy Detection
=============================================
Professional visualizations for sensor data EDA and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

COLORS = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#0891B2"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_occupancy_distribution(df, output_dir="outputs"):
    """Plot target variable distribution."""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Room Occupancy Distribution", fontweight="bold", fontsize=16)
    
    counts = df["Room_Occupancy_Count"].value_counts().sort_index()
    colors = COLORS[:len(counts)]
    
    axes[0].bar(counts.index.astype(str), counts.values, color=colors, edgecolor="white", linewidth=2)
    axes[0].set_title("Count by Occupancy Level")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_xlabel("Occupancy Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
    
    axes[1].pie(counts.values, labels=[f"{i} person(s)" for i in counts.index],
                autopct="%1.1f%%", colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportion")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "occupancy_distribution.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_sensor_variation_by_hour(df, sensor_type, output_dir="outputs"):
    """Plot sensor readings variation by hour."""
    ensure_dir(output_dir)
    
    # Prepare time feature if not present
    if "hours" not in df.columns:
        df = df.copy()
        df[["hours", "_m", "_s"]] = df["Time"].str.split(":", expand=True)
        df["hours"] = df["hours"].astype(int)
    
    sensors = ["S1", "S2", "S3", "S4"]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for sensor, color in zip(sensors, COLORS):
        col = f"{sensor}_{sensor_type}"
        if col in df.columns:
            grouped = df[[col, "hours"]].groupby("hours").mean()
            ax.plot(grouped.index, grouped[col], label=sensor, color=color, linewidth=2, marker="o", markersize=4)
    
    ax.set_title(f"{sensor_type} Variation by Hour", fontweight="bold", fontsize=16)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(sensor_type)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{sensor_type.lower()}_by_hour.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_sensor_boxplots(df, sensor_type, output_dir="outputs"):
    """Plot boxplots for sensor readings."""
    ensure_dir(output_dir)
    
    cols = [f"S{i}_{sensor_type}" for i in range(1, 5)]
    cols = [c for c in cols if c in df.columns]
    
    if not cols:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[cols], palette=COLORS[:len(cols)], ax=ax)
    ax.set_title(f"{sensor_type} Sensor Boxplots", fontweight="bold", fontsize=16)
    ax.set_ylabel("Value")
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{sensor_type.lower()}_boxplots.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_co2_by_occupancy(df, output_dir="outputs"):
    """Plot CO2 levels by occupancy and hour."""
    ensure_dir(output_dir)
    
    if "hours" not in df.columns:
        df = df.copy()
        df[["hours", "_m", "_s"]] = df["Time"].str.split(":", expand=True)
        df["hours"] = df["hours"].astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for occ, color in zip(range(4), COLORS):
        occ_df = df[df["Room_Occupancy_Count"] == occ][["S5_CO2", "hours"]].groupby("hours").mean()
        ax.plot(occ_df.index, occ_df["S5_CO2"], label=f"{occ} Occupants",
                color=color, linewidth=2, marker="o", markersize=4)
    
    ax.set_title("CO2 Levels by Occupancy & Hour", fontweight="bold", fontsize=16)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "co2_by_occupancy.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_correlation_heatmap(df, output_dir="outputs"):
    """Plot correlation heatmap."""
    ensure_dir(output_dir)
    
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                center=0, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=16)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_model_comparison(results_df, output_dir="outputs"):
    """Comprehensive model comparison dashboard."""
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Comparison Dashboard", fontweight="bold", fontsize=18, y=1.02)
    
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        values = results_df.loc[metric]
        models = values.index.tolist()
        scores = values.values.astype(float)
        
        bars = ax.bar(models, scores, color=COLORS[:len(models)], edgecolor="white", linewidth=2)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_confusion_matrices(detailed_results, output_dir="outputs"):
    """Plot confusion matrices for all models."""
    ensure_dir(output_dir)
    
    n = len(detailed_results)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=18, y=1.02)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(detailed_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        cm = result["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    annot_kws={"size": 14, "fontweight": "bold"}, linewidths=2)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def plot_performance_heatmap(results_df, output_dir="outputs"):
    """Performance heatmap of all models."""
    ensure_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(results_df.astype(float), annot=True, fmt=".4f", cmap="YlGnBu", ax=ax,
                linewidths=2, annot_kws={"size": 13, "fontweight": "bold"}, vmin=0.5, vmax=1.0)
    ax.set_title("Model Performance Heatmap", fontweight="bold", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "performance_heatmap.png")
    plt.savefig(path); plt.close()
    print(f"   📈 Saved: {path}")


def run_eda_visualizations(df, output_dir="outputs"):
    """Run all EDA visualizations."""
    print("\n" + "=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    plot_occupancy_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    for sensor_type in ["Temp", "Light", "Sound"]:
        plot_sensor_variation_by_hour(df, sensor_type, output_dir)
        plot_sensor_boxplots(df, sensor_type, output_dir)
    
    plot_co2_by_occupancy(df, output_dir)
    
    print(f"\n✅ EDA visualizations saved to '{output_dir}/'")


def run_model_visualizations(results_df, detailed_results, output_dir="outputs"):
    """Run all model comparison visualizations."""
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON VISUALIZATIONS")
    print("=" * 60)
    
    plot_model_comparison(results_df, output_dir)
    plot_confusion_matrices(detailed_results, output_dir)
    plot_performance_heatmap(results_df, output_dir)
    
    print(f"\n✅ Model visualizations saved to '{output_dir}/'")
