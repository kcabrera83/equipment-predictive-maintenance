"""Visualizacion de datos de mantenimiento predictivo."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


EQ_COLORS = {
    "centrifugal_pump": "#3498db",
    "reciprocating_pump": "#e67e22",
    "compressor": "#2ecc71",
    "separator": "#9b59b6",
}

STATUS_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
}


class EquipVisualizer:
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def plot_sensor_trends(self, df, unit_id=None):
        if unit_id is None:
            unit_id = df["unit_id"].unique()[0]
        w = df[df["unit_id"] == unit_id].sort_values("day")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        sensors = [
            ("vibration_x", "Vibration X (mm/s)", "#3498db"),
            ("temperature", "Temperature (C)", "#e74c3c"),
            ("pressure", "Pressure (psi)", "#2ecc71"),
            ("motor_current", "Motor Current (A)", "#e67e22"),
            ("flow_rate", "Flow Rate (m3/h)", "#9b59b6"),
            ("power_consumption", "Power (kW)", "#1abc9c"),
        ]
        for ax, (col, label, color) in zip(axes.flat, sensors):
            ax.plot(w["day"], w[col], color=color, linewidth=1.5, alpha=0.8)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Day")
            ax.grid(alpha=0.3)
            status_changes = w[w["status"] != w["status"].shift()]
            for _, row in status_changes.iterrows():
                if row["status"] > 0:
                    ax.axvline(x=row["day"], color=STATUS_COLORS.get(row["status"], "gray"),
                               linestyle="--", alpha=0.5, linewidth=0.8)
        fig.suptitle(f"Sensor Trends - {unit_id}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "01_sensor_trends")

    def plot_failure_distribution(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        status_counts = df.groupby(["unit_id", "status_label"]).size().reset_index(name="count")
        status_order = ["normal", "warning", "preventive_maintenance", "corrective_maintenance", "failure_imminent"]
        for label in status_order:
            subset = status_counts[status_counts["status_label"] == label]
            if len(subset) > 0:
                axes[0].bar(label, subset["count"].sum(), color=STATUS_COLORS[status_order.index(label)], alpha=0.8)
        axes[0].set_title("Distribucion de Estados", fontweight="bold")
        axes[0].set_xlabel("Estado")
        axes[0].set_ylabel("Registros")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].grid(axis="y", alpha=0.3)

        eq_counts = df.groupby("equipment_type")["unit_id"].nunique()
        for eq_type, color in EQ_COLORS.items():
            if eq_type in eq_counts.index:
                axes[1].bar(eq_type, eq_counts[eq_type], color=color, alpha=0.8)
        axes[1].set_title("Equipos por Tipo", fontweight="bold")
        axes[1].set_xlabel("Tipo")
        axes[1].set_ylabel("Cantidad")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "02_failure_distribution")

    def plot_correlation(self, df):
        numeric = df[["vibration_x", "vibration_y", "vibration_z", "temperature", "pressure",
                        "flow_rate", "motor_current", "bearing_temp", "oil_level",
                        "power_consumption", "status", "rul_days"]].copy()
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Correlation Matrix - Equipment Sensors", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "03_correlation_matrix")

    def plot_rul_distribution(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for eq_type, color in EQ_COLORS.items():
            subset = df[df["equipment_type"] == eq_type]
            if len(subset) > 0:
                last_records = subset.groupby("unit_id").last()
                axes[0].hist(last_records["rul_days"], bins=25, alpha=0.6, label=eq_type, color=color)
        axes[0].set_title("RUL Distribution by Equipment Type", fontweight="bold")
        axes[0].set_xlabel("Remaining Useful Life (days)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        for status in range(5):
            subset = df[df["status"] == status]
            if len(subset) > 0:
                axes[1].hist(subset["rul_days"], bins=25, alpha=0.6,
                             label=f"Status {status}", color=STATUS_COLORS[status])
        axes[1].set_title("RUL Distribution by Status", fontweight="bold")
        axes[1].set_xlabel("Remaining Useful Life (days)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "04_rul_distribution")

    def plot_model_comparison(self, results):
        models = list(results.keys())
        metrics = list(results[models[0]].keys()) if models else []
        fig, axes = plt.subplots(1, min(len(metrics), 4), figsize=(5 * min(len(metrics), 4), 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for i, metric in enumerate(metrics[:4]):
            vals = [results[m][metric] for m in models]
            bars = axes[i].barh(models, vals, color=colors, edgecolor="#2c3e50")
            axes[i].set_title(metric, fontweight="bold")
            axes[i].grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                             f"{val:.4f}", va="center", fontsize=9)
        fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "05_model_comparison")

    def plot_predictions_vs_actual(self, y_true, y_pred, title="RUL Predictions"):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, c="#3498db", edgecolors="#2c3e50", linewidth=0.3)
        lims = [min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1]
        axes[0].plot(lims, lims, "--", color="#e74c3c", linewidth=2, label="Perfect prediction")
        axes[0].set_xlabel("Actual RUL (days)")
        axes[0].set_ylabel("Predicted RUL (days)")
        axes[0].set_title("Scatter", fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, c="#e67e22", edgecolors="#2c3e50", linewidth=0.3)
        axes[1].axhline(y=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Predicted RUL")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals", fontweight="bold")
        axes[1].grid(alpha=0.3)

        axes[2].hist(residuals, bins=40, color="#2ecc71", edgecolor="#2c3e50", alpha=0.7)
        axes[2].axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[2].set_xlabel("Residual")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Residual Distribution", fontweight="bold")
        axes[2].grid(alpha=0.3)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "06_rul_predictions")
