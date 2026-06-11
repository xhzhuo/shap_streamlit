# -*- coding: utf-8 -*-
"""
RED SI Pipeline 图表绘制
=========================

提供静态 PNG 图（matplotlib）和交互式 HTML 图（plotly，可选）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import OUTPUT_PNG, OUTPUT_HTML


def plot_static_chart(model_df: pd.DataFrame, output_png: str = OUTPUT_PNG) -> None:
    """生成静态预测图 PNG。"""
    train_mask = model_df["brand_search_index_mil"].notna()
    forecast_mask = ~train_mask
    train = model_df.loc[train_mask]
    forecast = model_df.loc[forecast_mask]

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    ax.plot(
        model_df["week_start"],
        model_df["brand_search_index_mil"],
        color="#1f4e79",
        linewidth=2.2,
        label="Brand SI actual",
    )
    ax.plot(
        train["week_start"],
        train["fit_base"],
        color="#9e9e9e",
        linestyle="--",
        linewidth=1.5,
        label="Base fit",
    )

    if len(forecast) > 0:
        ax.plot(
            forecast["week_start"],
            forecast["forecast_base"],
            color="#f28e2b",
            linewidth=2.5,
            label="Base forecast",
        )
        ax.plot(
            forecast["week_start"],
            forecast["optimistic"],
            color="#59a14f",
            linestyle=":",
            linewidth=2.5,
            label="Optimistic forecast",
        )
        ax.plot(
            forecast["week_start"],
            forecast["pessimistic"],
            color="#e15759",
            linestyle=":",
            linewidth=2.5,
            label="Pessimistic forecast",
        )
        ax.axvline(forecast["week_start"].iloc[0], color="#666666", linestyle="--", linewidth=1)
        ax.text(forecast["week_start"].iloc[0], ax.get_ylim()[1] * 0.98, " Forecast start", va="top", fontsize=9)

    ax.set_ylabel("Brand Search Index (Mil.)")
    ax.set_xlabel("Week")
    ax.set_title(
        "RED Brand Search Index Attribution Forecast Demo\n"
        "Geometric Adstock + Positive Ridge; Scenario = Base ± 0.9σ residual"
    )
    ax.grid(True, alpha=0.25)

    # 右轴：行业 SI
    ax2 = ax.twinx()
    ax2.plot(
        model_df["week_start"],
        model_df["industry_total_search_index_mil"],
        color="#bdbdbd",
        linewidth=1.8,
        alpha=0.8,
        label="Industry SI",
    )
    ax2.set_ylabel("Industry Search Index (Mil.)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close()


def plot_interactive_chart(model_df: pd.DataFrame, output_html: str = OUTPUT_HTML) -> None:
    """生成交互式 HTML 图。若未安装 plotly，则跳过。"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly is not installed. Skip interactive HTML chart.")
        return

    train_mask = model_df["brand_search_index_mil"].notna()
    forecast_mask = ~train_mask
    train = model_df.loc[train_mask]
    forecast = model_df.loc[forecast_mask]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=model_df["week_start"],
            y=model_df["brand_search_index_mil"],
            mode="lines+markers",
            name="Brand SI actual",
            line=dict(color="#1f4e79", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=train["week_start"],
            y=train["fit_base"],
            mode="lines",
            name="Base fit",
            line=dict(color="#9e9e9e", width=2, dash="dash"),
        ),
        secondary_y=False,
    )

    if len(forecast) > 0:
        fig.add_trace(
            go.Scatter(
                x=forecast["week_start"],
                y=forecast["forecast_base"],
                mode="lines+markers",
                name="Base forecast",
                line=dict(color="#f28e2b", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["week_start"],
                y=forecast["optimistic"],
                mode="lines",
                name="Optimistic forecast",
                line=dict(color="#59a14f", width=3, dash="dot"),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["week_start"],
                y=forecast["pessimistic"],
                mode="lines",
                name="Pessimistic forecast",
                line=dict(color="#e15759", width=3, dash="dot"),
            ),
            secondary_y=False,
        )
        fig.add_vline(x=forecast["week_start"].iloc[0], line_dash="dash", line_color="#666666")

    fig.add_trace(
        go.Scatter(
            x=model_df["week_start"],
            y=model_df["industry_total_search_index_mil"],
            mode="lines",
            name="Industry SI",
            line=dict(color="#bdbdbd", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="RED Brand Search Index Attribution Forecast Demo<br>"
              "<sup>Geometric Adstock + Positive Ridge; Scenario = Base ± 0.9σ residual</sup>",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=650,
    )
    fig.update_yaxes(title_text="Brand Search Index (Mil.)", secondary_y=False)
    fig.update_yaxes(title_text="Industry Search Index (Mil.)", secondary_y=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
