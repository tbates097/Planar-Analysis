import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_metrics(input_path: str) -> pd.DataFrame:
    raw_text = Path(input_path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], dict):
        raise ValueError("Metrics.json must be a non-empty list with one object of metrics.")

    metrics: Dict[str, str] = data[0]
    rows: List[Dict[str, float]] = []

    def add_row(measure: str, stat: str, xy: int, value_str: str) -> None:
        if value_str is None:
            return
        try:
            value = float(value_str)
        except (TypeError, ValueError):
            return
        rows.append({"measure": measure, "stat": stat, "xy": int(xy), "value": value})

    for xy in (100, 200, 300):
        add_row("Repeat", "Average", xy, metrics.get(f"Average_Repeat_{xy}XY"))
        add_row("Repeat", "Excluded", xy, metrics.get(f"Excluded_Repeat_{xy}XY"))
        add_row("Repeat", "Included", xy, metrics.get(f"Included_Repeat_{xy}XY"))
        add_row("Repeat", "Pass_Percentage", xy, metrics.get(f"PlanarDL{xy}_Repeat_Pass_Percentage"))
        add_row("Repeat", "AvgTestsPerSerial", xy, metrics.get(f"AvgRepeatTestsPerSerial_{xy}XY"))
        add_row("Flatness", "Average", xy, metrics.get(f"Average_Flatness_{xy}XY"))
        add_row("Flatness", "Excluded", xy, metrics.get(f"Excluded_Flatness_{xy}XY"))
        add_row("Flatness", "Included", xy, metrics.get(f"Included_Flatness_{xy}XY"))
        add_row("Flatness", "Pass_Percentage", xy, metrics.get(f"PlanarDL{xy}_Flatness_Pass_Percentage"))
        add_row("Flatness", "AvgTestsPerSerial", xy, metrics.get(f"AvgFlatnessTestsPerSerial_{xy}XY"))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No metrics parsed from Metrics.json. Check key names.")
    df["measure"] = pd.Categorical(df["measure"], categories=["Repeat", "Flatness"], ordered=True)
    df = df.sort_values(["measure", "xy", "stat"]).reset_index(drop=True)
    return df


def add_incl_excl_annotations(fig: go.Figure, wide_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for _, row in wide_df.iterrows():
        # Ensure categorical labels are passed as strings and anchor to axes coords
        x_val = str(row[x_col])
        y_val = row.get(y_col)
        incl = row.get("Included")
        excl = row.get("Excluded")
        if pd.isna(y_val) or pd.isna(incl) or pd.isna(excl):
            continue
        fig.add_annotation(x=x_val, y=y_val, text=f"Incl: {int(incl)}, Excl: {int(excl)}", showarrow=False, yshift=12, font=dict(size=10, color="#444"), xref="x", yref="y")


def generate_charts(df: pd.DataFrame, outdir: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    html_sections: List[str] = []

    def save_png_if_possible(fig: go.Figure, file_path: Path) -> None:
        try:
            # Prefer write_image if kaleido is installed
            fig.write_image(str(file_path))
        except Exception:
            # Skip PNG export silently if engine not available
            pass

    label_order = ["PlanarDL-100XY", "PlanarDL-200XY", "PlanarDL-300XY"]
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure]
        wide = sub.pivot(index="xy", columns="stat", values="value").reset_index()
        wide["label"] = wide["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        wide["label"] = pd.Categorical(wide["label"], categories=label_order, ordered=True)

        fig_avg = px.bar(wide.sort_values("label"), x="label", y="Average", title=f"{measure} Average by Part Number", labels={"label": "Part Number", "Average": "Average"})
        add_incl_excl_annotations(fig_avg, wide, x_col="label", y_col="Average")
        save_png_if_possible(fig_avg, out_path / f"{measure.lower()}_avg.png")
        html_sections.append(fig_avg.to_html(full_html=False, include_plotlyjs=False))
        # Save standalone HTML for this chart
        (out_path / f"{measure.lower()}_avg.html").write_text(
            fig_avg.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8"
        )

        fig_pass = px.bar(wide.sort_values("label"), x="label", y="Pass_Percentage", title=f"{measure} Pass % by Part Number", labels={"label": "Part Number", "Pass_Percentage": "Pass %"})
        add_incl_excl_annotations(fig_pass, wide, x_col="label", y_col="Pass_Percentage")
        save_png_if_possible(fig_pass, out_path / f"{measure.lower()}_pass.png")
        html_sections.append(fig_pass.to_html(full_html=False, include_plotlyjs=False))
        # Save standalone HTML for this chart
        (out_path / f"{measure.lower()}_pass.html").write_text(
            fig_pass.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8"
        )

        fig_tests = px.bar(wide.sort_values("label"), x="label", y="AvgTestsPerSerial", title=f"{measure} Avg Tests/Serial", labels={"label": "Part Number", "AvgTestsPerSerial": "Avg Tests/Serial"})
        save_png_if_possible(fig_tests, out_path / f"{measure.lower()}_avg_tests.png")
        html_sections.append(fig_tests.to_html(full_html=False, include_plotlyjs=False))
        # Save standalone HTML for this chart
        (out_path / f"{measure.lower()}_avg_tests.html").write_text(
            fig_tests.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8"
        )

    comb = df[df["stat"].isin(["Average", "Pass_Percentage"])].pivot(index=["measure", "xy"], columns="stat", values="value").reset_index()
    # Styled scatter remains numeric but notated in title
    fig_scatter = px.scatter(comb, x="Average", y="Pass_Percentage", color="measure", symbol="xy", title="Average vs Pass % by Measure and Part Number", labels={"Average": "Average", "Pass_Percentage": "Pass %"})
    save_png_if_possible(fig_scatter, out_path / "avg_vs_pass.png")
    html_sections.append(fig_scatter.to_html(full_html=False, include_plotlyjs=False))
    # Save standalone HTML for this chart
    (out_path / "avg_vs_pass.html").write_text(
        fig_scatter.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8"
    )

    dashboard_html = out_path / "dashboard.html"
    dashboard_html.write_text(
        """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Planar Analysis Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
    .card { border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    h2 { margin: 0 0 8px 0; font-size: 18px; }
  </style>
  </head>
<body>
  <h1>Planar Analysis Dashboard</h1>
  <div class=\"grid\">
        """
        + "\n".join(f"<div class='card'>{section}</div>" for section in html_sections)
        + """
  </div>
</body>
</html>
        """,
        encoding="utf-8",
    )

    return dashboard_html


def write_fig_html(fig: go.Figure, path: Path) -> None:
    path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def generate_group_a(df: pd.DataFrame, outdir: str) -> None:
    out_path = Path(outdir) / "GroupA"
    out_path.mkdir(parents=True, exist_ok=True)

    # Lollipop charts for averages (per measure)
    label_order = ["PlanarDL-100XY", "PlanarDL-200XY", "PlanarDL-300XY"]
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        sub = sub.sort_values("label")
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        sub["label"] = pd.Categorical(sub["label"], categories=label_order, ordered=True)
        fig = go.Figure()
        # Stems
        for _, r in sub.sort_values("label").iterrows():
            fig.add_shape(type="line", x0=str(r["label"]), y0=0, x1=str(r["label"]), y1=r["Average"], line=dict(color="#8cb3d9", width=3), xref="x", yref="y")
        # Dots
        fig.add_trace(go.Scatter(
            x=sub.sort_values("label")["label"].astype(str), y=sub.sort_values("label")["Average"], mode="markers+text", marker=dict(size=12, color="#1f77b4"),
            text=[f"{v:.2f}" for v in sub["Average"]], textposition="top center",
            hovertemplate="XY %{x}<br>Average: %{y:.2f}<br>Incl %{customdata[0]} / Excl %{customdata[1]}<extra></extra>",
            customdata=sub[["Included", "Excluded"]].values,
            name=f"{measure} Average"
        ))
        # Included/Excluded annotations
        add_incl_excl_annotations(fig, sub.rename(columns={"xy": "label"}), x_col="label", y_col="Average")
        fig.update_layout(title=f"{measure} Average (Lollipop)", xaxis_title="Part Number", yaxis_title="Average", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_avg_lollipop.html")

    # Bullet charts for pass percentage (per measure)
    bands = [(0, 60, "#fee5d9"), (60, 75, "#fcbba1"), (75, 90, "#fc9272"), (90, 100, "#fb6a4a")]
    target = 80
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        categories = [str(x) for x in sub["label"].tolist()][::-1]
        fig = go.Figure()
        # Background bands as stacked bars for each category
        for (start, end, color) in bands:
            fig.add_trace(go.Bar(
                x=[end - start] * len(categories), y=categories, orientation="h",
                base=[start] * len(categories), marker_color=color, showlegend=False, hoverinfo="skip"
            ))
        # Actual markers
        fig.add_trace(go.Scatter(
            x=sub["Pass_Percentage"], y=[str(x) for x in sub["label"]], mode="markers+text",
            marker=dict(size=12, color="#2ca02c"), text=[f"{v:.1f}%" for v in sub["Pass_Percentage"]], textposition="middle right",
            hovertemplate="XY %{y}<br>Pass: %{x:.1f}%<br>Incl %{customdata[0]} / Excl %{customdata[1]}<extra></extra>",
            customdata=sub[["Included", "Excluded"]].values, name=f"{measure} Pass %"
        ))
        # Target line
        fig.add_vline(x=target, line_dash="dash", line_color="#555", annotation_text=f"Target {target}%", annotation_position="top left")
        fig.update_layout(title=f"{measure} Pass % (Bullet)", xaxis_title="Pass %", yaxis_title="Part Number", barmode="overlay", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_pass_bullet.html")

    # Bullet charts for Avg Tests/Serial (per measure)
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        categories = [str(x) for x in sub["label"].tolist()][::-1]
        max_val = float(sub["AvgTestsPerSerial"].max()) if not sub["AvgTestsPerSerial"].empty else 0.0
        upper = max(1.0, math.ceil(max_val * 1.1))
        band_edges = [0, upper * 0.33, upper * 0.66, upper * 0.85, upper]
        band_colors = ["#e6f2ff", "#cce5ff", "#99ccff", "#66b2ff"]
        fig = go.Figure()
        for i in range(len(band_edges) - 1):
            start, end = band_edges[i], band_edges[i + 1]
            fig.add_trace(go.Bar(
                x=[end - start] * len(categories), y=categories, orientation="h",
                base=[start] * len(categories), marker_color=band_colors[i], showlegend=False, hoverinfo="skip"
            ))
        fig.add_trace(go.Scatter(
            x=sub["AvgTestsPerSerial"], y=[str(x) for x in sub["label"]], mode="markers+text",
            marker=dict(size=12, color="#1f77b4"), text=[f"{v:.2f}" for v in sub["AvgTestsPerSerial"]], textposition="middle right",
            hovertemplate="%{y}<br>Avg Tests/Serial: %{x:.2f}<br>Incl %{customdata[0]} / Excl %{customdata[1]}<extra></extra>",
            customdata=sub[["Included", "Excluded"]].values, name=f"{measure} Avg Tests/Serial"
        ))
        fig.update_layout(title=f"{measure} Avg Tests/Serial (Bullet)", xaxis_title="Avg Tests/Serial", yaxis_title="Part Number", barmode="overlay", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_tests_bullet.html")

    # KPI tiles (overall)
    wide = df.pivot_table(index=["measure", "xy"], columns="stat", values="value").reset_index()
    # Weighted by Included counts across both measures
    total_included = wide["Included"].sum()
    weighted_avg = (wide["Average"] * wide["Included"]).sum() / total_included if total_included else float("nan")
    weighted_pass = (wide["Pass_Percentage"] * wide["Included"]).sum() / total_included if total_included else float("nan")
    total_excluded = wide["Excluded"].sum()
    avg_tests = wide["AvgTestsPerSerial"].mean()
    kpi_html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>KPIs</title>
<style>
body{{font-family:Arial, sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
.card{{border:1px solid #e3e3e3;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.v{{font-size:28px;font-weight:600}} .t{{color:#666;margin-top:4px}}
</style></head><body>
<div class='grid'>
  <div class='card'><div class='v'>{weighted_avg:.2f}</div><div class='t'>Weighted Average</div></div>
  <div class='card'><div class='v'>{weighted_pass:.1f}%</div><div class='t'>Weighted Pass %</div></div>
  <div class='card'><div class='v'>{int(total_included)}</div><div class='t'>Total Included</div></div>
  <div class='card'><div class='v'>{int(total_excluded)}</div><div class='t'>Total Excluded</div></div>
  <div class='card'><div class='v'>{avg_tests:.2f}</div><div class='t'>Avg Tests per Serial</div></div>
</div>
</body></html>
"""
    (out_path / "kpis.html").write_text(kpi_html, encoding="utf-8")

    # Styled scatter: Average vs Pass %
    comb = df[df["stat"].isin(["Average", "Pass_Percentage"])].pivot(index=["measure", "xy"], columns="stat", values="value").reset_index()
    fig = px.scatter(comb, x="Average", y="Pass_Percentage", color="measure", symbol="xy", title="Average vs Pass %", template="plotly_white")
    fig.add_hrect(y0=80, y1=100, fillcolor="#e8f5e9", opacity=0.5, line_width=0)
    fig.update_yaxes(range=[0, 100])
    write_fig_html(fig, out_path / "avg_vs_pass_styled.html")


def generate_group_b(df: pd.DataFrame, outdir: str) -> None:
    out_path = Path(outdir) / "GroupB"
    out_path.mkdir(parents=True, exist_ok=True)

    # Radial (polar) pass % per measure
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=sub["Pass_Percentage"], theta=[str(x) for x in sub["label"]],
            marker_color=["#b2df8a", "#66c2a5", "#1b9e77"], opacity=0.85, name=f"{measure} Pass %"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title=f"{measure} Pass % (Radial)", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_pass_radial.html")

    # Dumbbell chart: Average Repeat vs Flatness per XY
    rep = df[df["measure"] == "Repeat"].pivot(index="xy", columns="stat", values="value").reset_index()[["xy", "Average", "Included", "Excluded"]]
    rep = rep.rename(columns={"Average": "Repeat_Average", "Included": "Repeat_Included", "Excluded": "Repeat_Excluded"})
    rep["label"] = rep["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
    flat = df[df["measure"] == "Flatness"].pivot(index="xy", columns="stat", values="value").reset_index()[["xy", "Average", "Included", "Excluded"]]
    flat = flat.rename(columns={"Average": "Flatness_Average", "Included": "Flatness_Included", "Excluded": "Flatness_Excluded"})
    flat["label"] = flat["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
    mrg = pd.merge(rep, flat, on="xy")
    fig = go.Figure()
    for _, r in mrg.iterrows():
        fig.add_trace(go.Scatter(x=[r["Repeat_Average"], r["Flatness_Average"]], y=[str(r["label_x"]), str(r["label_x"])], mode="lines+markers",
                                 line=dict(color="#888"), marker=dict(size=10),
                                 name=str(r["label_x"]),
                                 text=[f"R {r['Repeat_Average']:.2f}", f"F {r['Flatness_Average']:.2f}"], textposition="top center",
                                 hovertext=[f"Incl {int(r['Repeat_Included'])} / Excl {int(r['Repeat_Excluded'])}", f"Incl {int(r['Flatness_Included'])} / Excl {int(r['Flatness_Excluded'])}"],
                                 hoverinfo="text+x+y"))
    fig.update_layout(title="Average Dumbbell: Repeat vs Flatness by Part Number", xaxis_title="Average", yaxis_title="Part Number", template="plotly_white")
    write_fig_html(fig, out_path / "avg_dumbbell_repeat_vs_flatness.html")

    # Heatmaps: Average and Pass %, normalized per measure across XY (min-max)
    wide = df.pivot_table(index=["measure", "xy"], columns="stat", values="value").reset_index()
    # Average heatmap
    avg_pivot = wide.pivot(index="measure", columns="xy", values="Average")
    avg_norm = (avg_pivot - avg_pivot.min(axis=1).values.reshape(-1, 1)) / (avg_pivot.max(axis=1).values.reshape(-1, 1) - avg_pivot.min(axis=1).values.reshape(-1, 1) + 1e-9)
    fig_avg = go.Figure(data=go.Heatmap(z=avg_norm.values, x=avg_norm.columns.astype(str), y=avg_norm.index, colorscale="YlGnBu"))
    fig_avg.update_layout(title="Heatmap (Normalized): Average", xaxis_title="XY", yaxis_title="Measure", template="plotly_white")
    write_fig_html(fig_avg, out_path / "heatmap_avg.html")
    # Pass heatmap
    pass_pivot = wide.pivot(index="measure", columns="xy", values="Pass_Percentage")
    pass_norm = (pass_pivot - pass_pivot.min(axis=1).values.reshape(-1, 1)) / (pass_pivot.max(axis=1).values.reshape(-1, 1) - pass_pivot.min(axis=1).values.reshape(-1, 1) + 1e-9)
    fig_pass = go.Figure(data=go.Heatmap(z=pass_norm.values, x=pass_norm.columns.astype(str), y=pass_norm.index, colorscale="YlOrRd"))
    fig_pass.update_layout(title="Heatmap (Normalized): Pass %", xaxis_title="XY", yaxis_title="Measure", template="plotly_white")
    write_fig_html(fig_pass, out_path / "heatmap_pass.html")
    # KPI tiles (own copy, computed similarly to Group A)
    wide2 = df.pivot_table(index=["measure", "xy"], columns="stat", values="value").reset_index()
    total_included2 = wide2["Included"].sum()
    weighted_avg2 = (wide2["Average"] * wide2["Included"]).sum() / total_included2 if total_included2 else float("nan")
    weighted_pass2 = (wide2["Pass_Percentage"] * wide2["Included"]).sum() / total_included2 if total_included2 else float("nan")
    total_excluded2 = wide2["Excluded"].sum()
    avg_tests2 = wide2["AvgTestsPerSerial"].mean()
    kpi_b_html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>KPIs (Group B)</title>
<style>
body{{font-family:Arial, sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
.card{{border:1px solid #e3e3e3;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.v{{font-size:28px;font-weight:600}} .t{{color:#666;margin-top:4px}}
</style></head><body>
<div class='grid'>
  <div class='card'><div class='v'>{weighted_avg2:.2f}</div><div class='t'>Weighted Average</div></div>
  <div class='card'><div class='v'>{weighted_pass2:.1f}%</div><div class='t'>Weighted Pass %</div></div>
  <div class='card'><div class='v'>{int(total_included2)}</div><div class='t'>Total Included</div></div>
  <div class='card'><div class='v'>{int(total_excluded2)}</div><div class='t'>Total Excluded</div></div>
  <div class='card'><div class='v'>{avg_tests2:.2f}</div><div class='t'>Avg Tests per Serial</div></div>
</div>
</body></html>
"""
    (out_path / "kpis_b.html").write_text(kpi_b_html, encoding="utf-8")


def generate_required_charts(df: pd.DataFrame, outdir: str) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    label_order = ["PlanarDL-100XY", "PlanarDL-200XY", "PlanarDL-300XY"]

    # Lollipop averages for Repeat and Flatness
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        sub["label"] = pd.Categorical(sub["label"], categories=label_order, ordered=True)
        fig = go.Figure()
        for _, r in sub.sort_values("label").iterrows():
            fig.add_shape(type="line", x0=str(r["label"]), y0=0, x1=str(r["label"]), y1=r["Average"], line=dict(color="#8cb3d9", width=3))
        fig.add_trace(go.Scatter(
            x=sub.sort_values("label")["label"].astype(str), y=sub.sort_values("label")["Average"], mode="markers+text",
            marker=dict(size=12, color="#1f77b4"), text=[f"{v:.2f}" for v in sub["Average"]], textposition="top center",
            hovertemplate="%{x}<br>Average: %{y:.2f}<br>Incl %{customdata[0]} / Excl %{customdata[1]}<extra></extra>",
            customdata=sub[["Included", "Excluded"]].values, name=f"{measure} Average"
        ))
        # Removed on-chart annotations per request
        fig.update_layout(title=f"{measure} Average (Lollipop)", xaxis_title="Part Number", yaxis_title="Average", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_avg_lollipop.html")

    # Radial pass percentage for Repeat and Flatness
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=sub["Pass_Percentage"], theta=[str(x) for x in sub["label"]],
            marker_color=["#b2df8a", "#66c2a5", "#1b9e77"], opacity=0.85, name=f"{measure} Pass %"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title=f"{measure} Pass % (Radial)", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_pass_radial.html")

    # Bullet charts for Avg Tests/Serial for Repeat and Flatness
    for measure in ["Repeat", "Flatness"]:
        sub = df[df["measure"] == measure].pivot(index="xy", columns="stat", values="value").reset_index()
        sub["label"] = sub["xy"].map(lambda v: f"PlanarDL-{int(v)}XY")
        categories = [str(x) for x in sub["label"].tolist()][::-1]
        max_val = float(sub["AvgTestsPerSerial"].max()) if not sub["AvgTestsPerSerial"].empty else 0.0
        upper = max(1.0, math.ceil(max_val * 1.1))
        band_edges = [0, upper * 0.33, upper * 0.66, upper * 0.85, upper]
        band_colors = ["#e6f2ff", "#cce5ff", "#99ccff", "#66b2ff"]
        fig = go.Figure()
        for i in range(len(band_edges) - 1):
            start, end = band_edges[i], band_edges[i + 1]
            fig.add_trace(go.Bar(
                x=[end - start] * len(categories), y=categories, orientation="h",
                base=[start] * len(categories), marker_color=band_colors[i], showlegend=False, hoverinfo="skip"
            ))
        fig.add_trace(go.Scatter(
            x=sub["AvgTestsPerSerial"], y=[str(x) for x in sub["label"]], mode="markers+text",
            marker=dict(size=12, color="#1f77b4"), text=[f"{v:.2f}" for v in sub["AvgTestsPerSerial"]], textposition="middle right",
            hovertemplate="%{y}<br>Avg Tests/Serial: %{x:.2f}<br>Incl %{customdata[0]} / Excl %{customdata[1]}<extra></extra>",
            customdata=sub[["Included", "Excluded"]].values, name=f"{measure} Avg Tests/Serial"
        ))
        fig.update_layout(title=f"{measure} Avg Tests/Serial (Bullet)", xaxis_title="Avg Tests/Serial", yaxis_title="Part Number", barmode="overlay", template="plotly_white")
        write_fig_html(fig, out_path / f"{measure.lower()}_tests_bullet.html")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Planar Metrics and generate charts")
    parser.add_argument("--input", default=str(Path(__file__).with_name("Metrics.json")), help="Path to Metrics.json")
    parser.add_argument("--outdir", default=str(Path(__file__).with_name("reports")), help="Output directory for charts and dashboard")
    parser.add_argument("--open", action="store_true", help="Open dashboard in default browser after generation")
    args = parser.parse_args()

    df = load_metrics(args.input)
    # Generate only the requested charts
    generate_required_charts(df, args.outdir)

    if args.open:
        import webbrowser
        webbrowser.open_new_tab((Path(args.outdir) / "repeat_pass_radial.html").resolve().as_uri())


if __name__ == "__main__":
    main()
