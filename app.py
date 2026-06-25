"""
Azad Engineering — Smart Production System
app.py — Main backend: data loading, AI risk scoring, KPI computation
"""

import os
import json
import math
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
import io

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "factory_logs.csv")

MACHINES = ["Machine_A", "Machine_B", "Machine_C", "Machine_D", "Machine_E"]
FAULT_TYPES = ["Overheating", "Pressure_Fault", "Vibration", "None"]
SHIFTS = {
    "Day":   (6,  14),
    "Eve":   (14, 22),
    "Night": (22, 24),
}


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV, parse timestamps, derive features."""
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Ensure all expected columns exist
    required = ["timestamp", "machine_id", "temperature", "pressure",
                "operating_time", "output_rate", "fault_occurred",
                "fault_type", "risk_score", "shift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["fault_type"] = df["fault_type"].fillna("None")
    return df


# ─── AI Risk Score Engine ──────────────────────────────────────────────────────
def compute_risk(temp: float, pressure: float,
                 op_time: float, output_rate: float) -> float:
    """
    Heuristic AI risk model (0–100).
    Replace with a trained scikit-learn model for production.

    Scoring bands:
      Temperature : 80–100 °C → low; 100–130 °C → medium; >130 °C → high
      Pressure    : 6–8 bar   → low; 8–12 bar   → medium; >12 bar  → high
      Op Time     : 8–10 hrs  → low; 10–16 hrs  → medium; >16 hrs  → high
      Output Rate : <60 uph   → penalty
    """
    risk = 0.0

    # ── Temperature (max 50 pts) ──────────────────────────────────────────────
    if temp > 130:
        risk += 50
    elif temp > 100:
        risk += 20 + (temp - 100) / 30 * 30   # 20–50 pts
    elif temp > 80:
        risk += (temp - 80) / 20 * 20          # 0–20 pts

    # ── Pressure (max 35 pts) ─────────────────────────────────────────────────
    if pressure > 12:
        risk += 35
    elif pressure > 8:
        risk += 10 + (pressure - 8) / 4 * 25  # 10–35 pts
    elif pressure > 6:
        risk += (pressure - 6) / 2 * 10        # 0–10 pts

    # ── Operating Time (max 10 pts) ───────────────────────────────────────────
    if op_time > 16:
        risk += 10
    elif op_time > 10:
        risk += (op_time - 10) / 6 * 10        # 0–10 pts
    elif op_time > 8:
        risk += (op_time - 8) / 2 * 4          # 0–4 pts

    # ── Low Output Penalty (max 5 pts) ────────────────────────────────────────
    if output_rate < 60:
        risk += (60 - output_rate) / 60 * 5

    return round(min(risk, 100.0), 2)


def classify_fault(risk: float, temp: float, pressure: float) -> str:
    if risk < 40:
        return "None"
    if temp > 110:
        return "Overheating"
    if pressure > 9:
        return "Pressure_Fault"
    return "Vibration"


# ─── KPI Computation ──────────────────────────────────────────────────────────
def overview_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    faults = int(df["fault_occurred"].sum())
    fault_rate = round(faults / total * 100, 2) if total else 0
    avg_output = round(df["output_rate"].mean(), 1)
    avg_risk = round(df["risk_score"].mean(), 1)
    high_risk = int((df["risk_score"] > 70).sum())

    return {
        "total_records": int(total),
        "total_faults": int(faults),
        "fault_rate_pct": float(fault_rate),
        "avg_output_uph": float(avg_output),
        "avg_risk_score": float(avg_risk),
        "high_risk_count": int(high_risk),
    }


def machine_health(df: pd.DataFrame) -> list:
    result = []
    for m in MACHINES:
        mdf = df[df["machine_id"] == m]
        if mdf.empty:
            continue
        risk_avg = round(mdf["risk_score"].mean(), 1)
        fault_rate = round(mdf["fault_occurred"].mean() * 100, 1)
        status = "good" if risk_avg < 40 else ("warn" if risk_avg < 70 else "crit")
        result.append({
            "machine": m,
            "avg_risk": float(risk_avg),
            "fault_rate_pct": float(fault_rate),
            "avg_temp": float(round(mdf["temperature"].mean(), 1)),
            "avg_pressure": float(round(mdf["pressure"].mean(), 1)),
            "avg_output": float(round(mdf["output_rate"].mean(), 1)),
            "status": status,
        })
    return result


def oee_per_machine(df: pd.DataFrame) -> list:
    """
    Simplified OEE:
      Availability = 1 - fault_rate
      Performance  = avg_output / max_output_for_machine
      Quality      = 1 - (high_risk_rows / total)
      OEE          = A × P × Q
    """
    result = []
    for m in MACHINES:
        mdf = df[df["machine_id"] == m]
        if mdf.empty:
            continue
        availability = round((1 - mdf["fault_occurred"].mean()) * 100, 1)
        max_out = mdf["output_rate"].max() or 1
        performance = round(mdf["output_rate"].mean() / max_out * 100, 1)
        quality = round((1 - (mdf["risk_score"] > 70).mean()) * 100, 1)
        oee = round(availability * performance * quality / 10000, 1)
        result.append({
            "machine": m,
            "availability": float(availability),
            "performance": float(performance),
            "quality": float(quality),
            "oee": float(oee),
            "world_class": bool(oee >= 85),
        })
    return result


def shift_kpis(df: pd.DataFrame) -> dict:
    out = {}
    for shift in ["Day", "Eve", "Night"]:
        sdf = df[df["shift"] == shift]
        out[shift] = {
            "count": int(len(sdf)),
            "fault_rate": float(round(sdf["fault_occurred"].mean() * 100, 1)) if len(sdf) else 0.0,
            "avg_output": float(round(sdf["output_rate"].mean(), 1)) if len(sdf) else 0.0,
            "avg_risk": float(round(sdf["risk_score"].mean(), 1)) if len(sdf) else 0.0,
        }
    return out


def generate_recommendations(df: pd.DataFrame) -> list:
    recs = []
    mh = machine_health(df)
    for m in mh:
        if m["status"] == "crit":
            recs.append({
                "level": "crit",
                "machine": m["machine"],
                "title": f"🔴 CRITICAL — {m['machine']} Immediate Shutdown",
                "detail": f"Avg risk score {m['avg_risk']}%. Temperature and pressure thresholds exceeded repeatedly.",
                "action": "Schedule immediate maintenance. Check cooling system and pressure relief valves.",
            })
        elif m["status"] == "warn":
            recs.append({
                "level": "warn",
                "machine": m["machine"],
                "title": f"🟡 WARNING — {m['machine']} Elevated Risk",
                "detail": f"Risk score {m['avg_risk']}%. Fault rate {m['fault_rate_pct']}% above safe threshold.",
                "action": "Inspect during next scheduled window. Monitor temperature and pressure closely.",
            })

    # Shift recommendation
    sk = shift_kpis(df)
    worst_shift = max(sk, key=lambda s: sk[s]["fault_rate"])
    recs.append({
        "level": "warn",
        "machine": "All",
        "title": f"🟡 SHIFT INSIGHT — {worst_shift} shift has highest fault rate",
        "detail": f"Fault rate {sk[worst_shift]['fault_rate']}% vs day avg. Output {sk[worst_shift]['avg_output']} uph.",
        "action": "Review staffing and machine workload allocation for this shift.",
    })
    return recs


# ─── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    df = load_data()
    return jsonify({
        "kpis": overview_kpis(df),
        "machine_health": machine_health(df),
    })


@app.route("/api/failures")
def api_failures():
    df = load_data()
    faults = df[df["fault_occurred"] == 1].copy()
    fault_dist = df["fault_type"].value_counts().to_dict()
    high_risk = df[df["risk_score"] > 70][
        ["timestamp", "machine_id", "temperature", "pressure",
         "output_rate", "risk_score", "fault_type", "shift"]
    ].sort_values("risk_score", ascending=False).head(50)
    return jsonify({
        "fault_distribution": fault_dist,
        "high_risk_records": high_risk.to_dict(orient="records"),
        "kpis": overview_kpis(df),
    })


@app.route("/api/oee")
def api_oee():
    df = load_data()
    return jsonify({"oee": oee_per_machine(df)})


@app.route("/api/shifts")
def api_shifts():
    df = load_data()
    return jsonify({"shifts": shift_kpis(df)})


@app.route("/api/recommendations")
def api_recommendations():
    df = load_data()
    return jsonify({"recommendations": generate_recommendations(df)})


@app.route("/api/sensors")
def api_sensors():
    machine = request.args.get("machine", "All")
    sensor = request.args.get("sensor", "temperature")
    df = load_data()
    if machine != "All":
        df = df[df["machine_id"] == machine]
    cols = ["timestamp", "machine_id", sensor, "fault_occurred"]
    result = df[cols].sort_values("timestamp").to_dict(orient="records")
    return jsonify({"data": result})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    temp = float(body.get("temperature", 75))
    pressure = float(body.get("pressure", 5))
    op_time = float(body.get("operating_time", 6))
    output_rate = float(body.get("output_rate", 100))
    machine = body.get("machine", "Machine_A")

    risk = compute_risk(temp, pressure, op_time, output_rate)
    fault = classify_fault(risk, temp, pressure)
    status = "Critical" if risk > 70 else ("Warning" if risk > 40 else "Normal")

    flags = []
    if temp > 100:
        flags.append(f"⚠ High Temperature: {temp}°C")
    if pressure > 8:
        flags.append(f"⚠ High Pressure: {pressure} bar")
    if op_time > 12:
        flags.append(f"⚠ Extended Operating Time: {op_time} hrs")
    if output_rate < 60:
        flags.append(f"⚠ Low Output Rate: {output_rate} uph")

    return jsonify({
        "machine": machine,
        "risk_score": risk,
        "fault_type": fault,
        "status": status,
        "flags": flags,
    })


@app.route("/api/rawdata")
def api_rawdata():
    df = load_data()
    page = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 50))
    machine = request.args.get("machine", "")
    fault = request.args.get("fault", "")
    search = request.args.get("search", "").lower()

    if machine:
        df = df[df["machine_id"] == machine]
    if fault == "1":
        df = df[df["fault_occurred"] == 1]
    elif fault == "0":
        df = df[df["fault_occurred"] == 0]
    if search:
        df = df[df.apply(lambda r: search in str(r).lower(), axis=1)]

    total = len(df)
    chunk = df.iloc[page * per_page:(page + 1) * per_page]
    return jsonify({
        "total": total,
        "page": page,
        "data": chunk.to_dict(orient="records"),
    })


@app.route("/api/export/csv")
def export_csv():
    df = load_data()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="factory_logs_export.csv",
    )


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Azad Engineering — Smart Production System")
    print("  Starting server on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
