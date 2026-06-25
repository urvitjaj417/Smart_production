# Azad Engineering — Smart Production System

A full-stack factory intelligence dashboard for **Azad Engineering** built with a Python/Flask backend and a self-contained HTML/JS frontend.

---

## 📁 Folder Structure

```
Smart_production/
├── app.py                   # Flask backend — data loading, AI risk scoring, REST APIs
├── data/
│   └── factory_logs.csv     # Shop-floor dataset (upload your real data here)
├── templates/
│   └── index.html           # Frontend dashboard (Chart.js, PapaParse, SheetJS)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPIs, Machine Health Grid, Fault Rate, Output, Timeline Forecast |
| **Failures** | Fault distribution, Risk histogram, AI Risk Score per machine |
| **Sensors** | Sensor trends, Fault scatter, Normal vs Fault distribution |
| **Shifts** | Day / Evening / Night comparison — fault rate, output |
| **Recommendations** | AI-powered maintenance & workflow recommendations |
| **OEE** | Overall Equipment Effectiveness — Availability × Performance × Quality |
| **Correlation** | Sensor correlation heatmap, Pairwise scatter plots |
| **Alerts** | Live alert feed — Critical / Warning / Info |
| **Shot Peening** | Work Order Tracker, Almen Intensity, Coverage & Quality, Spec Checker |
| **Live Predict** | Real-time AI fault risk from manual sensor inputs |
| **Raw Data** | Full data explorer — search, filter, sort, export CSV/Excel |
| **AI Assistant** | In-dashboard AI chat (shortcut: Ctrl+K) |

---

## 🏭 Metrics Tracked

- **Temperature** (°C) — per machine, per shift
- **Pressure** (bar) — per machine, per shift
- **Operating Time** (hrs) — daily and rolling
- **Output Rate** (units per hour / uph)
- **Fault Occurred** — binary label (0 = Normal, 1 = Fault)
- **Fault Type** — Overheating, Pressure_Fault, Vibration, None
- **AI Risk Score** (0–100%) — heuristic model; replace with trained ML model
- **OEE Score** — Availability × Performance × Quality

---

## 📥 CSV Data Format

Your `data/factory_logs.csv` must have these columns:

| Column | Type | Example |
|---|---|---|
| timestamp | datetime | 2024-01-01 06:05:00 |
| machine_id | string | Machine_A |
| temperature | float | 72.3 |
| pressure | float | 4.8 |
| operating_time | float | 6.2 |
| output_rate | float | 112 |
| fault_occurred | int (0/1) | 0 |
| fault_type | string | None |
| risk_score | float | 18.2 |
| shift | string | Day / Eve / Night |

---

## 🔧 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/overview` | KPIs + machine health |
| GET | `/api/failures` | Fault analysis + high-risk records |
| GET | `/api/oee` | OEE per machine |
| GET | `/api/shifts` | Shift-wise KPIs |
| GET | `/api/recommendations` | AI recommendations |
| GET | `/api/sensors?machine=&sensor=` | Sensor time-series |
| POST | `/api/predict` | Live AI risk prediction |
| GET | `/api/rawdata?page=&machine=&fault=` | Paginated raw data |
| GET | `/api/export/csv` | Download full CSV |

---

## ⚙️ Upgrading the AI Risk Model

The default `compute_risk()` in `app.py` is a rule-based heuristic.  
To replace it with a trained ML model:

```python
import joblib
model = joblib.load("models/risk_model.pkl")

def compute_risk(temp, pressure, op_time, output_rate):
    X = [[temp, pressure, op_time, output_rate]]
    return float(model.predict_proba(X)[0][1] * 100)
```

---

## 🌐 Deploy to Production

### Option 1 — Gunicorn (Linux/macOS)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 2 — Railway / Render / Heroku
Push this folder to a Git repo and connect to your platform.  
Set the start command to:
```
gunicorn app:app
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `1–9, 0` | Jump to dashboard page 1–10 |
| `Ctrl/Cmd + K` | Open AI Assistant |
| `T` | Toggle Dark / Light theme |
| `?` | Show keyboard shortcuts |
| `Esc` | Close / unfocus |

---

## 👤 Credits

Dashboard designed, Backend and System Design by **Urvit Jajoo**  

