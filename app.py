import json
import os
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model_bundle.pkl"
DB_PATH = BASE_DIR / "metahealth.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-for-production")


# ---------- Model loading ----------
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "model_bundle.pkl not found. Run `python train_model.py` first."
        )
    return joblib.load(BUNDLE_PATH)


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
imputer_values = bundle["imputer_values"]
feature_order = bundle["feature_order"]
numeric_cols = bundle.get("numeric_cols", ["Age", "BMI", "Cholesterol", "Sleep"])


# ---------- Database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            age REAL,
            gender REAL,
            bmi REAL,
            cholesterol REAL,
            sleep REAL,
            activity REAL,
            breakfast REAL,
            risk_label TEXT NOT NULL,
            risk_flag INTEGER NOT NULL,
            risk_probability REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    db.commit()
    db.close()


init_db()


# ---------- Helpers ----------
def parse_float(form_key, default=None):
    raw = request.form.get(form_key, "").strip()
    if raw == "":
        return default
    return float(raw)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def get_recent_predictions(user_id, limit=10):
    db = get_db()
    rows = db.execute(
        """
        SELECT created_at, risk_label, risk_probability, bmi, cholesterol, sleep
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return rows


def build_dashboard_payload(user_id):
    db = get_db()
    rows = db.execute(
        """
        SELECT created_at, risk_flag, risk_probability
        FROM predictions
        WHERE user_id = ?
        ORDER BY datetime(created_at) ASC
        """,
        (user_id,),
    ).fetchall()

    trend_labels = [
        datetime.fromisoformat(r["created_at"]).strftime("%d %b %H:%M") for r in rows
    ]
    trend_values = [round(r["risk_probability"], 2) for r in rows]

    high_count = sum(1 for r in rows if r["risk_flag"] == 1)
    low_count = sum(1 for r in rows if r["risk_flag"] == 0)

    coefficients = model.coef_[0]
    pairs = list(zip(feature_order, coefficients))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    importance_labels = [p[0] for p in pairs]
    importance_values = [round(float(p[1]), 4) for p in pairs]

    return {
        "trend_labels": trend_labels,
        "trend_values": trend_values,
        "risk_counts": [low_count, high_count],
        "importance_labels": importance_labels,
        "importance_values": importance_values,
        "total_predictions": len(rows),
    }


def generate_health_suggestions(input_values, risk_flag):
    suggestions = []

    if input_values["BMI"] >= 30:
        suggestions.append("Target gradual weight loss with a calorie-controlled, high-fiber meal plan.")
    elif input_values["BMI"] >= 25:
        suggestions.append("Aim for at least 150 minutes of moderate exercise weekly to improve BMI.")

    if input_values["Cholesterol"] >= 240:
        suggestions.append("Reduce saturated fats and include oats, nuts, and legumes to lower cholesterol.")
    elif input_values["Cholesterol"] >= 200:
        suggestions.append("Consider a heart-healthy diet pattern and repeat lipid testing with your clinician.")

    if input_values["Sleep"] < 6:
        suggestions.append("Improve sleep hygiene and target 7 to 8 hours of sleep each night.")

    if input_values["Activity"] == 0:
        suggestions.append("Start with 30 minutes of brisk walking 5 days per week.")

    if input_values["Breakfast"] == 0:
        suggestions.append("Avoid skipping breakfast; include protein plus whole grains in the morning.")

    if input_values["Age"] >= 45:
        suggestions.append("Schedule periodic metabolic screening for glucose, blood pressure, and lipids.")

    if risk_flag == 1 and not suggestions:
        suggestions.append("Book a clinician follow-up for a structured metabolic risk reduction plan.")

    if not suggestions:
        suggestions.append("Maintain your current healthy routine and continue regular preventive checkups.")

    return suggestions


def score_prediction(input_values):
    row = pd.DataFrame([{name: input_values[name] for name in feature_order}])
    row = row.astype(float)

    for col in feature_order:
        if pd.isna(row.at[0, col]):
            row.at[0, col] = imputer_values[col]

    scaled = row.copy()
    scaled[numeric_cols] = scaler.transform(row[numeric_cols])

    risk_flag = int(model.predict(scaled)[0])
    risk_prob = float(model.predict_proba(scaled)[0, 1]) * 100

    if risk_flag == 1:
        risk_label = "High Metabolic Risk"
        risk_theme = "high"
        note = "Lifestyle and clinical follow-up are recommended."
    else:
        risk_label = "Low Metabolic Risk"
        risk_theme = "low"
        note = "Continue healthy routines and regular screenings."

    suggestions = generate_health_suggestions(input_values, risk_flag)

    return {
        "label": risk_label,
        "theme": risk_theme,
        "probability": round(risk_prob, 2),
        "note": note,
        "flag": risk_flag,
        "suggestions": suggestions,
    }


# ---------- Auth routes ----------
@app.route("/")
def root():
    if "user_id" in session:
        return redirect(url_for("predictor"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3 or len(password) < 6:
            flash("Username must be 3+ chars and password must be 6+ chars.", "error")
            return render_template("register.html")

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), datetime.utcnow().isoformat()),
            )
            db.commit()
            flash("Account created. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("predictor"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Prediction + dashboard ----------
@app.route("/predictor")
@login_required
def predictor():
    recent = get_recent_predictions(session["user_id"], limit=8)
    return render_template(
        "index.html", prediction=None, form_data={}, recent_predictions=recent
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form_data = {
        "age": request.form.get("age", ""),
        "gender": request.form.get("gender", ""),
        "bmi": request.form.get("bmi", ""),
        "cholesterol": request.form.get("cholesterol", ""),
        "sleep": request.form.get("sleep", ""),
        "activity": request.form.get("activity", ""),
        "breakfast": request.form.get("breakfast", ""),
    }

    try:
        input_values = {
            "Age": parse_float("age"),
            "Gender": parse_float("gender"),
            "BMI": parse_float("bmi"),
            "Cholesterol": parse_float("cholesterol"),
            "Sleep": parse_float("sleep"),
            "Activity": parse_float("activity"),
            "Breakfast": parse_float("breakfast"),
        }
    except ValueError:
        recent = get_recent_predictions(session["user_id"], limit=8)
        return render_template(
            "index.html",
            prediction={"error": "Please enter valid numeric values for all fields."},
            form_data=form_data,
            recent_predictions=recent,
        )

    prediction = score_prediction(input_values)

    db = get_db()
    db.execute(
        """
        INSERT INTO predictions (
            user_id, created_at, age, gender, bmi, cholesterol, sleep, activity, breakfast,
            risk_label, risk_flag, risk_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            datetime.utcnow().isoformat(),
            input_values["Age"],
            input_values["Gender"],
            input_values["BMI"],
            input_values["Cholesterol"],
            input_values["Sleep"],
            input_values["Activity"],
            input_values["Breakfast"],
            prediction["label"],
            prediction["flag"],
            prediction["probability"],
        ),
    )
    db.commit()

    recent = get_recent_predictions(session["user_id"], limit=8)
    return render_template(
        "index.html",
        prediction=prediction,
        form_data=form_data,
        recent_predictions=recent,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    payload = build_dashboard_payload(session["user_id"])
    return render_template(
        "dashboard.html",
        payload=payload,
        trend_labels=json.dumps(payload["trend_labels"]),
        trend_values=json.dumps(payload["trend_values"]),
        risk_counts=json.dumps(payload["risk_counts"]),
        importance_labels=json.dumps(payload["importance_labels"]),
        importance_values=json.dumps(payload["importance_values"]),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
