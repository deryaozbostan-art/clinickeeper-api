"""
ClinicKeeper API  —  CHLA No-Show Prediction
FastAPI servisi. Modeli ve şemayı diskten yükler, insan-dostu girdileri
modelin beklediği 26 sütuna çevirir, no-show olasılığı + risk seviyesi döner.

Render start command:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ----------------------------------------------------------------------
# 1) Model ve şemayı yükle
# ----------------------------------------------------------------------
MODEL_PATH = "chla_noshow_model.pkl"
SCHEMA_PATH = "feature_schema.json"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r") as f:
    SCHEMA = json.load(f)

FEATURE_NAMES = SCHEMA["feature_names"]      # 26 sütun, DOĞRU sırada
THRESHOLD = SCHEMA.get("threshold", 0.15)

# Kabul edilen kategori seçenekleri (frontend'in gönderebileceği değerler)
CLINIC_OPTIONS = [
    "BAKERSFIELD CARE CLINIC",
    "ENCINO CARE CENTER",
    "SANTA MONICA CLINIC",
    "SOUTH BAY CARE CENTER",
    "VALENCIA CARE CENTER",
    "OTHER",   # yukarıdakilerden biri değilse (tüm klinik sütunları 0 kalır)
]
RACE_OPTIONS = [
    "Asian", "European", "MiddleEastern",
    "NorthAmerican", "Other", "SouthAmerican",
]
ETHNICITY_OPTIONS = ["Non-Hispanic", "Others", "Hispanic"]  # Hispanic = baseline (her iki sütun 0)
APPT_TYPE_OPTIONS = ["New", "Others", "Follow-up"]          # Follow-up = baseline (her iki sütun 0)

# ----------------------------------------------------------------------
# 2) İstek modeli (insan-dostu girdiler)
# ----------------------------------------------------------------------
class PredictRequest(BaseModel):
    lead_time: int = Field(..., ge=0, description="Randevu ile kayıt arasındaki gün sayısı")
    age: int = Field(..., ge=0, le=120)
    appt_num: int = Field(1, ge=1, description="Hastanın kaçıncı randevusu")
    total_cancellations: int = Field(0, ge=0)
    total_rescheduled: int = Field(0, ge=0)
    total_success_appointments: int = Field(0, ge=0)
    is_repeat: int = Field(0, ge=0, le=1, description="Tekrar eden hasta mı (0/1)")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Pazartesi ... 6=Pazar")
    week_of_month: int = Field(..., ge=1, le=5)
    month: int = Field(..., ge=1, le=12)
    hour_of_day: int = Field(..., ge=0, le=23)
    appt_type: str = Field("Follow-up", description=f"Seçenekler: {APPT_TYPE_OPTIONS}")
    ethnicity: str = Field("Hispanic", description=f"Seçenekler: {ETHNICITY_OPTIONS}")
    race: str = Field("Other", description=f"Seçenekler: {RACE_OPTIONS}")
    clinic: str = Field("OTHER", description=f"Seçenekler: {CLINIC_OPTIONS}")


# ----------------------------------------------------------------------
# 3) Girdiyi 26 sütuna çevir (one-hot), şema sırasına göre
# ----------------------------------------------------------------------
def build_feature_row(r: PredictRequest) -> pd.DataFrame:
    # Tüm sütunları 0 ile başlat
    row = {name: 0 for name in FEATURE_NAMES}

    # Sayısal alanlar
    row["LEAD_TIME"] = r.lead_time
    row["IS_REPEAT"] = r.is_repeat
    row["APPT_NUM"] = r.appt_num
    row["TOTAL_NUMBER_OF_CANCELLATIONS"] = r.total_cancellations
    row["TOTAL_NUMBER_OF_RESCHEDULED"] = r.total_rescheduled
    row["TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT"] = r.total_success_appointments
    row["DAY_OF_WEEK"] = r.day_of_week
    row["WEEK_OF_MONTH"] = r.week_of_month
    row["NUM_OF_MONTH"] = r.month
    row["HOUR_OF_DAY"] = r.hour_of_day
    row["AGE"] = r.age

    # Randevu tipi (Follow-up = baseline, iki sütun da 0 kalır)
    if r.appt_type == "New":
        row["APPT_TYPE_STANDARDIZE_New"] = 1
    elif r.appt_type == "Others":
        row["APPT_TYPE_STANDARDIZE_Others"] = 1

    # Etnik köken (Hispanic = baseline)
    if r.ethnicity == "Non-Hispanic":
        row["ETHNICITY_STANDARDIZE_Non-Hispanic"] = 1
    elif r.ethnicity == "Others":
        row["ETHNICITY_STANDARDIZE_Others"] = 1

    # Irk (biri seçili olmalı; tanınmazsa 'Other')
    race_col = f"RACE_STANDARDIZE_{r.race}"
    if race_col in row:
        row[race_col] = 1
    else:
        row["RACE_STANDARDIZE_Other"] = 1

    # Klinik (OTHER = baseline, tüm klinik sütunları 0)
    clinic_col = f"CLINIC_{r.clinic}"
    if clinic_col in row:
        row[clinic_col] = 1

    # Şema sırasına göre tek satırlık DataFrame
    return pd.DataFrame([[row[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)


def risk_band(prob: float) -> str:
    if prob >= 0.40:
        return "Yüksek"
    elif prob >= THRESHOLD:
        return "Orta"
    return "Düşük"


# ----------------------------------------------------------------------
# 4) FastAPI uygulaması
# ----------------------------------------------------------------------
app = FastAPI(title="ClinicKeeper API", version="2.0-CHLA")

# GitHub Pages frontend'inin API'yi çağırabilmesi için CORS açık
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # istersen sadece kendi domain'ini yazabilirsin
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "ClinicKeeper API",
        "version": "2.0-CHLA",
        "model": SCHEMA.get("model_type"),
        "threshold": THRESHOLD,
        "n_features": len(FEATURE_NAMES),
        "status": "ok",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/schema")
def schema():
    """Frontend'in hangi alanları göndermesi gerektiğini öğrenmesi için."""
    return {
        "features": FEATURE_NAMES,
        "threshold": THRESHOLD,
        "options": {
            "clinic": CLINIC_OPTIONS,
            "race": RACE_OPTIONS,
            "ethnicity": ETHNICITY_OPTIONS,
            "appt_type": APPT_TYPE_OPTIONS,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = build_feature_row(req)
        prob = float(model.predict_proba(X)[0, 1])
        return {
            "noshow_probability": round(prob, 4),
            "noshow_percent": round(prob * 100, 1),
            "risk_band": risk_band(prob),
            "will_flag": prob >= THRESHOLD,   # bu hastaya hatırlatma gitmeli mi?
            "threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
