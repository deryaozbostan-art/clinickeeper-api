"""
ClinicKeeper No-Show Risk API
-------------------------------
Colab'da egitilen Random Forest modelini (.pkl) yukleyip,
gelen hasta verisi icin no-show risk skoru dondurur.

Endpoint'ler:
  GET  /          -> API ayakta mi kontrolu (health check)
  POST /predict   -> Tek bir hasta icin risk tahmini
"""

import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Model ve yardimci dosyalari yukle (uygulama baslarken bir kez) ----
model = joblib.load("noshow_model.pkl")
MODEL_COLUMNS = json.load(open("model_columns.json"))
MODEL_INFO = json.load(open("model_info.json"))
THRESHOLD = MODEL_INFO.get("threshold", 0.35)

# ---- FastAPI uygulamasi ----
app = FastAPI(title="ClinicKeeper No-Show Risk API")

# Tarayicidan (farkli bir domain'den) istek gelebilmesi icin CORS izni.
# ClinicKeeper frontend'i baska bir adreste calistigi icin bu sart.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # demo icin herkese acik; gercek urunde kisitlanir
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Gelen hasta verisinin bekledigimiz sekli (frontend bunu gonderecek) ----
class Patient(BaseModel):
    age: int
    distance_to_clinic_km: float
    consultation_fee_gbp: float
    appointment_hour: int
    days_in_advance: int
    days_since_last_visit: int
    patient_total_appointments: int = 0
    patient_previous_noshow_count: int = 0
    is_weekend: int = 0
    is_peak_hour: int = 0
    has_chronic_condition_flag: int = 0
    sms_reminder_sent: int = 1
    reminder_hours_before: int = 24
    number_of_reminders_sent: int = 1
    gender: str = "Male"                 # "Male" / "Female"
    clinic_type: str = "Dental"          # Dental / GP / Dermatology / Physiotherapy
    appointment_type: str = "New Patient"  # "New Patient" / "Follow-up"
    reminder_response: str = "No Reminder" # Confirmed / Ignored / No Reminder / (bos)


def build_feature_row(p: Patient) -> pd.DataFrame:
    """
    Frontend'den gelen sade hasta bilgisini, modelin bekledigi
    41 sutunluk one-hot formata cevirir. Eksik sutunlar 0 olur.
    """
    row = {c: 0 for c in MODEL_COLUMNS}

    # Sayisal alanlar (isim birebir ayni)
    row["age"] = p.age
    row["distance_to_clinic_km"] = p.distance_to_clinic_km
    row["consultation_fee_gbp"] = p.consultation_fee_gbp
    row["appointment_hour"] = p.appointment_hour
    row["days_in_advance"] = p.days_in_advance
    row["days_since_last_visit"] = p.days_since_last_visit
    row["patient_total_appointments"] = p.patient_total_appointments
    row["patient_previous_noshow_count"] = p.patient_previous_noshow_count
    row["is_weekend"] = p.is_weekend
    row["is_peak_hour"] = p.is_peak_hour
    row["has_chronic_condition_flag"] = p.has_chronic_condition_flag
    row["sms_reminder_sent"] = p.sms_reminder_sent
    row["reminder_hours_before"] = p.reminder_hours_before
    row["number_of_reminders_sent"] = p.number_of_reminders_sent

    # Kategorik alanlar -> one-hot (sadece ilgili sutunu 1 yap, sutun varsa)
    def set_if_exists(col_name):
        if col_name in row:
            row[col_name] = 1

    if p.gender == "Male":
        set_if_exists("gender_Male")
    if p.clinic_type != "Dental":  # Dental referans (drop_first), digerleri sutun
        set_if_exists(f"clinic_type_{p.clinic_type}")
    if p.appointment_type == "New Patient":
        set_if_exists("appointment_type_New Patient")
    if p.reminder_response:
        set_if_exists(f"reminder_response_{p.reminder_response}")

    # Sutun sirasini modelin bekledigi sekilde sabitle
    return pd.DataFrame([row])[MODEL_COLUMNS]


@app.get("/")
def health():
    """API ayakta mi? Render'in uyanip uyanmadigini kontrol icin de kullanilir."""
    return {
        "status": "ok",
        "model": MODEL_INFO.get("model_type"),
        "threshold": THRESHOLD,
        "message": "ClinicKeeper No-Show Risk API calisiyor.",
    }


@app.post("/predict")
def predict(patient: Patient):
    """Tek bir hasta icin no-show risk skoru ve etiketi dondurur."""
    X = build_feature_row(patient)
    proba = float(model.predict_proba(X)[0, 1])  # no-show olasiligi (0..1)

    risk_percent = round(proba * 100)
    is_risky = proba >= THRESHOLD

    # 0-100 skoru uc gruba ayir (frontend'deki renklerle uyumlu)
    if risk_percent >= 70:
        level = "Yuksek"
    elif risk_percent >= 40:
        level = "Orta"
    else:
        level = "Dusuk"

    return {
        "no_show_probability": round(proba, 4),
        "risk_percent": risk_percent,
        "risk_level": level,
        "is_risky": is_risky,
        "threshold_used": THRESHOLD,
    }
