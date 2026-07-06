"""
ClinicKeeper API  —  No-Show Prediction (v2, sentetik model)
FastAPI servisi. Klinik kurallara dayalı sentetik veriyle eğitilmiş modeli yükler,
insan-dostu girdileri modelin beklediği sütunlara çevirir, no-show olasılığı +
risk seviyesi döner. AI hatırlatma mesajı Groq (veya Gemini) ile üretilir.

Klinik mantık:
  - Geç iptal (24 saatten az kala) / hiç gelmeme -> ağır risk (no-show'a en yakın)
  - Tek erken iptal (24+ saat kala)              -> nötr
  - 2+ erken iptal                                -> düzensiz hasta, risk artar
  - Sadakat (gelinen randevu)                     -> riski düşürür

Render start command:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import urllib.request
import urllib.error

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

# ----------------------------------------------------------------------
# 1) Model ve şemayı yükle
# ----------------------------------------------------------------------
MODEL_PATH = "noshow_model_v2.pkl"
SCHEMA_PATH = "feature_schema_v2.json"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

FEATURE_NAMES = SCHEMA["feature_names"]      # 16 sütun, DOĞRU sırada
THRESHOLD = SCHEMA.get("threshold", 0.30)

# ----------------------------------------------------------------------
# Gemini ayarları — anahtar Render Environment Variable'dan okunur
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_gemini_client = None
def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if genai is None:
            raise RuntimeError("google-genai kütüphanesi yüklü değil.")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

# ----------------------------------------------------------------------
# Groq ayarları — Gemini alternatifi (ücretsiz). GROQ_API_KEY tanımlıysa
# mesaj üretimi otomatik Groq'a geçer. Anahtar Environment'tan okunur.
# ----------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq(prompt: str) -> str:
    """Groq'un OpenAI-uyumlu API'sine istek atar, üretilen metni döner."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 400,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "ClinicKeeper/1.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (body["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq {e.code}: {detail}")
    except Exception as e:
        raise RuntimeError(f"Groq bağlantı hatası: {e}")

# ----------------------------------------------------------------------
# Kabul edilen kategori seçenekleri
# Şube -> modeldeki one-hot sütun eşlemesi:
#   Ataşehir = baseline (iki şube sütunu da 0)
#   Bakırköy -> CLINIC_Bakırköy = 1
#   Kadıköy  -> CLINIC_Kadıköy = 1
# ----------------------------------------------------------------------
CLINIC_OPTIONS = ["Kadıköy", "Ataşehir", "Bakırköy"]
APPT_TYPE_OPTIONS = ["New", "Follow-up", "Others"]   # Follow-up = baseline

# ----------------------------------------------------------------------
# 2) İstek modeli (insan-dostu girdiler)
# ----------------------------------------------------------------------
class PredictRequest(BaseModel):
    lead_time: int = Field(..., ge=0, description="Randevuya kalan gün")
    age: int = Field(..., ge=0, le=120)
    appt_num: int = Field(1, ge=1, description="Hastanın kaçıncı randevusu")
    total_gec_iptal: int = Field(0, ge=0, description="24 saatten az kala iptal (ağır)")
    total_erken_iptal: int = Field(0, ge=0, description="24+ saat kala iptal (hafif)")
    total_rescheduled: int = Field(0, ge=0, description="Erteleme sayısı")
    total_success_appointments: int = Field(0, ge=0, description="Gelinen randevu")
    is_repeat: int = Field(1, ge=0, le=1, description="Tekrar eden hasta mı (0/1)")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Pazartesi ... 6=Pazar")
    week_of_month: int = Field(2, ge=1, le=5)
    month: int = Field(..., ge=1, le=12)
    hour_of_day: int = Field(..., ge=0, le=23)
    appt_type: str = Field("Follow-up", description=f"Seçenekler: {APPT_TYPE_OPTIONS}")
    clinic: str = Field("Ataşehir", description=f"Seçenekler: {CLINIC_OPTIONS}")


class MessageRequest(BaseModel):
    """AI hatırlatma mesajı üretmek için gereken bilgiler."""
    patient_name: str = Field("Değerli hastamız")
    risk_band: str = Field(..., description="Düşük / Orta / Yüksek")
    noshow_percent: float = Field(..., description="No-show yüzdesi (0-100)")
    clinic: str = Field("kliniğimiz")
    appt_type: str = Field("Follow-up")
    lead_time: int = Field(0)
    tone: str = Field("samimi", description="samimi / resmi / kısa")


# ----------------------------------------------------------------------
# 3) Girdiyi model sütunlarına çevir (one-hot), şema sırasına göre
# ----------------------------------------------------------------------
def build_feature_row(r: PredictRequest) -> pd.DataFrame:
    row = {name: 0 for name in FEATURE_NAMES}

    # Sayısal alanlar
    row["LEAD_TIME"] = r.lead_time
    row["AGE"] = r.age
    row["APPT_NUM"] = r.appt_num
    row["TOTAL_GEC_IPTAL"] = r.total_gec_iptal
    row["TOTAL_ERKEN_IPTAL"] = r.total_erken_iptal
    row["TOTAL_RESCHEDULED"] = r.total_rescheduled
    row["TOTAL_SUCCESS"] = r.total_success_appointments
    row["IS_REPEAT"] = r.is_repeat
    row["DAY_OF_WEEK"] = r.day_of_week
    row["HOUR_OF_DAY"] = r.hour_of_day
    row["NUM_OF_MONTH"] = r.month
    row["WEEK_OF_MONTH"] = r.week_of_month

    # Randevu tipi (Follow-up = baseline)
    if r.appt_type == "New":
        row["APPT_TYPE_New"] = 1
    elif r.appt_type == "Others":
        row["APPT_TYPE_Others"] = 1

    # Şube (Ataşehir = baseline)
    if r.clinic == "Bakırköy":
        row["CLINIC_Bakırköy"] = 1
    elif r.clinic == "Kadıköy":
        row["CLINIC_Kadıköy"] = 1

    return pd.DataFrame([[row[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES).astype(float)


def risk_band(prob: float) -> str:
    if prob >= 0.40:
        return "Yüksek"
    elif prob >= 0.20:
        return "Orta"
    return "Düşük"


# ----------------------------------------------------------------------
# 4) FastAPI uygulaması
# ----------------------------------------------------------------------
app = FastAPI(title="ClinicKeeper API", version="3.0-synthetic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "ClinicKeeper API",
        "version": "3.0-synthetic",
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
    return {
        "features": FEATURE_NAMES,
        "threshold": THRESHOLD,
        "options": {
            "clinic": CLINIC_OPTIONS,
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
            "will_flag": prob >= THRESHOLD,
            "threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------------------------------------------------
# 5) AI hatırlatma mesajı
# ----------------------------------------------------------------------
def build_prompt(r: MessageRequest) -> str:
    tone_map = {
        "samimi": "sıcak, samimi ve nazik",
        "resmi": "resmi, saygılı ve profesyonel",
        "kısa": "çok kısa ve öz",
    }
    tone_desc = tone_map.get(r.tone, "sıcak ve nazik")
    return (
        f"Sen bir diş kliniğinin randevu hatırlatma asistanısın. "
        f"Aşağıdaki hasta için {tone_desc} bir Türkçe hatırlatma mesajı yaz.\n\n"
        f"Hasta adı: {r.patient_name}\n"
        f"Klinik: {r.clinic}\n"
        f"Randevu tipi: {r.appt_type}\n"
        f"Randevuya kalan gün: {r.lead_time}\n"
        f"No-show risk seviyesi: {r.risk_band} (%{r.noshow_percent})\n\n"
        f"Kurallar:\n"
        f"- Sadece mesaj metnini yaz, başka açıklama ekleme.\n"
        f"- Hastayı suçlama, baskı kurma; nazikçe hatırlat ve gelmesini kolaylaştır.\n"
        f"- Randevuyu onaylama veya değiştirme için kliniği aramaya davet et.\n"
        f"- Mesaj 2-4 cümle olsun.\n"
    )


@app.post("/generate-message")
def generate_message(req: MessageRequest):
    prompt = build_prompt(req)

    # 1) Groq anahtarı varsa önce Groq
    if GROQ_API_KEY:
        try:
            text = call_groq(prompt)
            if not text:
                raise HTTPException(status_code=502, detail="Groq boş cevap döndü.")
            return {"message": text, "model": GROQ_MODEL, "provider": "groq"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq hatası: {str(e)}")

    # 2) Groq yoksa Gemini
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI anahtarı tanımlı değil. Render'a GROQ_API_KEY (önerilen) "
                   "veya GEMINI_API_KEY ekleyin.",
        )
    try:
        client = get_gemini_client()
        config = genai_types.GenerateContentConfig(temperature=0.8, max_output_tokens=400)
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config,
        )
        text = (resp.text or "").strip()
        if not text:
            raise HTTPException(status_code=502, detail="Gemini boş cevap döndü.")
        return {"message": text, "model": GEMINI_MODEL, "provider": "gemini"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini hatası: {str(e)}")
