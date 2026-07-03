# ClinicKeeper No-Show Risk API

Colab'da eğitilen Random Forest modelini (`noshow_model.pkl`) canlı bir API olarak
sunar. ClinicKeeper frontend'i bu API'ye hasta verisi gönderip gerçek no-show risk
skoru alır.

## Endpoint'ler

- `GET /` — API ayakta mı kontrolü (health check)
- `POST /predict` — Tek hasta için risk tahmini

### Örnek istek (`POST /predict`)
```json
{
  "age": 45,
  "distance_to_clinic_km": 12.5,
  "consultation_fee_gbp": 250,
  "appointment_hour": 10,
  "days_in_advance": 25,
  "days_since_last_visit": 180,
  "clinic_type": "Dental",
  "appointment_type": "New Patient"
}
```

### Örnek yanıt
```json
{
  "no_show_probability": 0.577,
  "risk_percent": 58,
  "risk_level": "Orta",
  "is_risky": true,
  "threshold_used": 0.35
}
```

## Teknik detay
- Model: RandomForestClassifier (200 ağaç), scikit-learn 1.6.1 ile eğitildi
- Eşik (threshold): 0.35 (recall'ı optimize etmek için ayarlandı, ~%84 recall)
- Framework: FastAPI + Uvicorn
- 41 one-hot özellik; frontend'den gelen sade veri API içinde bu formata çevrilir

## Deploy (Render.com)
1. Bu repoyu Render'da yeni bir **Web Service** olarak bağla
2. Render `render.yaml` dosyasını otomatik okur (build + start komutları)
3. Deploy bitince `https://<servis-adi>.onrender.com` adresinde çalışır

Not: Render ücretsiz katmanında servis uzun süre kullanılmazsa uykuya geçer;
ilk istek ~30 sn gecikebilir (cold start).
