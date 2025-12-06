import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# --- 1. INISIALISASI APLIKASI ---
app = FastAPI(title="Churn Prediction API (Profit Max)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "Buka Pintu" untuk semua website (Aman buat development)
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan method POST, GET, dll
    allow_headers=["*"],
)

# Variabel global untuk menyimpan model
model_data = {}

# --- 2. LOAD MODEL SAAT SERVER NYALA ---
@app.on_event("startup")
def load_model():
    global model_data
    try:
        # Load file pickle yang sudah kita simpan tadi
        with open("churn_model_xgb_profit_max.pkl", "rb") as f:
            model_data = pickle.load(f)
        print("✅ Model dan Threshold berhasil dimuat!")
        print(f"💰 Threshold Operasi: {model_data['threshold']}")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")

# --- 3. DEFINISI INPUT DATA (SCHEMA) ---
# Kita gunakan Dict karena kolom hasil One-Hot Encoding banyak (30+)
# User harus mengirim data dalam format JSON dictionary
class CustomerData(BaseModel):
    features: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "gender": 1,
                    "SeniorCitizen": 0,
                    "Partner": 1,
                    "Dependents": 0,
                    "tenure": 12,              # Sudah langganan 1 tahun
                    "PhoneService": 1,
                    "PaperlessBilling": 1,
                    "MonthlyCharges": 70.35,
                    
                    # 2. Fitur Hasil One-Hot Encoding (Boolean/0-1)
                    # Ingat: TotalCharges TIDAK ADA karena sudah kita drop saat training.
                    
                    "MultipleLines_No phone service": 0,
                    "MultipleLines_Yes": 0,  # Berarti dia pakai MultipleLines='No' (Base case)
                    
                    "InternetService_Fiber optic": 1, # Pakai Fiber Optic
                    "InternetService_No": 0,
                    
                    "OnlineSecurity_No internet service": 0,
                    "OnlineSecurity_Yes": 0,
                    
                    "OnlineBackup_No internet service": 0,
                    "OnlineBackup_Yes": 0,
                    
                    "DeviceProtection_No internet service": 0,
                    "DeviceProtection_Yes": 1, # Punya proteksi device
                    
                    "TechSupport_No internet service": 0,
                    "TechSupport_Yes": 0,
                    
                    "StreamingTV_No internet service": 0,
                    "StreamingTV_Yes": 1, # Suka nonton TV streaming
                    
                    "StreamingMovies_No internet service": 0,
                    "StreamingMovies_Yes": 1, # Suka nonton Movie streaming
                    
                    "Contract_One year": 0,
                    "Contract_Two year": 0, # Berarti dia Month-to-month (Bahaya!)
                    
                    "PaymentMethod_Credit card (automatic)": 0,
                    "PaymentMethod_Electronic check": 1, # Bayar manual (Bahaya!)
                    "PaymentMethod_Mailed check": 0
                }
            }
        }

# --- 4. ENDPOINT UTAMA (PREDIKSI) ---
@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model_data:
        raise HTTPException(status_code=500, detail="Model belum dimuat.")
    
    model = model_data['model']
    threshold = model_data['threshold']
    required_columns = model_data['features'] # List nama kolom dari X_train

    try:
        # A. Konversi JSON ke DataFrame
        input_df = pd.DataFrame([data.features])

        # B. MENYAMAKAN KOLOM (CRUCIAL STEP!)
        # Masalah: User mungkin lupa kirim input "Partner_Yes": 0
        # Solusi: Kita isi kolom yang hilang dengan 0
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Pastikan urutan kolom SAMA PERSIS dengan saat training
        input_df = input_df[required_columns]

        # C. Prediksi Probabilitas
        probabilitas = model.predict_proba(input_df)[:, 1][0]

        # D. Logika Bisnis (Threshold Profit Max)
        is_churn = bool(probabilitas >= threshold)
        
        # E. Susun Respon JSON
        result = {
            "prediction_label": "CHURN" if is_churn else "STAY",
            "churn_probability": round(float(probabilitas), 4),
            "threshold_used": threshold,
            "business_recommendation": (
                "BAHAYA! Segera kirim voucher retensi $10." 
                if is_churn 
                else "AMAN. Tidak perlu tindakan."
            )
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 5. ENDPOINT CEK KESEHATAN ---
@app.get("/")
def home():
    return {"message": "API Churn Prediction Aktif. Gunakan endpoint /predict untuk memprediksi."}