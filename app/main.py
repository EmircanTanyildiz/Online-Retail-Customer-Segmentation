from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.preprocessor import Predictor, SEGMENT_INFO
from app.schemas import CustomerInput, PredictionResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Online Retail Customer Segmentation",
    description="Müşteri segmentasyonu tahmin API'si",
    version="1.0.0",
)

predictor: Predictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    predictor = Predictor()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def get_options() -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "payment_methods": predictor.payment_methods,
        "regions": predictor.regions,
        "segments": SEGMENT_INFO,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if customer.payment_method not in predictor.payment_methods:
        raise HTTPException(status_code=422, detail="Invalid payment method.")

    if customer.region not in predictor.regions:
        raise HTTPException(status_code=422, detail="Invalid region.")

    try:
        return predictor.predict(customer.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
