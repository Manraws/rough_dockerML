import json
import logging
import os
import time
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ml_api")

# --- App & model -------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None


# --- Schemas -----------------------------------------------------------------
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to analyze")


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None


# --- Monitoring middleware (now actually attached) ---------------------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            }
        )
    )
    return response


# --- Routes ------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Sentiment API is running. See /docs for usage."}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    pred = pipeline.predict([request.text])[0]
    label = "positive" if pred == 1 else "negative"

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        confidence = float(pipeline.predict_proba([request.text]).max())

    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "input_length": len(request.text),
                "sentiment": label,
                "confidence": confidence,
            }
        )
    )
    return SentimentResponse(sentiment=label, confidence=confidence)


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Sentiment Tester</title></head>
    <body style="font-family: sans-serif; max-width: 500px; margin: 60px auto;">
      <h2>Sentiment Analyzer</h2>
      <textarea id="text" rows="4" style="width:100%; font-size:16px;"
                placeholder="Type a sentence..."></textarea><br><br>
      <button onclick="predict()" style="padding:8px 16px; font-size:16px;">Predict</button>
      <h3 id="result" style="margin-top:20px;"></h3>

      <script>
        async function predict() {
          const text = document.getElementById("text").value;
          const res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text: text})
          });
          const data = await res.json();
          document.getElementById("result").innerText = res.ok
            ? data.sentiment + " (confidence: " + (data.confidence ?? "n/a") + ")"
            : "Error: " + (data.detail ?? res.status);
        }
      </script>
    </body>
    </html>
    """
