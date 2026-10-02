from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(title="Sentiment Analysis API")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to analyze")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None

@app.get("/")
def root():
    return {"message": "Sentiment API is running. See /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

# @app.post("/predict", response_model=SentimentResponse)
# def predict(request: SentimentRequest):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")

#     pred = pipeline.predict([request.text])[0]
#     label = "positive" if pred == 1 else "negative"

#     confidence = None
#     if hasattr(pipeline, "predict_proba"):
#         confidence = float(pipeline.predict_proba([request.text]).max())

#     return SentimentResponse(sentiment=label, confidence=confidence)


import logging
import time

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(request: SentimentRequest):

    start_time = time.time()

    logger.info("Prediction request received")

    prediction = model.predict([request.text])[0]

    response_time = time.time() - start_time

    logger.info(
        "Prediction completed | prediction=%s | response_time=%.3f",
        prediction,
        response_time
    )

    return {
        "prediction": prediction
    }



import json
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request

# --- Basic logging setup -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("ml_api")


# --- Logging helpers ---------------------------------------------------------

def log_prediction_event(
    request_body: dict,
    prediction: dict,
    latency_ms: float,
) -> None:
    """
    Log a single prediction event as structured JSON.

    In a real production system, this would go to a log aggregator
    (e.g., CloudWatch, Datadog, ELK) rather than stdout — but the
    structured format is the same idea.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_body,
        "output": prediction,
        "latency_ms": round(latency_ms, 2),
    }
    logger.info(json.dumps(event))


# --- Middleware --------------------------------------------------------------

def add_monitoring_middleware(app: FastAPI) -> FastAPI:
    """
    Attach a middleware that times every request and logs basic metrics.
    """

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

    return app


from fastapi.responses import HTMLResponse
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
          document.getElementById("result").innerText =
            data.sentiment + " (confidence: " + (data.confidence ?? "n/a") + ")";
        }
      </script>
    </body>
    </html>
    """
