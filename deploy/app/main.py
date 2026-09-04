"""
Dhaka Kacchi Review Intelligence - API + frontend server.

Serves:
  GET  /              -> the one-page frontend
  GET  /health         -> health check (used by AWS App Runner)
  POST /predict         -> {"text": "..."} -> aspect-level sentiment breakdown
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from mangum import Mangum

from app.model import ReviewAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-intelligence")

app = FastAPI(title="Dhaka Kacchi Review Intelligence", version="1.0")

logger.info("Loading model...")
analyzer = ReviewAnalyzer()
logger.info("Model loaded.")


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class AspectResult(BaseModel):
    sentiment: str
    confidence: float


class ReviewResponse(BaseModel):
    results: dict[str, AspectResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ReviewResponse)
def predict(payload: ReviewRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    try:
        result = analyzer.predict(text)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")
    return {"results": result}


# Serve the one-page frontend at the root, and its assets under /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# Lambda entry point - Mangum translates API Gateway/Function URL events
# into ASGI requests that FastAPI understands, and translates the response
# back. Running locally with `uvicorn app.main:app` is unaffected - this
# handler is only used when deployed as a Lambda container.
handler = Mangum(app)