import os
import time
import uuid
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.preprocessing import extract_mfcc, validate_audio
from app.model import load_model, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",       # mp3
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading LSTM model...")
    load_model("models/lstm_best.keras")
    logger.info("Model ready. Server is up.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Speech Emotion Recognition API",
    description=(
        "Predict emotion from audio using a Bidirectional LSTM trained on "
        "RAVDESS, SAVEE, TESS, and CREMA-D datasets. "
        "Supports 6 emotions: neutral, happy, sad, angry, fearful, disgust."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    request_id:     str
    emotion:        str
    emoji:          str
    confidence:     float
    inference_ms:   float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    version:       str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Quick liveness check — use this to confirm the server is up."""
    from app.model import _model
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_emotion(audio: UploadFile = File(...)):
    """
    Upload an audio file and get the predicted emotion.

    - Accepted formats: WAV, MP3, OGG, FLAC
    - Audio is trimmed / padded to 3 seconds internally
    - Returns top emotion + confidence for all 6 classes
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received file: {audio.filename}")

    if audio.filename:
        ext = os.path.splitext(audio.filename)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Accepted: {ALLOWED_EXTENSIONS}",
            )


    if audio.content_type and audio.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{audio.content_type}'.",
        )


    suffix = os.path.splitext(audio.filename or "audio.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        try:
            validate_audio(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        t0 = time.perf_counter()
        mfcc = extract_mfcc(tmp_path)


        result = predict(mfcc)
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"[{request_id}] → {result['emotion']} "
            f"({result['confidence']:.0%}) in {inference_ms}ms"
        )

        return PredictionResponse(
    request_id=request_id,
    emotion=result["emotion"],
    emoji=result["emoji"],
    confidence=result["confidence"],
    inference_ms=inference_ms,
)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        os.unlink(tmp_path)