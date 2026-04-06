# Speech Emotion Recognition (SER)

A production-ready Speech Emotion Recognition system built with a Bidirectional LSTM, served via FastAPI, with a Streamlit frontend.

**Live demo:** [Add Hugging Face Spaces link here]

---

## Results

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| **LSTM v1** (deployed) | **75.37%** | 0.6979 |
| CNN (baseline) | 43.98% | 2.0696 |

### Per-class performance (LSTM v1)

| Emotion | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| angry   | 0.84 | 0.87 | **0.86** | 377 |
| neutral | 0.79 | 0.79 | **0.79** | 330 |
| fearful | 0.78 | 0.68 | **0.73** | 378 |
| happy   | 0.73 | 0.73 | **0.73** | 378 |
| disgust | 0.78 | 0.67 | **0.72** | 377 |
| sad     | 0.63 | 0.78 | **0.70** | 377 |

Sad and disgust show the most confusion — expected, as both share low energy and slow tempo acoustic characteristics.

---

## Dataset

| Dataset | Emotions | Samples |
|---------|----------|---------|
| RAVDESS | 8 (6 used) | ~1440 |
| SAVEE   | 7 (6 used) | ~480  |
| TESS    | 7 (6 used) | ~2800 |
| CREMA-D | 6         | ~7442 |
| **Total** | **6** | **14,774** |

Emotions: `neutral` `happy` `sad` `angry` `fearful` `disgust`

---

## Architecture

```
Input: MFCC features (130 timesteps × 40 coefficients)
       ↓
Bidirectional LSTM (128 units, return_sequences=True)
       ↓
Dropout (0.3)
       ↓
Bidirectional LSTM (64 units)
       ↓
Dropout (0.3) → BatchNormalization
       ↓
Dense (64, ReLU) → Dropout (0.3)
       ↓
Dense (6, Softmax)
```

**Feature extraction:**
- Sample rate: 22,050 Hz
- Duration: 3 seconds (pad/truncate)
- n_mfcc: 40 coefficients
- max_frames: 130
- Normalisation: per-feature z-score

---

## Project structure

```
ser_project/
├── app/
│   ├── main.py          # FastAPI app — /predict and /health endpoints
│   ├── model.py         # Model loading and inference
│   ├── preprocessing.py # Audio → MFCC pipeline
│   └── __init__.py
├── models/
│   └── lstm_best.keras  # Trained model (not tracked in git — see below)
├── streamlit_app.py     # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running locally

**Prerequisites:** Python 3.11, Docker (optional)

### Option 1 — Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/yourusername/ser-project
cd ser-project

# Place your trained model
cp /path/to/lstm_best.keras models/

# Run everything
docker compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

### Option 2 — Local Python

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Start FastAPI
uvicorn app.main:app --reload

# In a second terminal — start Streamlit
streamlit run streamlit_app.py
```

---

## API

### `POST /predict`

Upload an audio file, get emotion prediction.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@your_audio.wav"
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "emotion": "angry",
  "emoji": "😠",
  "confidence": 0.8732,
  "low_confidence": false,
  "all_emotions": {
    "neutral": 0.0341,
    "happy":   0.0218,
    "sad":     0.0109,
    "angry":   0.8732,
    "fearful": 0.0412,
    "disgust": 0.0188
  },
  "inference_ms": 87.4
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

---

## Model not in Git

The trained model file (`lstm_best.keras`, ~15MB) is not tracked in this repo.

Download: [Add link — Hugging Face Hub / Google Drive]

Or retrain from scratch:
```bash
# 1. Run feature extraction
jupyter nbconvert --to script 2_feature_extraction.ipynb --execute

# 2. Run training
jupyter nbconvert --to script 3_model_training.ipynb --execute
```

---

## Tech stack

`TensorFlow 2.16` · `FastAPI` · `Streamlit` · `librosa` · `Docker`