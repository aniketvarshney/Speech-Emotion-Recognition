import numpy as np
import tensorflow as tf
from pathlib import Path

int_to_emotion={
    0:"Neutral",
    1:"Happy",
    2:"Sad",
    3:"Angry",
    4:"Fearful",
    5:"Disgust"
}

emotion_emoji={
    "Neutral":"😐",
    "Happy":"😄",
    "Sad":"😢",
    "Angry":"😠",
    "Fearful":"😨",
    "Disgust":"🤢"
}

_model=None

def load_model(model_path:str="models/lstm_best.keras")->tf.keras.Model:
    global _model
    if _model is None:
        path=Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. "
                                    "Make sure lstm_best.keras is in the models/ folder")
        _model=tf.keras.models.load_model(str(path))
        print(f"[Model] loaded from {model_path}")
        print(f"[Model] Input shape expected: {_model.input_shape}")
    return _model

def predict(mfcc:np.ndarray)->dict:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model first.")
    x=np.expand_dims(mfcc,axis=0)
    probs=_model.predict(x,verbose=0)[0]
    top_idx=int(np.argmax(probs))
    top_emotion=int_to_emotion[top_idx]
    top_confidence=float(probs[top_idx])
    return {
        "emotion":top_emotion,
        "emoji":emotion_emoji[top_emotion],
        "confidence":round(top_confidence,4)
    }