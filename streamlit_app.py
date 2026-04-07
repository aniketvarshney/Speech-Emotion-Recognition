import streamlit as st
import requests
import tempfile
import os


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

API_URL = os.getenv("API_URL", "http://speech-emotion-recognition-production.up.railway.app")

st.title("🎙️ Speech Emotion Recognition")
st.caption(
    "Bidirectional LSTM · RAVDESS, SAVEE, TESS & CREMA-D · "
    "6 emotions · 75.37% test accuracy"
)
st.divider()


with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** Bidirectional LSTM  
    **Datasets:** RAVDESS, SAVEE, TESS, CREMA-D  
    **Samples:** 14,774  
    **Test accuracy:** 75.37%  
    """)

    st.divider()
    st.subheader("Per-class F1")
    for emotion, score in [
        ("Angry",   0.86),
        ("Neutral", 0.79),
        ("Fearful", 0.73),
        ("Happy",   0.73),
        ("Disgust", 0.72),
        ("Sad",     0.70),
    ]:
        st.progress(score, text=f"{emotion} — {score:.2f}")


st.subheader("Upload audio")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac"],
    help="Audio is trimmed/padded to 3 seconds internally.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    st.caption(f"`{uploaded_file.name}` · {uploaded_file.size / 1024:.1f} KB")
    st.divider()

    if st.button("🔍 Predict emotion", use_container_width=True, type="primary"):

        with st.spinner("Analysing audio..."):
            try:
                response = requests.post(
                f"{API_URL}/predict",
                files={
                    "audio": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
                timeout=30,
            )

                if response.status_code == 200:
                    result = response.json()
                    emotion    = result["emotion"]
                    emoji      = result["emoji"]
                    confidence = result["confidence"]

                    st.markdown(f"## {emoji} {emotion}")
                    col1, col2 = st.columns(2)
                    col1.metric("Emotion",    emotion)
                    col2.metric("Confidence", f"{confidence:.1%}")
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                    if confidence < 0.50:
                        st.warning("⚠️ Low confidence")
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"API error {response.status_code}: {detail}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to `{API_URL}`.")
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

else:
    st.info(
        "Upload a WAV, MP3, OGG, or FLAC file to get started. "
        "The model analyses the first 3 seconds of audio.",
        icon="ℹ️",
    )
