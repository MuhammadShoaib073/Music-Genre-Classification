import os
import warnings
import tempfile

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

warnings.filterwarnings("ignore")

# =========================
# SETTINGS
# =========================
MODEL_PATH = "gtzan_trained_model.h5"

LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

TARGET_SHAPE = (150, 150)
CHUNK_DURATION = 4
OVERLAP_DURATION = 2


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# =========================
# AUDIO PREPROCESSING
# =========================
def load_and_preprocess_audio(file_path, target_shape=TARGET_SHAPE):
    data = []

    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    chunk_samples = CHUNK_DURATION * sample_rate
    overlap_samples = OVERLAP_DURATION * sample_rate
    step = chunk_samples - overlap_samples

    if len(audio_data) < chunk_samples:
        audio_data = np.pad(audio_data, (0, chunk_samples - len(audio_data)))

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / step)) + 1

    for i in range(num_chunks):
        start = i * step
        end = start + chunk_samples

        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk,
            sr=sample_rate
        )

        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max
        )

        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

        mel_spectrogram = resize(
            mel_spectrogram,
            target_shape
        )

        data.append(mel_spectrogram)

    return np.array(data)


# =========================
# MODEL PREDICTION
# =========================
def predict_genre(file_path):
    model = load_trained_model()

    X_test = load_and_preprocess_audio(file_path)

    predictions = model.predict(X_test)

    avg_prediction = np.mean(predictions, axis=0)

    predicted_index = np.argmax(avg_prediction)
    confidence = avg_prediction[predicted_index]

    return predicted_index, confidence


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="centered"
)

st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About Project", "Prediction"]
)


# =========================
# HOME PAGE
# =========================
if app_mode == "Home":
    st.title("🎵 Music Genre Classification System")
    st.write(
        """
        Welcome to the Music Genre Classification System.

        Upload an audio file and the model will classify it into one of the following genres:
        """
    )

    st.write(", ".join(LABELS))

    image_path = "music_genre_home.png"

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.info("Go to the Prediction page from the sidebar to upload an audio file.")


# =========================
# ABOUT PAGE
# =========================
elif app_mode == "About Project":
    st.title("About Project")

    st.write(
        """
        This project classifies music genres using deep learning.

        The audio file is converted into Mel spectrogram chunks.
        These spectrograms are passed to a CNN model trained on the GTZAN dataset.

        The model predicts one of these genres:
        """
    )

    st.write(LABELS)


# =========================
# PREDICTION PAGE
# =========================
elif app_mode == "Prediction":
    st.title("🎧 Music Genre Prediction")

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Predict Genre"):
            with st.spinner("Processing audio..."):

                file_extension = uploaded_file.name.split(".")[-1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{file_extension}"
                ) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                try:
                    predicted_index, confidence = predict_genre(temp_file_path)

                    predicted_genre = LABELS[predicted_index]

                    st.success(f"Predicted Genre: {predicted_genre}")
                    st.write(f"Confidence: {confidence:.2f}")

                    st.balloons()

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# st.subheader("hello world")
# st.selectbox("which  programming do you like?", [
#              "Python", "JavaScript", "Java", "C++"])
# st.checkbox("Python")
# st.checkbox("JavaScript")
# st.checkbox("Java")
# st.checkbox("C++")
# st.slider("Select a value", 0, 100, 50)
# st.select_slider("Select entry ", ["best ", "average", "worst",])
# st.progress(10)
# st.button("Click me")
# st.write("Upload an audio file to classify")

# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:
#     st.audio(uploaded_file)
#     st.write("Processing...")

#     # Here you will call your model later
#     st.success("Prediction: Rock 🎸")
