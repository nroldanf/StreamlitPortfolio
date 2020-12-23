import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import io

from utils import *

n_fft = 256
n_mels = 128
formato = "%+2.0f DB"

# PAGES = {
#     "Home": "",
#     "Resources": src.pages.resources,
#     "Gallery": src.pages.gallery.index,
#     "Vision": src.pages.vision,
#     "About": src.pages.about,
# }

v_signal, fs = librosa.load("utils/voice1.mp3", sr=None)
t = np.arange(0.0,len(v_signal)/fs,(1/fs))

# Time signal
fig_audio = plot_audio(t, v_signal, fs)

# Short Fourier Transform
fft = librosa.stft(v_signal, n_fft)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(fft), ref=np.max)

# Mel spectrogram
fig_mel = plot_mel_spectrogram(v_signal, fs, 128)


st.title('Voice Verification App')
st.sidebar.title("Menu")
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

col1, col2 = st.beta_columns(2)

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a"])    
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    filename = "{}.{}".format("audio", uploaded_file.name.split(".")[-1])

    st.write(file_details)
    st.audio(uploaded_file.read(), format=uploaded_file.type)

    uploaded_file.seek(0)

    with open(filename, "bx") as f:
        f.write(uploaded_file.read())
    # Audio signal
    v_signal, fs = librosa.load(filename, sr=None)
    t = np.arange(0.0,len(v_signal)/fs,(1/fs))

    # Time signal
    fig_audio = plot_audio(t, v_signal, fs)

    # Mel Spectrogram
    fig_mel = plot_mel_spectrogram(v_signal, fs, 128)

    os.remove(filename)

    col1.pyplot(fig_audio, clear_figure=False)
    col2.pyplot(fig_mel, clear_figure=False)
