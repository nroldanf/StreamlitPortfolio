import librosa
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt

def load_audio(audio_filename):
    v_signal, fs = librosa.load(audio_filename, sr=None)
    t = np.arange(0.0,len(v_signal)/fs,(1/fs))
    return t, v_signal, fs

def plot_audio(t, v_signal, fs):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, v_signal)
    ax.grid(True)
    ax.set_xlim(0, len(v_signal)/fs)
    return fig

def plot_mel_spectrogram(v_signal, fs, n_mels):
    ps = librosa.feature.melspectrogram(y=v_signal, sr=fs, n_mels=n_mels)
    ps = librosa.power_to_db(ps, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(ps, y_axis='mel', x_axis='time', ax=ax, sr=fs)
    plt.title('Mel-frequency spectrogram')
    return fig


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)