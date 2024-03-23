import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

AUDIO_FREQ = 8000

def audio_waveform(file_path, fs=AUDIO_FREQ, max_len=3*AUDIO_FREQ, normalize_wav=True, add_noise=False):
    y, _ = librosa.load(file_path, mono=True, sr=fs)
    y = np.pad(y, (0, max_len - len(y)), constant_values=0)

    if add_noise:
        NOISE_MEAN = 0
        NOISE_STD = 0.01
        y += np.random.normal(NOISE_MEAN, NOISE_STD, y.shape[0])

    if normalize_wav:
        y = y / np.max(np.abs(y))

    return y

def mfcc(file_path, n_mfcc, n_fft, hop, normalize=True):
    y = audio_waveform(file_path, fs=AUDIO_FREQ)

    mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_FREQ, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, center=False)

    # Uncomment to remove the first coefficient
    # mfcc = mfcc[1:, :]

    if normalize:
        mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
        mfcc = mfcc / np.std(mfcc)
        # mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    return mfcc

def log_mel_spectrogram(file_path, n_mels, normalize):
    y = audio_waveform(file_path, fs=AUDIO_FREQ)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=AUDIO_FREQ, n_mels=n_mels, center=False)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=60)

    if normalize:
        log_mel_spec = (log_mel_spec - np.min(log_mel_spec)) / (np.max(log_mel_spec) - np.min(log_mel_spec))
        log_mel_spec = log_mel_spec / np.std(log_mel_spec)

    return log_mel_spec

class FeatureExtractor:
    paths: pd.Series

    def __init__(self, paths) -> None:
        self.paths = paths

    def mfcc(self, out_file, n_mfcc=13, n_fft=2048, hop=512, normalize=True):
        features = []

        for _, path in tqdm(self.paths.items(), position=0, leave=True, total=self.paths.shape[0]):
            print(path)
            if os.path.exists(path):
                features.append(mfcc(path, n_mfcc=n_mfcc, n_fft=n_fft, hop=hop, normalize=normalize))

        features = np.array(features)

        np.save(out_file, features)

        return features

    def log_mel_spectrogram(self, out_file, n_mels=40, normalize=True):
        features = []

        for _, path in tqdm(self.paths.items(), position=0, leave=True, total=self.paths.shape[0]):
            if os.path.exists(path):
                features.append(log_mel_spectrogram(path, n_mels=n_mels, normalize=normalize))

        features = np.array(features)

        np.save(out_file, features)

        return features

        


