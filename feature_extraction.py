import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from typing import Callable, List

AUDIO_FREQ = 8000

class FeatureExtractor:
    paths: pd.DataFrame

    def __init__(self, paths: pd.DataFrame) -> None:
        self.paths = paths

    def audio_waveform(file_path, fs=AUDIO_FREQ, max_len=3*AUDIO_FREQ, normalize_wav=True, augmentations=[]):
        y, _ = librosa.load(file_path, mono=True, sr=fs)
        y = np.pad(y, (0, max_len - len(y)), constant_values=0)

        if 'add_noise' in augmentations:
            NOISE_MEAN = 0
            NOISE_STD = 0.01
            y += np.random.normal(NOISE_MEAN, NOISE_STD, y.shape[0])

        if 'shift' in augmentations:
            MAX_SHIFT_SEC = 0.5
            SHIFT_DIRECTION = [-1, 1]
            shift = np.random.randint(fs * MAX_SHIFT_SEC)
            shift_dir = np.random.choice(SHIFT_DIRECTION)
            
            y = np.roll(y, shift_dir * shift)

        if 'pitch_shift' in augmentations:
            shift_steps = np.random.randint(-3, 3)
            y = librosa.effects.pitch_shift(y, sr=fs, n_steps=shift_steps)

        if normalize_wav:
            y = y / np.max(np.abs(y))

        return y

    def mfcc(file_path, n_mfcc=13, n_fft=2048, hop=512, normalize=True, augmentations=[]):
        y = FeatureExtractor.audio_waveform(file_path, fs=AUDIO_FREQ, augmentations=augmentations)

        mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_FREQ, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, center=False)

        if normalize:
            mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
            mfcc = mfcc / np.std(mfcc)

        return mfcc

    def mel_spectrogram(file_path, n_mels=40, n_fft=2048, hop=512, normalize=True, log=True, transpose=False, augmentations=[]):
        y = FeatureExtractor.audio_waveform(file_path, fs=AUDIO_FREQ, augmentations=augmentations)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=AUDIO_FREQ, n_mels=n_mels, n_fft=n_fft, hop_length=hop, center=False)
        
        if log:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=60)

        if normalize:
            mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
            mel_spec = mel_spec / np.std(mel_spec)

        if transpose:
            mel_spec = np.transpose(mel_spec)

        return mel_spec
    
    def extract(self, function: Callable, out_file: str = '', augmentations=[], **kwargs):
        features = []

        for _, row in tqdm(self.paths.iterrows(), position=0, leave=False, total=self.paths.shape[0]):
            if row['Augment']:
                features.append(function(row['Path'], augmentations=augmentations, **kwargs))
            else:
                features.append(function(row['Path'], augmentations=[], **kwargs))

        features = np.array(features)

        if out_file:
            np.save(out_file, features)

        return features

        


