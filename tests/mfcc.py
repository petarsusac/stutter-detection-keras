import sys
import os

# Hack to be able to import modules from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from feature_extraction import FeatureExtractor

import pandas as pd
import librosa
from matplotlib import pyplot as plt

NUM_SAMPLES = 10
SAMPLE_TO_PLOT = 1
AUDIO_FREQ = 8000
MFCC_PLOT_FILE = 'tests/output/mfcc.png'
WAV_PLOT_FILE = 'tests/output/wav.png'

df = pd.read_csv('SEP-28k_labels_with_path.csv').head(NUM_SAMPLES)

feature_extractor = FeatureExtractor(df['Path'])

mfccs = feature_extractor.extract(feature_extractor.mfcc, n_mfcc=13, n_fft=512, hop=256)

print('MFCC array shape: ', mfccs.shape)

mfcc_to_plot = mfccs[SAMPLE_TO_PLOT]
wave_to_plot, _ = librosa.load(df['Path'][SAMPLE_TO_PLOT], mono=True, sr=AUDIO_FREQ)

plt.figure()
librosa.display.waveshow(wave_to_plot, sr=AUDIO_FREQ, color='blue')
plt.savefig(WAV_PLOT_FILE)

plt.figure()
librosa.display.specshow(mfcc_to_plot, sr=AUDIO_FREQ, n_fft=512, hop_length=256, x_axis='time')
plt.savefig(MFCC_PLOT_FILE)
