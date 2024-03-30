import sys
import os

# Hack to be able to import modules from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from feature_extraction import FeatureExtractor

import pandas as pd
import librosa
from matplotlib import pyplot as plt

NUM_SAMPLES = 10
SAMPLE_TO_PLOT = 5
AUDIO_FREQ = 8000
SPEC_PLOT_FILE = 'tests/output/log_mel_spec.png'
WAV_PLOT_FILE = 'tests/output/wav.png'
AUGMENTED_SPEC_PLOT_FILE = 'tests/output/log_mel_spec_aug.png'
AUGMENTATIONS = ['add_noise', 'shift', 'pitch_shift']

df = pd.read_csv('SEP-28k_labels_with_path.csv').head(NUM_SAMPLES)

feature_extractor = FeatureExtractor(df['Path'])

spec_array = feature_extractor.extract(FeatureExtractor.log_mel_spectrogram, n_mels=128, n_fft=512, hop=184)

print('Spectrogram array shape: ', spec_array.shape)

spec_to_plot = spec_array[SAMPLE_TO_PLOT]
wave_to_plot, _ = librosa.load(df['Path'][SAMPLE_TO_PLOT], mono=True, sr=AUDIO_FREQ)

plt.figure()
librosa.display.waveshow(wave_to_plot, sr=AUDIO_FREQ, color='blue')
plt.savefig(WAV_PLOT_FILE)

plt.figure()
librosa.display.specshow(spec_to_plot, sr=AUDIO_FREQ, n_fft=512, hop_length=256, x_axis='time')
plt.savefig(SPEC_PLOT_FILE)

# Test data augmentation
# feature_extractor = FeatureExtractor(df['Path'], augmentations=AUGMENTATIONS)
# augm_spec_array = feature_extractor.extract(FeatureExtractor.log_mel_spectrogram, n_mels=128, n_fft=512, hop=256)
# spec_to_plot = augm_spec_array[SAMPLE_TO_PLOT]

# plt.figure()
# librosa.display.specshow(spec_to_plot, sr=AUDIO_FREQ, n_fft=512, hop_length=256, x_axis='time')
# plt.savefig(AUGMENTED_SPEC_PLOT_FILE)

