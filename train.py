from feature_extraction import FeatureExtractor, audio_waveform

import pandas as pd
import librosa
from matplotlib import pyplot as plt

DATASET_PATH = '/storage/home/psusac/'
LABELS_CSV_FILE = 'SEP-28k_labels_with_path.csv'
pos_labels=['Prolongation', 'Repetition', 'Block']

df = pd.read_csv(LABELS_CSV_FILE)
df['Path'] = DATASET_PATH + df['Path'].astype(str)

paths = df['Path'].head(5)

feature_extractor = FeatureExtractor(paths)
mfccs = feature_extractor.mfcc('features/mfcc.npy')

plt.figure()
librosa.display.specshow(mfccs[0], sr=8000, x_axis='time')
plt.savefig('mfcc.png')

plt.figure()
librosa.display.waveshow(audio_waveform('/storage/home/psusac/clips/HeStutters/0/HeStutters_0_0.wav'), sr=8000, color="blue")
plt.savefig('wave.png')

