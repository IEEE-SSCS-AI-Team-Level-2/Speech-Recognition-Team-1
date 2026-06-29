import librosa
import numpy as np


def extract_features(file_path):

    # Load audio
    audio, sr = librosa.load(file_path,sr=22050,mono=True)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta MFCCs
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Energy
    rms = librosa.feature.rms(y=audio)
    energy = np.mean(rms)

    feature_vector = np.concatenate([mfcc_mean,delta_mean,[energy]])

    return feature_vector