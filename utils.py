# utils.py
import librosa
import numpy as np


def extract_features_from_csv_style(file_path):
    y, sr = librosa.load(file_path, sr=None)

    features = {
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)[0]),
        "rmse": np.mean(librosa.feature.rms(y=y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "mfcc1": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[0]),
        "mfcc2": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1]),
        "mfcc3": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[2]),
        "mfcc4": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[3]),
        "mfcc5": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[4]),
        "mfcc6": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[5]),
        "mfcc7": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[6]),
        "mfcc8": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[7]),
        "mfcc9": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[8]),
        "mfcc10": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[9]),
        "mfcc11": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[10]),
        "mfcc12": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[11]),
        "mfcc13": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[12]),
    }

    return np.array(list(features.values()))
