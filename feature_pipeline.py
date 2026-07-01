import os
import numpy as np
import librosa


def extract_features(file_path):
    """
    Extract a 27-dimensional feature vector from a single audio file.
    (13 MFCC + 13 Delta MFCC + 1 Energy)
    """

    # Load audio
    audio, sr = librosa.load(file_path, sr=22050, mono=True)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta MFCCs
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Energy
    rms = librosa.feature.rms(y=audio)
    energy = np.mean(rms)

    return np.concatenate([mfcc_mean, delta_mean, [energy]])


def extract_person_features(person_folder):
    """
    Extract one feature vector representing a single person
    by averaging all recordings in ON and OFF folders.
    """

    all_features = []

    for state in ["ON", "OFF"]:
        state_folder = os.path.join(person_folder, state)

        if not os.path.exists(state_folder):
            continue

        for file in os.listdir(state_folder):
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                file_path = os.path.join(state_folder, file)

                try:
                    feature = extract_features(file_path)
                    all_features.append(feature)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if len(all_features) == 0:
        return None

    return np.mean(all_features, axis=0)


def build_dataset(dataset_path):
    X = []
    y = []

    # Get all unique person names from ON and OFF
    people = set()

    for state in ["ON", "OFF"]:
        state_path = os.path.join(dataset_path, state)

        if os.path.isdir(state_path):
            for person in os.listdir(state_path):
                person_path = os.path.join(state_path, person)
                if os.path.isdir(person_path):
                    people.add(person)

    # Process each person
    for person in sorted(people):
        print(f"Processing {person}...")

        features = []

        for state in ["ON", "OFF"]:
            person_path = os.path.join(dataset_path, state, person)

            if not os.path.isdir(person_path):
                continue

            for file in os.listdir(person_path):
                if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                    file_path = os.path.join(person_path, file)

                    try:
                        features.append(extract_features(file_path))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        if len(features) > 0:
            X.append(np.mean(features, axis=0))
            y.append(person)

    return np.array(X), np.array(y)




dataset_path = "dataset"

X, y = build_dataset(dataset_path)

print(X.shape)
print(X)
print(y)