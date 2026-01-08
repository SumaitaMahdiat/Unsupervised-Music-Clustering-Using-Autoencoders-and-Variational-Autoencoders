import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

audio_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/trims" 
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features/easy"
os.makedirs(output_folder, exist_ok=True)
def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None, duration=40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

all_features = []
for file_name in tqdm(os.listdir(audio_folder)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        feats = extract_features(file_path, n_mfcc=40)
        all_features.append(feats)

all_features = np.array(all_features)
print("Features shape:", all_features.shape)
X_train, X_temp = train_test_split(all_features, test_size=0.3, random_state=42)
X_valid, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
pd.DataFrame(X_train).to_csv(os.path.join(output_folder, "train.csv"), index=False)
pd.DataFrame(X_valid).to_csv(os.path.join(output_folder, "valid.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(output_folder, "test.csv"), index=False)
print("CSV files saved in", output_folder)
