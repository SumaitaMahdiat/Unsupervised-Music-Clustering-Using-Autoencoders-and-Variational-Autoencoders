import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3

def normalize_song_id(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_clip\d+$", "", name) 
    return name.strip()

audio_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/trims"
lyrics_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/lyric"
raw_data_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/audio"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
os.makedirs(output_folder, exist_ok=True)

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None, duration=40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

audio_features = []
song_ids = []

for fname in tqdm(os.listdir(audio_folder), desc="Extracting audio features"):
    if fname.lower().endswith(".wav"):
        song_id = normalize_song_id(fname)
        path = os.path.join(audio_folder, fname)

        try:
            feats = extract_features(path)
            audio_features.append(feats)
            song_ids.append(song_id)
        except Exception as e:
            print(f"Audio error ({fname}): {e}")

audio_df = pd.DataFrame(audio_features)
audio_df["song_id"] = song_ids
print("Audio features shape:", audio_df.shape)

def clean_english_lyrics(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

lyrics_data = []

for fname in os.listdir(lyrics_folder):
    if fname.lower().endswith(".txt"):
        song_id = normalize_song_id(fname)

        if song_id in song_ids:
            path = os.path.join(lyrics_folder, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except:
                text = ""

            lyrics_data.append({
                "song_id": song_id,
                "cleaned_lyrics": clean_english_lyrics(text)
            })

lyrics_df = pd.DataFrame(lyrics_data)
print("Lyrics entries:", lyrics_df.shape)
genre_map = {}

def read_genre(mp3_path):
    try:
        audio = MP3(mp3_path, ID3=EasyID3)
        return audio.get("genre", ["Unknown"])[0]
    except:
        return "Unknown"

for fname in os.listdir(raw_data_folder):
    path = os.path.join(raw_data_folder, fname)
    if fname.lower().endswith(".mp3") and os.path.isfile(path):
        song_id = normalize_song_id(fname)
        genre_map[song_id] = read_genre(path)

for sub in os.listdir(raw_data_folder):
    sub_path = os.path.join(raw_data_folder, sub)
    if os.path.isdir(sub_path):
        for fname in os.listdir(sub_path):
            if fname.lower().endswith(".mp3"):
                song_id = normalize_song_id(fname)
                genre_map[song_id] = read_genre(
                    os.path.join(sub_path, fname)
                )

genre_df = pd.DataFrame(
    genre_map.items(),
    columns=["song_id", "genre"]
)

print("Genre df before filter:", genre_df.shape)
genre_df = genre_df[genre_df["song_id"].isin(song_ids)]
print("Genre entries:", genre_df.shape)

audio_train, audio_temp = train_test_split(
    audio_df, test_size=0.3, random_state=42
)
audio_valid, audio_test = train_test_split(
    audio_temp, test_size=0.5, random_state=42
)

lyrics_train = lyrics_df[lyrics_df["song_id"].isin(audio_train["song_id"])]
lyrics_valid = lyrics_df[lyrics_df["song_id"].isin(audio_valid["song_id"])]
lyrics_test  = lyrics_df[lyrics_df["song_id"].isin(audio_test["song_id"])]

genre_train = genre_df[genre_df["song_id"].isin(audio_train["song_id"])]
genre_valid = genre_df[genre_df["song_id"].isin(audio_valid["song_id"])]
genre_test  = genre_df[genre_df["song_id"].isin(audio_test["song_id"])]

audio_train.to_csv(os.path.join(output_folder, "train_audio.csv"), index=False)
audio_valid.to_csv(os.path.join(output_folder, "valid_audio.csv"), index=False)
audio_test.to_csv(os.path.join(output_folder, "test_audio.csv"), index=False)

lyrics_train.to_csv(os.path.join(output_folder, "train_lyrics.csv"), index=False)
lyrics_valid.to_csv(os.path.join(output_folder, "valid_lyrics.csv"), index=False)
lyrics_test.to_csv(os.path.join(output_folder, "test_lyrics.csv"), index=False)

genre_train.to_csv(os.path.join(output_folder, "train_genre.csv"), index=False)
genre_valid.to_csv(os.path.join(output_folder, "valid_genre.csv"), index=False)
genre_test.to_csv(os.path.join(output_folder, "test_genre.csv"), index=False)

print("All 9 CSVs saved in:", output_folder)
