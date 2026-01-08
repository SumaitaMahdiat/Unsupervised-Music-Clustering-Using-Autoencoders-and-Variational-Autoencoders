# ---------------------------
# Script 1: Data Preparation
# ---------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Paths
# ---------------------------
csv_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"

# Load CSVs
audio_train = pd.read_csv(f"{csv_folder}/train_audio.csv")
audio_valid = pd.read_csv(f"{csv_folder}/valid_audio.csv")
audio_test  = pd.read_csv(f"{csv_folder}/test_audio.csv")

lyrics_train = pd.read_csv(f"{csv_folder}/train_lyrics.csv")
lyrics_valid = pd.read_csv(f"{csv_folder}/valid_lyrics.csv")
lyrics_test  = pd.read_csv(f"{csv_folder}/test_lyrics.csv")

# Fill NaN in lyrics with empty string
lyrics_train['cleaned_lyrics'] = lyrics_train['cleaned_lyrics'].fillna("")
lyrics_valid['cleaned_lyrics'] = lyrics_valid['cleaned_lyrics'].fillna("")
lyrics_test['cleaned_lyrics'] = lyrics_test['cleaned_lyrics'].fillna("")

genre_train = pd.read_csv(f"{csv_folder}/train_genre.csv")
genre_valid = pd.read_csv(f"{csv_folder}/valid_genre.csv")
genre_test  = pd.read_csv(f"{csv_folder}/test_genre.csv")

# Merge on song_id to align rows
train_df = audio_train.merge(lyrics_train, on='song_id', how='inner').merge(genre_train, on='song_id', how='inner')
valid_df = audio_valid.merge(lyrics_valid, on='song_id', how='inner').merge(genre_valid, on='song_id', how='inner')
test_df = audio_test.merge(lyrics_test, on='song_id', how='inner').merge(genre_test, on='song_id', how='inner')

# Extract components
audio_train = train_df.drop(['song_id', 'cleaned_lyrics', 'genre'], axis=1)
lyrics_train = train_df[['song_id', 'cleaned_lyrics']]
genre_train = train_df[['song_id', 'genre']]

audio_valid = valid_df.drop(['song_id', 'cleaned_lyrics', 'genre'], axis=1)
lyrics_valid = valid_df[['song_id', 'cleaned_lyrics']]
genre_valid = valid_df[['song_id', 'genre']]

audio_test = test_df.drop(['song_id', 'cleaned_lyrics', 'genre'], axis=1)
lyrics_test = test_df[['song_id', 'cleaned_lyrics']]
genre_test = test_df[['song_id', 'genre']]

# ---------------------------
# Encode genre labels as integers
# ---------------------------
le = LabelEncoder()
genre_train['genre_label'] = le.fit_transform(genre_train['genre'])
genre_valid['genre_label'] = le.transform(genre_valid['genre'])
genre_test['genre_label'] = le.transform(genre_test['genre'])

# ---------------------------
# Lyrics embedding (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=500)  # 500 dims for speed
lyrics_train_vec = vectorizer.fit_transform(lyrics_train['cleaned_lyrics']).toarray()
lyrics_valid_vec = vectorizer.transform(lyrics_valid['cleaned_lyrics']).toarray()
lyrics_test_vec  = vectorizer.transform(lyrics_test['cleaned_lyrics']).toarray()

# ---------------------------
# Combine modalities: audio + lyrics + genre
# ---------------------------
X_train = np.hstack([audio_train.values, lyrics_train_vec, genre_train[['genre_label']].values])
X_valid = np.hstack([audio_valid.values, lyrics_valid_vec, genre_valid[['genre_label']].values])
X_test  = np.hstack([audio_test.values, lyrics_test_vec, genre_test[['genre_label']].values])

y_train = genre_train['genre_label'].values
y_valid = genre_valid['genre_label'].values
y_test  = genre_test['genre_label'].values

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape:", X_test.shape)

# Save data for other scripts
import numpy as np
np.save(f"{output_folder}/X_train.npy", X_train)
np.save(f"{output_folder}/X_valid.npy", X_valid)
np.save(f"{output_folder}/X_test.npy", X_test)
np.save(f"{output_folder}/y_train.npy", y_train)
np.save(f"{output_folder}/y_valid.npy", y_valid)
np.save(f"{output_folder}/y_test.npy", y_test)
audio_train.to_pickle(f"{output_folder}/audio_train.pkl")
genre_train.to_pickle(f"{output_folder}/genre_train.pkl")

# Save label encoder
import pickle
with open(f"{output_folder}/label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)
