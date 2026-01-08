import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

input_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
os.makedirs(output_folder, exist_ok=True)

train_df = pd.read_csv(os.path.join(input_folder, "train_lyrics.csv"))
valid_df = pd.read_csv(os.path.join(input_folder, "valid_lyrics.csv"))
test_df  = pd.read_csv(os.path.join(input_folder, "test_lyrics.csv"))

model = SentenceTransformer('all-MiniLM-L6-v2')
def embed_lyrics(df):
    embeddings = []
    for lyric in tqdm(df['cleaned_lyrics'].fillna(""), desc="Embedding lyrics"):
        emb = model.encode(lyric)
        embeddings.append(emb)
    return pd.DataFrame(embeddings)

train_emb = embed_lyrics(train_df)
valid_emb = embed_lyrics(valid_df)
test_emb  = embed_lyrics(test_df)

train_emb['song_id'] = train_df['song_id']
valid_emb['song_id'] = valid_df['song_id']
test_emb['song_id']  = test_df['song_id']

train_emb.to_csv(os.path.join(output_folder, "train_lyrics_emb.csv"), index=False)
valid_emb.to_csv(os.path.join(output_folder, "valid_lyrics_emb.csv"), index=False)
test_emb.to_csv(os.path.join(output_folder, "test_lyrics_emb.csv"), index=False)

print("Lyrics embeddings saved.")
