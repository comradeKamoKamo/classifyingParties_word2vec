#%%
import logging

# BERT 読み込み

from keras_bert import load_trained_model_from_checkpoint

config_path = 'bert-wiki-ja/bert_config.json'
checkpoint_path = 'bert-wiki-ja/model.ckpt-1400000' # 拡張子.index/.meta/.data~省く

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
bert.summary()

# 特徴量取得関数
import sentencepiece as spm
import numpy as np

maxlen = 512
bert_dim = 768

sp = spm.SentencePieceProcessor()
sp.Load('bert-wiki-ja/wiki-ja.model')

def get_feature(text):
    common_seg_input = np.zeros( (1, maxlen), dtype=np.float32)
    indices = np.zeros((1, maxlen), dtype=np.float32)

    tokens = []
    tokens.append("[CLS]") # 文頭記号
    tokens.extend(sp.encode_as_pieces(text))
    tokens.append("[SEP]") # 番兵

    for t, token in enumerate(tokens):
        try:
            indices[0, t] = sp.piece_to_id(token)
        except:
            logging.warn(f"{token} is unknown.")
            indices[0, t] = sp.piece_to_id('<unk>')
    vector = bert.predict([indices, common_seg_input])[0]
    
    return vector


#%%
# データロード
from get_tweets import xml_to_tweet
from pathlib import Path
def make_data(politician_name):
    x2t = xml_to_tweet.XmlToTweets(f"get_tweets/{politician_name}/tweets.xml")
    tweets = x2t.xml_to_tweets(exclude_wakati=True)
    vetcors = [get_feature(t["text"]) for t in tweets]
    vetcors = np.array(vetcors)
    np.save(f"features/{politician_name}.npy", vetcors)
    return

for p in Path("get_tweets/").glob("*/"):
    if (not (p / "tweets.xml").exists()) : continue
    print(f"{str(p)} ... ")
    make_data(str(p.name))


