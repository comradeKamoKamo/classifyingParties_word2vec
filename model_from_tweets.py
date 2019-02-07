#%%

"""MeCab Version"""

from logging import getLogger, StreamHandler, DEBUG, NullHandler, Formatter, INFO, basicConfig
import re
import os
from pathlib import Path
import pickle

from gensim.models import Word2Vec, word2vec
import MeCab

from get_tweets.xml_to_tweet import XmlToTweets

#%%
def train_corpus(file):
    model = Word2Vec(word2vec.LineSentence(file),
                sg=0,
                size=200,
                hs=0,
                negative=5,
                sample=1e-3)
    return model

def is_train_data(tweet_id, owner, split_info):
    return len(
        [v for v in split_info["test"][owner] if v==tweet_id]
    ) == 0


#%% 
if __name__=="__main__":
    logger = getLogger()
    logger.setLevel(INFO)
    handler = NullHandler()
    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    CORPUS_PATH = "make_corpus/tweet_corpus.txt"

    with Path("split_data.pickle").open("rb") as f:
        split_info = pickle.load(f)

    if Path(CORPUS_PATH).exists():
        logger.error("{0} aleredy exists!".format(CORPUS_PATH))
        exit(1)
    with Path(CORPUS_PATH).open("a",encoding="utf-8") as corpus_file:
        for p in [ v for v in Path("get_tweets").glob("*") if v.is_dir()]:
            xmlpath = p / Path("tweets.xml")
            if xmlpath.exists():
                logger.info(p.name)
                x2t = XmlToTweets(str(xmlpath))
                tweets = x2t.xml_to_tweets(exclude_text=True)
                for tweet in tweets:
                    if is_train_data(tweet["id"], p.name, split_info):
                        corpus_file.write(tweet["wakati"] + "\n")
    model = train_corpus(CORPUS_PATH)
    model.save("make_corpus/tweet.model")
    logger.info("finish!")
