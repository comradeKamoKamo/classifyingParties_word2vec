#%%
from pathlib import Path 
import pickle
from logging import getLogger, StreamHandler, INFO, Formatter, NullHandler

import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from get_tweets.xml_to_tweet import XmlToTweets 

#%%
def add_dataset(dataset, wakati_text):
    doc = []
    for line in wakati_text.split("\n"):
        doc.extend(line.split(" "))
    dataset.append(doc)
    return dataset

def is_train_data(tweet_id, owner, split_info):
    return len(
        [v for v in split_info["test"][owner] if v==tweet_id]
    ) == 0

#%%
if __name__=="__main__" :

    logger = getLogger()
    logger.setLevel(INFO)
    handler = NullHandler()
    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    dataset = []
    with Path("split_data.pickle").open("rb") as f:
        split_info = pickle.load(f)

    for p in [ v for v in Path("get_tweets").glob("*") if v.is_dir()]:
        xmlpath = p / Path("tweets.xml")
        if xmlpath.exists():
            logger.info(p.name)
            x2t = XmlToTweets(str(xmlpath))
            tweets = x2t.xml_to_tweets(exclude_text=True)
            for tweet in tweets:
                if is_train_data(tweet["id"], p.name, split_info):
                    add_dataset(dataset, tweet["wakati"])

    dct = Dictionary(dataset)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
    model = TfidfModel(corpus,smartirs="ntn")

    model.save("tfidf.model")
    dct.save("dct.model")

   