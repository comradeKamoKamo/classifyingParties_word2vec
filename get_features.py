#%%
from pathlib import Path
import pickle

from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np

from get_tweets.xml_to_tweet import XmlToTweets

#%%
def get_tweet_vector(words,model,words_tfidf):
    vector = np.zeros(200,dtype=float)
    for w, t in zip(words, words_tfidf):
        try:
            vector += model.wv[w] * t
        except KeyError:
            pass
    vector /= len(words)
    return vector

def get_tfidf(words,tfidf_model,dct):
    corpus = dct.doc2bow(words)
    words_tfidf = tfidf_model[corpus]
    tfidf = []
    for doc in words_tfidf:
        tfidf.append(doc[1])
    return tfidf

#%%
if __name__ == "__main__" :

    model = Word2Vec.load(r"make_corpus/jawiki.model")
    tfidf_model = TfidfModel.load(r"tfidf.model")
    dct = Dictionary.load(r"dct.model")
    Path("data").mkdir(exist_ok=True)

    for pd in [d for d in Path("get_tweets").glob("*") if d.is_dir()]:
        path = pd / Path("tweets.xml")
        if path.exists():
            x2t = XmlToTweets(str(path))
            tweets = x2t.xml_to_tweets(exclude_text=True)
            vectors = []
            for t in tweets:
                words = t["wakati"].replace("\n"," ").split(" ")
                words_tfidf = get_tfidf(words, tfidf_model, dct)
                vector = get_tweet_vector(words,model,words_tfidf)
                vectors.append((t["id"], pd.name, vector))
            with Path("politicians/{0}.pickle".format(pd.name)).open("wb") as f:
                pickle.dump(vectors,f)

