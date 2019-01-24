#%%
from pathlib import Path

from gensim.models import Word2Vec
import numpy as np

from get_tweets.xml_to_tweet import XmlToTweets

#%%
def get_tweet_vector(words,model):
    vector = np.zeros(200,dtype=float)
    for w in words:
        try:
            vector += model.wv[w]
        except KeyError:
            pass
    vector /= len(words)
    return vector

#%%
if __name__ == "__main__" :

    model = Word2Vec.load(r"make_corpus/jawiki.model")
    Path("data").mkdir(exist_ok=True)

    for pd in [d for d in Path("get_tweets").glob("*") if d.is_dir()]:
        path = pd / Path("tweets.xml")
        if path.exists():
            x2t = XmlToTweets(str(path))
            tweets = x2t.xml_to_tweets(exclude_text=True)
            vectors = []
            for t in tweets:
                words = t["wakati"].replace("\n"," ").split(" ")
                vector = get_tweet_vector(words,model)
                vectors.append(vector)
            np.save("data/{0}.npy".format(pd.name),vectors)

