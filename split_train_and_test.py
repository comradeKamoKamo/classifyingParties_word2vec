#%%
from pathlib import Path
import pickle

import numpy as np

from get_tweets.xml_to_tweet import XmlToTweets

def main():
    data_index = []
    train_indexes = dict()
    test_indexes = dict()
    for p in [ v for v in Path("get_tweets").glob("*") if v.is_dir()]:
        xmlpath = p / Path("tweets.xml")
        if xmlpath.exists():
            x2t = XmlToTweets(str(xmlpath))
            tweets = x2t.xml_to_tweets(exclude_text=True,exclude_wakati=True)
            ids = [v["id"] for v in tweets]
            for tid in ids: 
                data_index.append(
                    (tid, p.name)
                )
            train_indexes[p.name] = []
            test_indexes[p.name] = []
    
    test_nums = [int(v) for v in np.random.rand(int(len(data_index)*0.3)) * len(data_index)]
    i = 0
    for data in data_index:
        if i in test_nums:
            test_indexes[data[1]].append(data[0])
        else:
            train_indexes[data[1]].append(data[0])
        i+=1
    split_info = dict()
    split_info["train"] = train_indexes
    split_info["test"] = test_indexes
    with Path("split_data.pickle").open("wb") as f:
        pickle.dump(split_info, f)
    return split_info

if __name__=="__main__":
    main()