import xml.etree.ElementTree as ET
from pathlib import Path
from logging import getLogger, INFO, StreamHandler

import MeCab

import xml_to_tweet

def main():
    logger = getLogger()
    handler = StreamHandler()
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.setLevel(INFO)
    logger.propagate = False

    for xml_dir in Path("get_tweets").glob("*"):
        if xml_dir.is_dir():
            logger.info("target: {0}".format(xml_dir.name))
            path = str((xml_dir / Path("tweets.xml")))
            if not Path(path).exists():
                continue
            x2t = xml_to_tweet.XmlToTweets(path,logger=logger)
            tweets = x2t.xml_to_tweets(exclude_wakati=True)

            m = MeCab.Tagger("-Ochasen")
            n_tweets = []
            for tweet in tweets:
                wakati = ""
                for l in tweet["text"].split("\n"):
                    for token in m.parse(l).split("\n"):
                        t = token.split("\t")
                        if len(t) < 3:
                            continue
                        wakati += t[2] + " "
                    wakati = wakati[:-1] + "\n"
                tweet["wakati"] = wakati[:-1]
                n_tweets.append(tweet)
            x2t.tweets_to_xml(str((xml_dir / Path("tweets.xml"))),n_tweets)
            
if __name__=="__main__":
    main()
