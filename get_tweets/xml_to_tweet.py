#%%
from logging import getLogger, NullHandler , INFO
import xml.etree.ElementTree as ET

#%%
class XmlToTweets:

    Logger = getLogger(__name__)

    def __init__(self,xmlfilepath,*,logger=None):
        if logger is None:
            Logger.setLevel(INFO)
            handler = NullHandler
            Logger.addHandler(handler)
            Logger.propagate = False
        else:
            Logger = logger
        
        try:
            with open(xmlfilepath,"r",encoding="utf-8") as f:
                self.XmlString = f.read()
        except FileNotFoundError:
            Logger.error("FileNotFonundError{0}".format(xmlfilepath))
            raise
        
    def xml_to_tweets(self,limit=0):
        try:
            root = ET.fromstring(self.XmlString)
        except ET.ParseError as e:
            Logger.error(e.args)

        tweets = []
        c = 0
        for element in root.getchildren():
            tweet = dict()
            tweet["id"] = element.items()[0][1]
            logger.info("Now target: {0}".format(tweet["id"]))
            for subelement in element.getchildren():
                if subelement.tag == "wakati":
                    tweet["wakati"] = subelement.text
                elif subelement.tag == "text":
                    tweet["text"]  = subelement.text
            tweets.append(tweet)
            c = c + 1
            if limit != 0 and c >= limit:
                break

        return tweets

#%%
if __name__ == "__main__":
    from logging import getLogger, StreamHandler, DEBUG
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    X2T = XmlToTweets("get_tweets/AbeShinzo/tweets.xml",logger=logger)
    l = X2T.xml_to_tweets(2)
    print(l[0])
        