#%%
from logging import getLogger , NullHandler
import xml.etree.ElementTree as ET

#%%
class XmlToTweets:

    
    def __init__(self,xmlfilepath,*,logger=None):
        __logger = getLogger(__name__)
        __logger.addHandler(NullHandler())
        __logger.propagate = False
        logger = logger or __logger
        try:
            with open(xmlfilepath,"r",encoding="utf-8") as f:
                self.XmlString = f.read()
        except FileNotFoundError:
            logger.error("FileNotFonundError{0}".format(xmlfilepath))
            raise
    
    def tweets_to_xml(self,path,tweets):
        root = ET.Element("tweets")
        for tweet in tweets:
            tweetElement = ET.SubElement(root,"tweet")
            tweetElement.set("id",tweet["id"])
            textElement = ET.SubElement(tweetElement,"text")
            textElement.text = tweet["text"]
            wakatiElement = ET.SubElement(tweetElement,"wakati")
            wakatiElement.text = tweet["wakati"]
        with open(path,"w",encoding="utf-8") as f:
            et = ET.ElementTree(root)
            et.write(f,encoding="unicode")

    def xml_to_tweets(self,limit=0,*,exclude_wakati=False,exclude_text=False,logger=None):
        __logger = getLogger(__name__)
        __logger.addHandler(NullHandler())
        __logger.propagate = False
        logger = logger or __logger
        try:
            root = ET.fromstring(self.XmlString)
        except ET.ParseError as e:
            logger.error(e.args)

        tweets = []
        c = 0
        for element in root.getchildren():
            tweet = dict()
            tweet["id"] = element.items()[0][1]
            logger.info("Now target: {0}".format(tweet["id"]))
            for subelement in element.getchildren():
                if subelement.tag == "wakati" and not exclude_wakati:
                    tweet["wakati"] = subelement.text
                elif subelement.tag == "text" and not exclude_text:
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
        