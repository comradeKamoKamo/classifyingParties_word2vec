#%%
import os
import tweepy
from janome import tokenizer, tokenfilter, analyzer
from pathlib import Path
import xml.etree.ElementTree as ET
import time

#%%
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

#%%
leaders = [
    "AbeShinzo",        #自民党 安倍晋三
    "edanoyukio0531",   #立憲民主党 枝野幸男
    "tamakiyuichiro",   #国民民主党 玉木雄一郎
    "gogoichiro",       #日本維新の会 松井一郎
    "shiikazuo"]        #日本共産党 志位和夫
for leader in leaders:
    Path("get_tweets\\{0}".format(leader)).mkdir(exist_ok=True)

#%%
api_key = os.environ.get("API_KEY")
api_secret = os.environ.get("API_SECRET")
access_token = os.environ.get("ACCESS_TOKEN")
access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")
auth = tweepy.OAuthHandler(api_key,api_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

#%%
"""
t = tokenizer.Tokenizer()
char_filters = [analyzer.UnicodeNormalizeCharFilter(),
                analyzer.RegexReplaceCharFilter(r"@[a-zA-Z\d]*",""),
                analyzer.RegexReplaceCharFilter(r"[#$]",""),
                analyzer.RegexReplaceCharFilter(r"https?:[a-zA-Z\d/\.]*","")]
token_filters = [tokenfilter.LowerCaseFilter()]
t_analyzer = analyzer.Analyzer(char_filters,t,token_filters)

def get_wakati(text):
    if len(text)==0:
        return text
    wakati_text = ""
    for token in t_analyzer.analyze(text):
        wakati_text += token.base_form + " "
    return wakati_text[0:-1]
"""
#%%
def get_and_save_tweets(screen_name):
    logger.info(screen_name)
    with Path("get_tweets/{0}/tweets.xml".format(screen_name)).open("a",encoding="utf-8") as f:
        c = tweepy.Cursor(api.user_timeline,screen_name=screen_name,exclude_replies=True,tweet_mode="extended").items()
        f.write("<tweets>")
        i = 0
        while True:
            try:
                tweet = c.next()
                if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
                    logger.info("GET TWEET ID: {0}".format(tweet.id))
                    tweetElement = ET.Element("tweet",id=str(tweet.id))
                    text = get_display_text(tweet)
                    textElement = ET.SubElement(tweetElement,"text")
                    textElement.text = text
                    """
                    wakati_text = get_wakati(text)
                    if len(wakati_text)==0 :
                        continue
                    else:
                        i = i + 1
                        if i > 500:
                            break
                    wakatiElement = ET.SubElement(tweetElement,"wakati")
                    wakatiElement.text=wakati_text
                    """
                    tree = ET.ElementTree(tweetElement)
                    tree.write(f,encoding="unicode",xml_declaration=False)
                    f.write("</tweets>")
            except tweepy.TweepError:
                time.sleep(60 * 15)
                continue
            except StopIteration:
                logger.warning("The number of tweets is low of 500.")
                break
                
def get_display_text(tweet):
    s , e = tuple(tweet.display_text_range)
    return tweet.full_text[s:e]

#%%
for leader in leaders:
    get_and_save_tweets(leader)