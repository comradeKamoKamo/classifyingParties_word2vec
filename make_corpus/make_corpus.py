#%%
from logging import getLogger, StreamHandler, DEBUG, NullHandler, Formatter, INFO
import re

from lxml import etree
from gensim.models import Word2Vec
from janome import tokenizer, analyzer, tokenfilter
import numpy as np


#%%
def check_category(page_id):
    #stab
    return True

#%%
def train_corpus(sentences,model=None):
    if model is None:
            model = Word2Vec(sentences=sentences,
                        sg=1,
                        size=200,
                        hs=0,
                        min_count=0,
                        negative=3)
    else:
        model.train(sentences)
    return model

#%%
def get_sentences_xml(path,*,logger=None):
    __logger = getLogger(__name__)
    __logger.addHandler(NullHandler())
    logger = logger or __logger
    logger.info("open: {0}".format(path))
    with open(path,"r",encoding="utf-8") as f:
        xml_string = "<docs>" + f.read() + "</docs>"
    xml_string = re.sub(r"</??[^(doc)(docs)]+?(\s.+?)??>","",xml_string)
    root = etree.XML(xml_string,etree.XMLParser(recover=True))
    sentences = []
    t_analyzer = get_analyzer()
    for doc in root:
        page_id = doc.attrib["id"]
        if check_category(page_id):
            logger.info("Now: {0}:{1}".format(page_id,doc.attrib["title"]))
            lines = doc.text.split("\n")
            for l in lines:
                logger.debug(l)
                sentences.append(get_wakati(l,t_analyzer))
    return sentences

def get_wakati(text,t_analyzer):
    if len(text)==0:
        return text
    wakati = []
    for token in t_analyzer.analyze(text):
        wakati.append(token.base_form)
    return wakati

def get_analyzer():
    t = tokenizer.Tokenizer()
    char_filters = [analyzer.UnicodeNormalizeCharFilter(),
                analyzer.RegexReplaceCharFilter(r"@[a-zA-Z\d]*",""),
                analyzer.RegexReplaceCharFilter(r"[#$]",""),
                analyzer.RegexReplaceCharFilter(r"https?:[a-zA-Z\d/\.]*","")]
    token_filters = [tokenfilter.LowerCaseFilter()]
    return analyzer.Analyzer(char_filters,t,token_filters)

#%%
if __name__=="__main__":
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler = StreamHandler()
    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    model = train_corpus(get_sentences_xml("E:\\wiki\\AA\\wiki_00"))
    model.save("debug.bin")