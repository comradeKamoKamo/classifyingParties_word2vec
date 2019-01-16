#%%
from logging import getLogger, StreamHandler, DEBUG, NullHandler, Formatter, INFO, basicConfig
import re
import os
from pathlib import Path

from lxml import etree
from gensim.models import Word2Vec
from janome import tokenizer, analyzer, tokenfilter
import mysql.connector

#%%
def setup_connector():
    username = os.environ.get("MYSQL_USERNAME")
    password = os.environ.get("MYSQL_PASSWORD")
    conn = mysql.connector.connect(user=username, password=password, host="localhost")
    cur = conn.cursor()
    return conn, cur

def close_connection(conn,cur):
    cur.close()
    conn.close()

def check_category(cur, page_id,*,scheme):
        cur.execute("""
        select exists ( select category from wiki_temp.{0} where category in 
        (select cl_to from `jawiki-page`.categorylinks where cl_from = {1}) );
       """.format(scheme,page_id))
        result = cur.fetchall()
        return result[0][0]

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
def get_sentences_xml(path,cur,t_analyzer,*,logger=None):
    __logger = getLogger(__name__)
    __logger.addHandler(NullHandler())
    logger = logger or __logger
    logger.info("open: {0}".format(path))
    with open(path,"r",encoding="utf-8") as f:
        xml_string = "<docs>" + f.read() + "</docs>"
    xml_string = re.sub(r"</??[^(doc)(docs)]+?(\s.+?)??>","",xml_string)
    root = etree.XML(xml_string,etree.XMLParser(recover=True))
    sentences = []
    for doc in root:
        page_id = doc.attrib["id"]
        logger.debug("{0} : {1}".format(check_category(cur,page_id,scheme="politics"),doc.attrib["title"]))
        if check_category(cur,page_id,scheme="politics"):
            logger.info("Now: {0}:{1}".format(page_id,doc.attrib["title"]))
            lines = doc.text.split("\n")
            for l in lines:
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
    logger = getLogger()
    logger.setLevel(INFO)
    handler = StreamHandler()
    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    conn, cur = setup_connector()
    wiki_dir = Path("E:\wiki")
    model = None
    t_analyzer = get_analyzer()
    for subdir in wiki_dir.glob("*"):
        if subdir.is_dir():
            for xml in subdir.glob("*"):
                if xml.is_dir():
                    continue
                if model is None:
                    model = train_corpus(
                        get_sentences_xml(str(xml),cur,t_analyzer,logger=logger))
                else:
                    model = train_corpus(
                        get_sentences_xml(str(xml),cur,t_analyzer,logger=logger),model)
                model.save("make_corpus\politics.model")
                
    #model = train_corpus(get_sentences_xml("E:\\wiki\\AA\\wiki_00",cur,logger=logger))
    #model.save("debug2.bin")
    close_connection(conn,cur)
