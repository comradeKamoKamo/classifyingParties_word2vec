#%%

"""MeCab Version"""

from logging import getLogger, StreamHandler, DEBUG, NullHandler, Formatter, INFO, basicConfig
import re
import os
from pathlib import Path

from lxml import etree
from gensim.models import Word2Vec
import MeCab
import mysql.connector

#%%

SCHEME = "politics"

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
def get_sentences_xml(path,cur,tagger,*,logger=None):
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
        if check_category(cur,page_id,scheme=SCHEME):
            logger.info("Wakati: {0}:{1}".format(page_id,doc.attrib["title"]))
            lines = doc.text.split("\n")
            for l in lines:
                sentences.append(get_wakati(l,tagger))
        else:
            logger.info("Ignore: {0}:{1}".format(page_id,doc.attrib["title"]))
    return sentences

def get_wakati(text,tagger):
    if len(text)==0:
        return text
    lines = text.split("\n")
    wakati = []
    for l in lines:
        # base_form
        for token in tagger.parse(l).split("\n"):
            t = token.split("\t")
            if len(t) < 3 :
                continue
            wakati.append(t[2])
    return wakati

def get_tagger():
    """remove the filter funcs"""
    m = MeCab.Tagger("-Ochasen")
    return m

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
    tagger = get_tagger()
    for subdir in wiki_dir.glob("*"):
        if subdir.is_dir():
            for xml in subdir.glob("*"):
                if xml.is_dir():
                    continue
                if model is None:
                    model = train_corpus(
                        get_sentences_xml(str(xml),cur,tagger,logger=logger))
                else:
                    model = train_corpus(
                        get_sentences_xml(str(xml),cur,tagger,logger=logger),model)
                model.save("make_corpus\politics.model")
                logger.info("Saved: {0}".format(str(xml)))

    close_connection(conn,cur)

