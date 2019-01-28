#%%

"""MeCab Version"""

from logging import getLogger, StreamHandler, DEBUG, NullHandler, Formatter, INFO, basicConfig
import re
import os
from pathlib import Path

from lxml import etree
from gensim.models import Word2Vec, word2vec
import MeCab
import mysql.connector

#%%

SCHEME = "japan_politics"
page_count = 0
target_page_count = 0

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
def train_corpus(file):
    model = Word2Vec(word2vec.LineSentence(file),
                sg=0,
                size=200,
                hs=0,
                negative=5,
                sample=1e-3)
    return model

#%%
def add_sentences_xml(path,cur,tagger,file,*,logger=None):
    __logger = getLogger(__name__)
    __logger.addHandler(NullHandler())
    logger = logger or __logger
    logger.info("open: {0}".format(path))
    with open(path,"r",encoding="utf-8") as f:
        xml_string = "<docs>" + f.read() + "</docs>"
    for tags in re.findall(r"<.+?>",xml_string):
        if re.match(r"</??((doc)|(docs))+?(\s.+?)??>",tags) is None:
            xml_string = xml_string.replace(tags,"")
            logger.debug("RE:{0}".format(tags))
    root = etree.XML(xml_string,etree.XMLParser(recover=True))
    for doc in root:
        page_id = doc.attrib["id"]
        global page_count, target_page_count
        page_count += 1
        if check_category(cur,page_id,scheme=SCHEME):
            target_page_count += 1
            logger.info("Wakati: {0}:{1}".format(page_id,doc.attrib["title"]))
            lines = doc.text.split("\n")
            for l in lines:
                wakati_line = ""
                for token in get_wakati(l,tagger):
                    wakati_line += token + " "
                file.write(wakati_line[:-1] + "\n")
        else:
            logger.debug("Ignore: {0}:{1}".format(page_id,doc.attrib["title"]))
    return file

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

    CORPUS_PATH = "make_corpus/jawiki_japan_politics_corpus.txt"

    conn, cur = setup_connector()
    wiki_dir = Path("D:\wiki")
    tagger = get_tagger()
    if Path(CORPUS_PATH).exists():
        logger.error("{0} aleredy exists!".format(CORPUS_PATH))
        exit(1)
    with Path(CORPUS_PATH).open("a",encoding="utf-8") as corpus_file:
        for subdir in wiki_dir.glob("*"):
            if subdir.is_dir():
                for xml in subdir.glob("*"):
                    if xml.is_dir():
                        continue  
                    add_sentences_xml(str(xml),cur,tagger,corpus_file,logger=logger)
                    logger.info("target: {0} pages / all: {1} pages".format(target_page_count,page_count))
    model = train_corpus(CORPUS_PATH)
    model.save("make_corpus\jawiki_japan_politics.model")
    logger.info("target: {0} pages / all: {1} pages".format(target_page_count,page_count))
    logger.info("finish!")

    close_connection(conn,cur)

