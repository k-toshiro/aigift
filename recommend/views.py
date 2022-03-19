from django.shortcuts import render
from django.http import HttpResponse
#tweepy
import tweepy
import MeCab
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import nltk
from nltk.corpus import wordnet
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import itertools
import pickle

#Tweepyによるデータ取得
#ライブラリのインポート
#Twitterの認証
consumer_key = 'uuG1jzBJAJ0C2SuSDAGsgvDlu'
consumer_secret = 'T7avTm0YbFvthT054mpSfmxpbzTDTjVNhLrSFgQL8lNv1d1tFS'
access_token = '1377047808096628736-BD0rwABTVDqmgIq5RbhYNfgw5fRqUE'
access_token_secret = 'Hp41WO3oTcZZfsqH9M4VIbHuZ55S4LYuhwlNaVuYyjdvP'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def parse(text):
   mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -O chasen')
   mecab.parse('')
   node = mecab.parseToNode(text)
   word_list = list()
   while node:
       word = node.surface
       word_type = node.feature.split(",")[0]

       if word_type in ["名詞", "動詞", "形容詞", "副詞"]:
           if word != "*":
               word_list.append(word)
               
       node = node.next
   return word_list

file_name="params2.pkl"
vectorizer =None
with open(file_name, mode='rb') as f:
    parse=parse
    vectorizer = pickle.load(f)
    print("load vectorizer OK!!")

vecs = mmread("./tfidffile2.mtx")

df=pd.read_csv("lightasin.csv")
df_h=df.iloc[:,0]
df_v=df.iloc[:,1]
#アイテムとurl別に
print(len(df_h))
print(df_h.iloc[0])

#ここまでファイルの読み込み

#入力列処理用
def lower_text(text):
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = lower_text(normalized_text)
    return normalized_text

def reg_ct(compare):
   n = re.compile(r"[\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]") #文字コードコンパイル
   code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。｡゜、､？！♪♡～☻⋆✧○〇✦｢｣｀＋￥％①②③⑨⑩④⑤⑥⑦⑧⑪⑫⑬⑭]')
   compare=re.sub("@[A-Za-z0-9_]+","", compare)
   compare=re.sub("#[A-Za-z0-9_\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]+","",compare)
    #URL無効化
   compare = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', compare)
   compare = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', compare)
   #桁区切りと数字の無効化
   compare = re.sub(r'(\d)([,.])(\d+)', r'\1\3', compare)
   compare = re.sub(r'\d+', '0', compare)
   compare = "".join(t for t in compare if n.search(t)) #一文字ずつre.searchで判定
   compare = code_regex.sub('', compare)
   compare = compare.replace('質問', '')
   compare = compare.replace('マシュマロ', '').replace('iriam', '').replace('投げ', '').replace('合お', '').replace('定期', '').replace('bot', '').replace('リプライ', '').replace('前日', '').replace('比', '').replace('自動', '')
   compare = compare.replace('&amp;', '').replace('\\', '').replace('\'', '').replace('amp;', '').replace('amp', '')
   compare = compare.replace('こと', '').replace('わたし', '').replace('私', '').replace('自分', '').replace('する', '').replace('そう', '').replace('いる', '').replace('とき', '').replace('ロシア', '').replace('ウクライナ', '')
   compare = compare.replace('さん', '').replace('0', '').replace('00', '').replace('in', '').replace('rt', '')
    #正規化
   compare=normalize(compare)
   return compare

   #レコメンドアイテムの根拠を表示
def show_result(feature_names, scores):
   result = dict()
   for token, weight in zip(feature_names, scores):
       result[token] = weight

   sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
   return sorted_result[:10]

def recommend_items(twid):
    #demodataを入力とする"
    twetlist=[tweet.text for tweet in tweepy.Cursor(api.user_timeline, id=twid).items(100) if (list(tweet.text)[:2]!=['R', 'T']) & (list(tweet.text)[0]!='@')]
    rawtwet=''.join(twetlist)
    str_=reg_ct(rawtwet)

    instr = parse(str_)
    print("instr=", instr )
    x= vectorizer.transform( [str_])

    #print( "x=",x)
    #Cosine類似度（cosine_similarity）の算出
    num_sim=cosine_similarity(x , vecs)
    num_sim=list(itertools.chain.from_iterable(num_sim))
    index = np.argmax( num_sim )
    topn_indices = np.argsort(num_sim)[::-1][:20]
    return topn_indices

#購入アイテムのレコメンド
def bestbuy(topn_indices):
    bestbuy =[df_h.iloc[int(i)] for i in topn_indices]
    return bestbuy

def displayurl(topn_indices):
    bestbuy =[list(df_v.iloc[int(i)].split(',')) for i in topn_indices]
    return bestbuy

def container(topn_indices,num):
    container=[]
    item = re.split(r'\',\'|","|", "|\', \'|\', "|", \'',df_h.iloc[int(topn_indices[num])])
    url=df_v.iloc[int(topn_indices[num])].split(',')
    for i in range(min(len(item),len(url))):
        container.append([item[i],url[i].replace(' ', '')])
    return container

#ここからアイテム選出のキーワードでトップコサイン類似度のものを表示する関数
def reason(topn_indices):
    reason=[]
    for i in range(5): #5つまで表示
        j=topn_indices[i]
        feature_names = vectorizer.get_feature_names()
        scores = vecs.getrow(j).toarray()[0]
        reason.append(show_result(feature_names, scores)[0][0])
    return reason

#homeページ表示用
def index(request):
    if request.method=="GET":
        return render(
            request,
            "recommend/home.html",
        )
    else:
        twid=request.POST["twid"]
        topn_indices=recommend_items(twid)
        print(topn_indices)
        besturl0=container(topn_indices,0)
        besturl1=container(topn_indices,1)
        besturl2=container(topn_indices,2)
        besturl3=container(topn_indices,3)
        besturl4=container(topn_indices,4)
        besturl5=container(topn_indices,5)
        besturl6=container(topn_indices,6)
        besturl7=container(topn_indices,7)
        besturl8=container(topn_indices,8)
        besturl9=container(topn_indices,9)
        besturl10=container(topn_indices,10)
        besturl11=container(topn_indices,11)
        besturl12=container(topn_indices,12)
        besturl13=container(topn_indices,13)
        besturl14=container(topn_indices,14)
        besturl15=container(topn_indices,15)
        besturl16=container(topn_indices,16)
        rsn=reason(topn_indices)
        print(type(bestbuy))
        print(bestbuy)
        return render(
            request,
            "recommend/home.html",
            {"reason":rsn,"besturl0":besturl0,"besturl1":besturl1,"besturl2":besturl2,"besturl3":besturl3,"besturl4":besturl4,"besturl5":besturl5,"besturl6":besturl6,"besturl7":besturl7,"besturl8":besturl8,"besturl9":besturl9,"besturl10":besturl10,
            "besturl11":besturl11,"besturl12":besturl12,"besturl13":besturl13,"besturl1":besturl14,"besturl15":besturl15,"besturl16":besturl16,},
        )
