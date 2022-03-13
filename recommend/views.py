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
   mecab = MeCab.Tagger('-Owakati')
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

file_name="params.pkl"
vectorizer =None
with open(file_name, mode='rb') as f:
    parse=parse
    vectorizer = pickle.load(f)
    print("load vectorizer OK!!")

vecs = mmread("./tfidffile.mtx")

df_h=pd.read_csv("light.csv")
print(len(df_h))
print(df_h.iloc[0])

#ここまでファイルの読み込み
def recommend_items(twid):
    #demodataを入力とする"
    twetlist=[tweet.text for tweet in tweepy.Cursor(api.user_timeline, id=twid).items(10) if (list(tweet.text)[:2]!=['R', 'T']) & (list(tweet.text)[0]!='@')]
    rawtwet=''.join(twetlist)
    str_=rawtwet

    instr = parse(str_)
    print("instr=", instr )
    x= vectorizer.transform( [str_])

    #print( "x=",x)
    #Cosine類似度（cosine_similarity）の算出
    num_sim=cosine_similarity(x , vecs)
    num_sim=list(itertools.chain.from_iterable(num_sim))
    print(num_sim)
    index = np.argmax( num_sim )

    topn_indices = np.argsort(num_sim)[::-1][:10]
    bestbuy =[df_h.iloc[int(i)] for i in topn_indices]
    return bestbuy

#homeページ表示用
def index(request):
    if request.method=="GET":
        return render(
            request,
            "recommend/home.html",
        )
    else:
        twid=request.POST["twid"]
        bestbuy=recommend_items(twid)
        print(type(bestbuy))
        print(bestbuy)
        return render(
            request,
            "recommend/home.html",
            {"best": bestbuy},
        )
