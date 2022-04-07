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
import collections
from social_django.models import UserSocialAuth
# from recommend.models import Userinfo
from django.contrib.auth.decorators import login_required

from django.views.generic import (ListView,
                                  DetailView,
                                  CreateView,
                                  DeleteView,
                                  UpdateView)
from . import models
from django.urls import reverse_lazy
from django.db.models import Q

#Tweepyによるデータ取得
#ライブラリのインポート
#Twitterの認証
consumer_key = 'uuG1jzBJAJ0C2SuSDAGsgvDlu'
consumer_secret = 'T7avTm0YbFvthT054mpSfmxpbzTDTjVNhLrSFgQL8lNv1d1tFS'
#access_token = '1377047808096628736-BD0rwABTVDqmgIq5RbhYNfgw5fRqUE'
#access_token_secret = 'Hp41WO3oTcZZfsqH9M4VIbHuZ55S4LYuhwlNaVuYyjdvP'

#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_token_secret)

#api = tweepy.API(auth)

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

file_name="params5.pkl"
vectorizer =None
with open(file_name, mode='rb') as f:
    parse=parse
    vectorizer = pickle.load(f)
    print("load vectorizer OK!!")

vecs = mmread("./tfidffile5.mtx")

df=pd.read_csv("lightasin6.csv")
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
   code_regex = re.compile('[!"#$%&\'\\\\()*+,./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。｡゜、､？！♪♡～☻⋆✧○〇✦｢｣｀＋￥％①②③⑨⑩④⑤⑥⑦⑧⑪⑫⑬⑭]')
   compare=re.sub("@[A-Za-z0-9_]+","", compare)
   compare=re.sub("#[A-Za-z0-9_\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]+","",compare)
    #メンション，ハッシュタグ,質問箱無効化
   compare=re.sub("●[A-Za-z0-9_\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]+","",compare)
   compare=re.sub("◎[A-Za-z0-9_\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]+","",compare)
   compare=re.sub("・[A-Za-z0-9_\u0020-\u007e\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]+","",compare)
    #URL無効化
   compare = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', compare)
   compare = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', compare)
   #桁区切りと数字の無効化
   compare = re.sub(r'(\d)([,.])(\d+)', r'\1\3', compare)
   compare = re.sub(r'\d+', '0', compare)
   compare = "".join(t for t in compare if n.search(t)) #一文字ずつre.searchで判定
   compare = compare.replace('こんな質問', '')
   compare = compare.replace('マシュマロ', '').replace('iriam', '').replace('投げ', '').replace('合お', '').replace('定期', '').replace('bot', '').replace('リプライ', '')
   compare = compare.replace('ツイート', '').replace('&amp;', '').replace('いいね', '')
   compare = re.sub(r'ww+', 'w', compare)
   compare = re.sub(r'笑笑+', '笑', compare)
   compare = re.sub(r'ーー+', 'ー', compare)
   compare = re.sub(r'--+', '-', compare)
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

def recommend_items(twid, hint, user):
    #demodataを入力とする"
    consumer_key = 'uuG1jzBJAJ0C2SuSDAGsgvDlu'
    consumer_secret = 'T7avTm0YbFvthT054mpSfmxpbzTDTjVNhLrSFgQL8lNv1d1tFS'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    print('TOKENS------------------------')
    print(user.tokens)
    auth.set_access_token(user.tokens['oauth_token'], user.tokens['oauth_token_secret'])
    api = tweepy.API(auth)

    twetlist=[tweet.text for tweet in tweepy.Cursor(api.user_timeline, id=twid).items(100) if (list(tweet.text)[:2]!=['R', 'T']) & (list(tweet.text)[0]!='@')]
    print(twetlist)
    rawtwet=''.join(twetlist)
    str_=reg_ct(rawtwet)+reg_ct(hint)*3 #定数を調整してキーワードの反映割合を変更
    print(str_)

    instr = parse(str_)
    print("instr=", instr )
    x= vectorizer.transform( [str_])

    #print( "x=",x)
    #Cosine類似度（cosine_similarity）の算出
    num_sim=cosine_similarity(x , vecs)
    num_sim=list(itertools.chain.from_iterable(num_sim))
    index = np.argmax( num_sim )
    topn_indices = np.argsort(num_sim)[::-1][:50]
    return topn_indices

def recommend_items_by_key(keyword):
    #demodataを入力とする"
    # twetlist=[tweet.text for tweet in tweepy.Cursor(api.user_timeline, id=twid).items(10) if (list(tweet.text)[:2]!=['R', 'T']) & (list(tweet.text)[0]!='@')]
    # rawtwet=''.join(twetlist)
    str_=keyword

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

def by_keyword(request):
    if request.method=="GET":
        return render(
            request,
            "recommend/by_key.html",
        )
    else:
        if request.POST["keyword"] != None:
            keyword=request.POST["keyword"]
            best_by_key=recommend_items_by_key(keyword)
        else:
            best_by_key='no'

        return render(
            request,
            "recommend/by_key.html",
            # {"best": bestbuy},
            {"best_by_key": best_by_key}
        )

#購入アイテムのレコメンド
def bestbuy(topn_indices):
    bestbuy =[df_h.iloc[int(i)] for i in topn_indices]
    return bestbuy

def displayurl(topn_indices):
    bestbuy =[list(df_v.iloc[int(i)].split(',')) for i in topn_indices]
    return bestbuy

def container(topn_indices,num):
    topn_indices=topn_indices[:30]
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

def raderchart(topn_indices):
    idto=pd.read_csv("idtocategory.csv")
    cat=pd.read_csv("categoryidx4.csv")
    tmplist=[]
    for i in range(len(topn_indices)):
        tmpval1=cat.iloc[topn_indices[i],0]
        if pd.isnull(tmpval1):
                tmpval1=-1
        tmpval=int(tmpval1)
        if tmpval!=-1:
                tmplist.append(idto[idto['id']==tmpval].loc[tmpval,"item"])
    c = collections.Counter(tmplist)
    return c

#homeページ表示用
@login_required
def home(request):
    if request.user.is_authenticated:
        user = UserSocialAuth.objects.get(user_id=request.user.id)
        userid = user.extra_data['access_token']['screen_name']
        uselog = models.Uselog.objects.filter(userid=userid).order_by('id').reverse()
    
    if request.method=="GET":
        return render(
            request,
            "recommend/home.html",
            {'user': user, 'uselog': uselog}
        )
    else:
        twid=request.POST["twid"]
        hint=request.POST["hint"]
        user = UserSocialAuth.objects.get(user_id=request.user.id)
        uselog_target = models.Uselog.objects.filter(targetid=twid).exclude(userid=userid).order_by('id').reverse()
        topn_indices=recommend_items(twid, hint, user)
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
        besturl17=container(topn_indices,17)
        besturl18=container(topn_indices,18)
        besturl19=container(topn_indices,19)
        besturl20=container(topn_indices,20)
        besturl21=container(topn_indices,21)
        besturl22=container(topn_indices,22)
        besturl23=container(topn_indices,23)
        besturl24=container(topn_indices,24)
        besturl25=container(topn_indices,25)
        besturl26=container(topn_indices,26)
        besturl27=container(topn_indices,27)
        besturl28=container(topn_indices,28)
        besturl29=container(topn_indices,29)
        rsn=reason(topn_indices)
        raderitem=raderchart(topn_indices)
        raderitem2 = {k: v for k, v in sorted(raderitem.items(), key=lambda item: item[1],reverse=True)} #大きい順にソート
        rader2=dict([(k, str(v) if type(v) is int else v) for k,v in raderitem2.items()]) #表示用にstr型へ
        print(type(bestbuy))
        print(bestbuy)
        import json
        items={}
        vals={}
        items["data0"] = list(rader2.keys())[0]
        items["data1"] = list(rader2.keys())[1]
        items["data2"] = list(rader2.keys())[2]
        vals["data0"] = int(list(rader2.values())[0])
        vals["data1"] = int(list(rader2.values())[1])
        vals["data2"] = int(list(rader2.values())[2])
        print((list(rader2.keys())[0]))
        return render(
            request,
            "recommend/home.html",
            {'items':json.dumps(items, ensure_ascii=False),"vals":vals,"twid":twid,"reason":rsn,"rader2":rader2,"user":user, "uselog":uselog, "uselog_target":uselog_target,"besturl0":besturl0,"besturl1":besturl1,"besturl2":besturl2,"besturl3":besturl3,"besturl4":besturl4,"besturl5":besturl5,"besturl6":besturl6,"besturl7":besturl7,"besturl8":besturl8,"besturl9":besturl9,"besturl10":besturl10,
            "besturl11":besturl11,"besturl12":besturl12,"besturl13":besturl13,"besturl14":besturl14,"besturl15":besturl15,"besturl16":besturl16,"besturl17":besturl17,"besturl18":besturl18,"besturl19":besturl19,"besturl20":besturl20,"besturl21":besturl21,"besturl22":besturl22,"besturl23":besturl23,"besturl24":besturl24,"besturl25":besturl25,
            "besturl26":besturl26,"besturl27":besturl27,"besturl28":besturl28,"besturl29":besturl29,},
        )

def toppage(request):
    if request.user.is_authenticated:
        user = UserSocialAuth.objects.get(user_id=request.user.id)
        userid = user.extra_data['access_token']['screen_name']
        uselog = models.Uselog.objects.filter(userid=userid).order_by('id').reverse()
        return render(
            request,
            "recommend/toppage.html",
            {'user': user, 'uselog': uselog}
            )
    else:
        return render(
            request, 
            'recommend/toppage.html'
            )


class see(ListView):
    model = models.Userinfo
    context_object_name = 'user_list'
    template_name = 'recommend/see.html'

class create(CreateView):
    model = models.Userinfo
    fields = ('name', 'token')
    template_name = 'recommend/create.html'
    success_url = reverse_lazy('recommend:see')


def postdata(request):
    if request.user.is_authenticated:
        user = UserSocialAuth.objects.get(user_id=request.user.id)

    if request.method == 'POST':
        from django.http import QueryDict

        # request.bodyに入っている。
        dic = QueryDict(request.body, encoding='utf-8')
        userid = user.extra_data['access_token']['screen_name']
        targetid = dic.get('targetid')
        clicked_item = dic.get('clicked_item')

        print('-----------CHECK==========')
        print(userid, targetid, clicked_item)

        uselog = models.Uselog.objects.create(
            userid = userid,
            targetid = targetid,
            clicked_item = clicked_item
            )
        uselog.save()

        # r1, r2 = do_something(v1, v2)

        # from json import dumps
        # ret = dumps({'k1': r1, 'k2': r2})
        return HttpResponse('ok', content_type='application/json')
    # else:
    #     return do_something_else()
