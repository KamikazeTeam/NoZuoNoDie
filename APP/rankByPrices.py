import nltk
from nltk.corpus import wordnet as wn
import collections
from collections import namedtuple
import pickle
import zipfile
from scipy import spatial
import pandas as pd
import pandas_datareader.data as web
import time

SimNewsFormat = namedtuple('SimNewsFormat',
                ['timemark', 'headline', 'contents', 'keycount'])
NewNewsFormat = namedtuple('NewNewsFormat',
                ['timemark', 'headline', 'contents', 'keywords', 'simnews_list'])
ScoreSimNewsFormat = namedtuple('ScoreSimNewsFormat',
                     ['timemark', 'headline', 'contents', 'keycount', 'scorevec'])
ScoreNewNewsFormat = namedtuple('ScoreNewNewsFormat',
                     ['timemark', 'headline', 'contents', 'keywords', 'simnews_list', 'scorevec'])
###################################################################################################
def getSimNewsScore(simnews_list,prices,goback,gofoward):
  simnews_score_list=[]
  for simnews in simnews_list:
    scorevec=[]
    for price in prices:
      try:
        sttprice=price["Close"][:simnews.timemark].tail(goback)[0]
        endprice=price["Close"][simnews.timemark:].head(gofoward)[-1]
        scorevec.append("%.2f" % float(100.0*(endprice-sttprice)/endprice))
      except:
        scorevec.append("%.2f" % float(100.0*0.0))
        #print("Holiday!")
    if len(prices)>1:#add difference with N225
      scorevec=["%.2f" % (float(scorevec[1])-float(scorevec[0]))]+scorevec
    simnews_score_list.append(ScoreSimNewsFormat(simnews.timemark,
                                                 simnews.headline,
                                                 simnews.contents,
                                                 simnews.keycount,
                                                 scorevec))
  return simnews_score_list
def getScorebyPrices(newnews_sim_list,prices,goback,gofoward):
  start_time=time.time()
  newnews_sim_score_list=[]
  for newnews_sim in newnews_sim_list:
    simnews_score_list = getSimNewsScore(newnews_sim.simnews_list,prices,goback,gofoward)
    simnews_score_list = sorted(simnews_score_list,key=lambda x: float(x.scorevec[0]),reverse=True)#
    scorevec = []
    dimofvec = len(prices)
    if len(prices)>1: dimofvec+=1#add difference with N225
    for idim in range(dimofvec):
      score = 0.0
      addedtimemark = []
      for simnews_score in simnews_score_list:
        if simnews_score.timemark in addedtimemark: continue
        score += float(simnews_score.scorevec[idim])
        addedtimemark.append(simnews_score.timemark)
      if len(addedtimemark)!=0:
        score/=len(addedtimemark)
      scorevec.append("%.2f" % score)
    newnews_sim_score_list.append(ScoreNewNewsFormat(newnews_sim.timemark,
                                                     newnews_sim.headline,
                                                     newnews_sim.contents,
                                                     newnews_sim.keywords,
                                                     simnews_score_list,
                                                     scorevec))
  end_time=time.time()-start_time
  print("cal score: %.6s" % end_time)
  return newnews_sim_score_list
###################################################################################################
def getPrices(tickersfile, sourcesite, startdate):
  tickers = pd.read_csv(open(tickersfile))
  prices=[]
  for ticker in tickers:
    price=web.DataReader(ticker, sourcesite, start=startdate)
    prices.append(price)
  return prices
###################################################################################################
if __name__ == "__main__":
  newnews_sim_list=pickle.load(open("./newnews_sim_list.p","rb"))
  tickersfile="./tickers"
  sourcesite ="yahoo"
  startdate  ="2005-01-01"
  prices  =getPrices(tickersfile, sourcesite, startdate)
  goback  =2
  gofoward=30
  newnews_sim_score_list=getScorebyPrices(newnews_sim_list,prices,goback,gofoward)
  newnews_sim_score_list=sorted(newnews_sim_score_list,key=lambda x: abs(float(x.scorevec[0])),reverse=True)#abs

  pickle.dump(newnews_sim_score_list,open("./graphinput.p","wb"), protocol=2)

  fr = open("./simresult","w")
  for idx,newnews_sim_score in enumerate(newnews_sim_score_list):
    print("%d" % idx,end="|",file=fr)
    print(newnews_sim_score.scorevec,end="|",file=fr)
    print(newnews_sim_score.timemark,end="|",file=fr)
    print(newnews_sim_score.headline,file=fr,flush=True)
  for idx,newnews_sim_score in enumerate(newnews_sim_score_list):
    print("",file=fr)
    print("%d" % idx,end="|",file=fr)
    print(newnews_sim_score.scorevec,end="|",file=fr)
    print(newnews_sim_score.timemark,end="|",file=fr)
    print(newnews_sim_score.headline,file=fr,flush=True)
    print(newnews_sim_score.keywords,file=fr)
    print("----------------------------------------------",file=fr,flush=True)
    for simnews in newnews_sim_score.simnews_list:
      print(simnews.scorevec,end="|",file=fr)
      print(simnews.keycount,end="|",file=fr)
      print(simnews.timemark,end="|",file=fr)
      print(simnews.headline,file=fr,flush=True)

  for idx,newnews_sim_score in enumerate(newnews_sim_score_list):
    print("")
    print("%d" % idx,end="|")
    print(newnews_sim_score.scorevec,end="|")
    print(newnews_sim_score.timemark,end="|")
    print(newnews_sim_score.headline,flush=True)
    print(newnews_sim_score.keywords)
    print("----------------------------------------------",flush=True)
    for simnews in newnews_sim_score.simnews_list:
      print(simnews.scorevec,end="|")
      print(simnews.keycount,end="|")
      print(simnews.timemark,end="|")
      print(simnews.headline,flush=True)
  for idx,newnews_sim_score in enumerate(newnews_sim_score_list):
    print("%d" % idx,end="|")
    print(newnews_sim_score.scorevec,end="|")
    print(newnews_sim_score.timemark,end="|")
    print(newnews_sim_score.headline,flush=True)


