# -*- coding: utf-8 -*-
# %matplotlib inline
import sys
import pickle
import pandas as pd
import ffn
from dateutil import parser
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from flask import Flask, render_template, jsonify

from collections import namedtuple
ScoreSimNewsFormat = namedtuple('ScoreSimNewsFormat',
                     ['timemark', 'headline', 'contents', 'keycount', 'scorevec'])
ScoreNewNewsFormat = namedtuple('ScoreNewNewsFormat',
                     ['timemark', 'headline', 'contents', 'keywords', 'simnews_list', 'scorevec'])

app = Flask(__name__)

def plotpic(s):
  date_string=s[0]
  dt = parser.parse(date_string)
  dt_min = dt - datetime.timedelta(100)
  dt_max = dt + datetime.timedelta(100)
  df_min_string = dt_min.date().isoformat()
  df_max_string = dt_max.date().isoformat()
  prices=s[1]
  priceslice = prices[df_min_string:df_max_string]
  percent= priceslice.rebase()
  #print(percent.head())
  plots = percent.plot(figsize=(10,5))
  #print("okplots")

  priceshead = prices[:date_string].tail(2)
  pricestail = prices[date_string:].head(30)
  df_stt_string = priceshead.index.values[0]
  df_end_string = pricestail.index.values[-1]
  #dt_stt = dt - datetime.timedelta(2)
  #dt_end = dt + datetime.timedelta(30)
  #df_stt_string = dt_stt.date().isoformat()
  #df_end_string = dt_end.date().isoformat()
  plt.axvline(x=df_stt_string, linewidth=0.5, color='k')
  plt.axvline(x=df_end_string, linewidth=0.5, color='k')

  figorg = plots.get_figure()
  figorg.tight_layout()

  figfile = BytesIO()
  figorg.savefig(figfile, format='svg')
  figsvg = '<svg' + figfile.getvalue().split('<svg')[1]
  figsvg = unicode(figsvg, 'utf-8')
  return figsvg

@app.route("/")
def index():
  return render_template("HomePage.html", tickers=tickers, newslist=newslist, oldlists=oldlists, figlist=figlists[0])

@app.route("/<int:inew>",methods=['GET'])
def getFigure(inew):
  print "the fig list is ",inew
  try:
    #figlist=[]
    #for iold in range(len(oldlists[inew])):
    #  print("figstart")
    #  fig = plotpic((oldlists[inew][iold][0],tickers))
    #  print("figok")
    #  figlist.append(fig)
    #return figlist
    return jsonify(figlist=figlists[inew])
  except:
    return "No fig"
    #abort(404)
    #flash()

if __name__ == "__main__":
  tickers = pd.read_csv(open("./tickers"))
  tickers = ",".join([tickers.columns[i] for i in range(len(tickers.columns))])
  prices  = ffn.get(tickers, start="2005-01-01")
  newsdata= pickle.load(open("./graphinput.p","rb"))
  #newsdata=newsdata[:3]
  newslist=[]
  oldlists=[]
  for record in newsdata:
    newslist.append((record.timemark,record.headline,",".join(record.keywords),",".join(record.scorevec)))
    oldlist=[]
    for oldrecord in record.simnews_list:
      oldlist.append((oldrecord.timemark,oldrecord.headline,str(oldrecord.keycount),",".join(oldrecord.scorevec)))
    oldlists.append(oldlist)
  figlists=[]
  for inew in range(len(newslist)):
    print(inew)
    sys.stdout.flush()
    figlist=[]
    for iold in range(len(oldlists[inew])):
      figlist.append(plotpic((oldlists[inew][iold][0],prices)))
    figlists.append(figlist)

  app.run(debug = True,host= '0.0.0.0')