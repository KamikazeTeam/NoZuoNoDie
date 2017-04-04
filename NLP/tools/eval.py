#!/usr/bin/python3
#set fileencoding=utf8
from __future__ import print_function
from __future__ import division
import numpy as np

import zipfile
import collections
from scipy import spatial
import nltk
from nltk.corpus import wordnet as wn
from collections import namedtuple
import time
import pickle

OrgNewsFormat = namedtuple('OrgNewsFormat',
                ['timemark', 'headline', 'contents'])
SimNewsFormat = namedtuple('SimNewsFormat',
                ['timemark', 'headline', 'contents', 'keycount'])
NewNewsFormat = namedtuple('NewNewsFormat',
                ['timemark', 'headline', 'contents', 'keywords', 'simnews_list'])
#############################################################################
def loadVecdict(file):
  print("Loading vecdict from %s" % file, flush=True)
  with zipfile.ZipFile(file+'.zip', 'r') as myzip:
    with myzip.open(file+'.txt', 'r') as f:
      vecdict = {}
      for line in f:
        splitLine = line.split()
        try:
          word = splitLine[0].decode("utf-8")
        except:
          continue
        vector = [float(val) for val in splitLine[1:]]
        vecdict[word] = vector
  print("Done. %d words loaded!" % len(vecdict))
  return vecdict
def simEnough(word,simword,vecdict,simThr):
  if word in vecdict and simword in vecdict:
    simscore = 1-spatial.distance.cosine(vecdict[word],vecdict[simword])
    if simscore > simThr: return True
  return False
def getWordSims(word,use_filter,vecdict,simThr):
  wordsims=[]
  for synset in wn.synsets(word,lang="jpn"):
    simwords=wn.synset(synset.name()).lemma_names(lang="jpn")
    for simword in simwords:
      if simword in wordsims: continue
      if use_filter and simEnough(word,simword,vecdict,simThr)==False: continue
      wordsims.append(simword)
  return wordsims
def getKeyExpands(keywords,use_expand,use_filter,vecdict,simThr):
  if use_expand==False: return keywords
  keyexpands = keywords
  for word in keywords:
    wordsims = getWordSims(word,use_filter,vecdict,simThr)
    for simword in wordsims:
      if simword not in keyexpands:
        keyexpands.append(simword)
  return keyexpands
def getKeyWords(words,dictionary,normalwords):
  keywords = []
  for word in words:
    if word not in dictionary: continue
    if dictionary[word] < normalwords: continue
    keywords.append(word)
  return keywords
def getSimNewsbyKeys(oldnews_list,newnews_list,dictionary,normalwords,
                     use_expand,use_filter,vecdict,simThr,countThr):
  start_time =time.time()
  newnews_sim_list=[]
  for newnews in newnews_list:
    print("")
    print(newnews.timemark,end="|")
    print(newnews.headline)
    newwords = newnews.headline.split()
    keywords = getKeyWords(newwords,dictionary,normalwords)
    keywords = getKeyExpands(keywords,use_expand,use_filter,vecdict,simThr)
    print(keywords)
    print("----------------------------------------------", flush=True)
    simnews_list = []
    for oldnews in oldnews_list:
      oldwords = oldnews.headline.split()
      keycount = 0
      for oldword in oldwords: 
        if oldword in keywords:
          keycount += 1
      if keycount < countThr or keycount < len(keywords)/10: continue#
      if oldnews == newnews: continue
      #if oldnews in/near simnews_list: continue
      if len(simnews_list)>=10: continue#should be deleted
      simnews_list.append(SimNewsFormat(oldnews.timemark,
                                        oldnews.headline,
                                        oldnews.contents,
                                        keycount))
      print(keycount,end="|")
      print(oldnews.timemark,end="|")
      print(oldnews.headline)
    print(len(simnews_list))
    newnews_sim_list.append(NewNewsFormat(newnews.timemark,
                                          newnews.headline,
                                          newnews.contents,
                                          keywords,
                                          simnews_list))
  end_time=time.time()-start_time
  print("find similar news: %.6s" % end_time)
  return newnews_sim_list
#############################################################################
def readOrgNews(file):
  lines=open(file).read().splitlines()
  print(len(lines)) 
  orgnews = []
  for line in lines:
    columns  = line.split("|")
    timemark = columns[0]
    headline = columns[2]
    contents = columns[3]
    orgnews.append(OrgNewsFormat(timemark,headline,contents))
  return orgnews
#############################################################################
def getSimNewsbyCNN(oldnews_list,newnews_list,dictionary,normalwords,model,task):
  start_time =time.time()
  newnews_sim_list=[]
  for newnews in newnews_list:
    print("")
    print(newnews.timemark,end="|")
    print(newnews.headline)
    newwords = newnews.headline.split()

    simnews_list = []
    for oldnum,oldnews in enumerate(oldnews_list):
      oldwords = oldnews.headline.split()


      s0 = []
      s1 = []
      s0.append(newwords)
      s1.append(oldwords)
      si0, sj0 = task.vocab.vectorize(s0, task.embed, spad=task.c['spad0'])###
      si1, sj1 = task.vocab.vectorize(s1, task.embed, spad=task.c['spad1'])###
      se0 = task.embed.map_jset(sj0)###
      se1 = task.embed.map_jset(sj1)###
      import pysts.nlp
      f0,   f1 = pysts.nlp.Sentence_flags(s0, s1, task.c['spad0'], task.c['spad1'])###
      #grsl = {'score': y, 's0': s0, 's1': s1,
      #               'score1': np.array([s/float(task.c['maxscore']) for s in y]),
      #               'classes': task.sts_labels2categorical(y,int(task.c['maxscore']+1)),
      #               'si0': si0, 'si1': si1, 'sj0': sj0, 'sj1': sj1, 'se0': se0, 'se1': se1, 'f0': f0, 'f1': f1}

      simporb  = model.predict([si0,si1,se0,se1,f0,f1])
      simscore = np.dot(np.array(simporb),np.arange(6))[0]
      
      if oldnum%100==0:
        print(simscore,end=',')
      if oldnum%1000==0:
        print(oldnum,end='/')
        print(len(oldnews_list))

      if simscore < 1.3: continue

      #print('Great!')
      #print(simscore)
      #exit()


      simnews_list.append(SimNewsFormat(oldnews.timemark,
                                        oldnews.headline,
                                        oldnews.contents,
                                        simscore))
    print(len(simnews_list))
    newnews_sim_list.append(NewNewsFormat(newnews.timemark,
                                          newnews.headline,
                                          newnews.contents,
                                          'CNN',
                                          simnews_list))
  end_time=time.time()-start_time
  print("find similar news: %.6s" % end_time)
  return newnews_sim_list
#############################################################################
import tasks
import mods

fname_plot="./simplot"
fname_log ="./simlog"
path_ress ='../ress/'
path_data ='../data/'
path_wf   ='../weights/bestvalid-'
path_pred ='./'

if __name__ == "__main__":
    import time
    start_time = time.time()
    import sys
    modname, taskname, trainf, validf, testf = sys.argv[1:6]
    params = sys.argv[6:]

    import importlib
    task= importlib.import_module('.'+taskname,'tasks').task()
    mod = importlib.import_module( '.'+modname, 'mods').mod()

    conf = task.get_conf()
    mod.add_conf(conf)
    for param in params:
        name, value= param.split('=')
        try:
            conf[name] = eval(value)###eval
        except:
            conf[name] = value###eval
    fplot = open(fname_plot,'a')
    conf['fplot']=fplot###
    task.set_conf(conf)
    import json
    h = hash(json.dumps(dict([(n, str(v)) for n, v in conf.items()]), sort_keys=True))
    print('', file=fplot)###
    print('H: %x' % h, file=fplot)
    print(conf,end='',file=fplot)
    flog  = open(fname_log,'a')
    print('H: %x' % h, file=flog)
    print(conf,file=flog)

    task.initial()
    task.load_resources(path_ress)
    task.load_data(path_data, trainf, validf, testf)
    end_time = time.time()
    print('Time:%f' %(end_time-start_time))
    print('Time:%f' %(end_time-start_time), file=flog)

    for i_run in range(1):###
        print('Creating Model')
        task.create_model(mod)
        model = task.get_model()
        model.load_weights(path_wf+taskname+'-'+modname+'-'+conf['fweight'], by_name=True)
        task.set_model(model)

        print('Evaluating')
        task.compile_model()
        #resT, resv, rest = task.eval_model()
        #task.pred_model(predout=path_pred+'%s'%conf['fweight'])

    end_time = time.time()
    print('Time:%f' %(end_time-start_time))
    print('Time:%f' %(end_time-start_time), file=flog)



    oldfile="./STSJP/sts.UNK.100000.20062015.head.clips"
    newfile="./STSJP/sts.UNK.100000.20062015.rnew"
    oldnews_list=readOrgNews(oldfile)
    newnews_list=readOrgNews(newfile)
    dictionary = pickle.load(open("./STSJP/dictionary","rb"))
    normalwords= 500
    use_expand = False#True
    use_filter = False#True
    vecdict    = {}#loadVecdict("jawiki_vector")
    simThr     = 0.3
    countThr   = 3

    model = task.get_model()
    newnews_sim_list=getSimNewsbyCNN(oldnews_list,newnews_list,dictionary,normalwords,model,task)
    #newnews_sim_list=getSimNewsbyKeys(oldnews_list,newnews_list,dictionary,normalwords,
    #                                  use_expand,use_filter,vecdict,simThr,countThr)
    pickle.dump(newnews_sim_list,open("./STSJP/newnews_sim_list.p","wb"), protocol=2)
