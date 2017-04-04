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
if __name__ == "__main__":
  oldfile="./sts.UNK.100000.20062015.head.clips"
  newfile="./sts.UNK.100000.20062015.rnew"
  oldnews_list=readOrgNews(oldfile)
  newnews_list=readOrgNews(newfile)
  dictionary = pickle.load(open("./dictionary","rb"))
  normalwords= 500
  use_expand = False#True
  use_filter = False#True
  vecdict    = {}#loadVecdict("jawiki_vector")
  simThr     = 0.3
  countThr   = 3
  newnews_sim_list=getSimNewsbyKeys(oldnews_list,newnews_list,dictionary,normalwords,
                                    use_expand,use_filter,vecdict,simThr,countThr)
  pickle.dump(newnews_sim_list,open("./newnews_sim_list.p","wb"), protocol=2)
