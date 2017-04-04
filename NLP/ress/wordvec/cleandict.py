import nltk
import random
import pickle
from collections import namedtuple
import string

punc = list("！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")
punc.extend(string.punctuation)
fnew = open("jawiki.200","w")
fold = open("jawikiold.200","r")

oldlines = fold.read().splitlines()

for i,oldline in enumerate(oldlines):
    #if i<33306: continue
    fullelements = oldline.split()
    if len(fullelements)==0: continue
    try:
      word=''.join(fullelements[:-200])
      #print(word)
      #word=fullelements[0]
      elements = fullelements[-200:]
      #print(elements)
      if len(elements)!=200: exit()
      #elements = fullelements[1:]
      chars=list(word)
      cleanword=[]
      for char in chars:
        if char in punc:
          continue
        else:
          cleanword.append(char)
      if len(cleanword)==0: continue
      #print(cleanword)
      #print(word)
      #print(chars)
      #if i>33307 :exit()
      print(''.join(cleanword),end=" ",file=fnew)
      for j,element in enumerate(elements):
        try:
          value = float(element)
        except:
          print(i)
          print(fullelements)
          exit()
        if j < len(elements)-1:
          print(element,end=" ",file=fnew)
        else:
          print(element,file=fnew)
    except:
      print(fullelements)
      exit()



















exit()
dictionary = pickle.load(open("./dictionary","rb"))
nwords = 500
ymdmai = ""#"2006"#-6"
ymdnik = ""#. 6"
limit  = 0
deletenikwords=["Ｗ杯","サッカー","ラグビー"]
deletemaiwords=["■","【","★","◇"]
simformat = namedtuple('simformat',
  ['simheadcount','simparacount','simheadwords','simparawords','simdate','simhead','simpara'])

for j,nikline in enumerate(niklines):
  if j%100==0: print(j, len(niklines),flush=True)
  nikcol =nikline.split("|")
  nikdate=nikcol[0]
  if ymdnik not in nikdate: continue
  nikhead=nikcol[1]
  if "。" not in nikhead: continue#choose only sentence
  nikhead=nikhead.split("。")[0]#first head
  nikpara=nikcol[2].split("。")[0]#first para
  if len(nikpara) > 80: continue#delete long sentence
  if len(nikpara) < len(nikhead): continue#delete short sentence
  nikwords=nikhead+nikpara

  STOP=False
  for deletenikword in deletenikwords:
    if deletenikword in nikwords: STOP=True#delete words
  if STOP: continue

  simlines = []
  for i,mailine in enumerate(mailines):
    #if i%100000==0: print(i, len(mailines),flush=True)
    columns = mailine.split("|")
    maidate=columns[0]
    if ymdmai not in maidate: continue#-5-10":
    maipage=columns[1]
    maihead=columns[2]
    maipara=columns[3].split("。")[0]
    maipara=maipara.split("…")[0]#first para
    maiheadwords=maihead.split(" ")
    maiparawords=maipara.split(" ")
    if len(maiparawords) > 30: continue#delete long sentence
    if len(maiparawords) < len(maiheadwords): continue#delete short sentence
    maiwords=maiheadwords+maiparawords

    STOP=False
    for deletemaiword in deletemaiwords:
      for maiword in maiwords:
        if deletemaiword in maiword: STOP=True#delete words
    if STOP: continue

    keyheadwords = []
    for maiheadword in maiheadwords:
      if len(maiheadword)==1: continue#delete 1 word word
      if maiheadword not in dictionary: continue
      if dictionary[maiheadword] < nwords: continue
      keyheadwords.append(maiheadword)
    keyparawords = []
    for maiparaword in maiparawords:
      if len(maiparaword)==1: continue#delete 1 word word
      if maiparaword not in dictionary: continue
      if dictionary[maiparaword] < nwords: continue
      keyparawords.append(maiparaword)

    keyheadcount=0
    cheadwords = []
    for keyheadword in keyheadwords:
      if keyheadword in nikhead:
        keyheadcount+=1
        if keyheadword not in cheadwords:
            cheadwords.append(keyheadword)
    keyparacount=0
    cparawords = []
    for keyparaword in keyparawords:
      if keyparaword in nikpara:
        keyparacount+=1
        if keyparaword not in cparawords:
            cparawords.append(keyparaword)

    #if keyheadcount+keyparacount<5: continue
    #if abs(keyheadcount-keyparacount)<2: continue
    if keyheadcount<3: continue
    if keyparacount>2: continue
    #if nikdate.split('.')[1]!=maidate.split('-')[1]: continue
    simlines.append(simformat(keyheadcount,keyparacount,cheadwords,cparawords,
                              maidate,maiheadwords,maiparawords))
  
  #simlines=sorted(simlines,key=lambda x: abs(x.simheadcount-x.simparacount),reverse=True)
  simlines=sorted(simlines,key=lambda x: x.simparacount,reverse=True)
  eachlimit=0
  for simline in simlines:
    #print(simline.simheadcount,end="|",file=fpai)
    #print(simline.simparacount,end="|",file=fpai)
    #print(simline.simheadwords,end="",file=fpai,flush=True)
    #print(simline.simparawords,file=fpai,flush=True)
    #print(nikdate,end="|",file=fpai)
    #print(nikhead,end="\t\t",file=fpai)
    #print(simline.simdate,end="|",file=fpai)
    #print("".join(simline.simhead),file=fpai,flush=True)
    #print("",end="\t",file=fpai)
    print(nikpara,end="\t\t",file=fpai)
    print("".join(simline.simpara),file=fpai,flush=True)
    #print(simline.simwords,end="\t",file=fpai)
    #print("".join(maiheadwords),end="\t\t",file=fpai)
    #print(simline.simhead,file=fpai,flush=True)
    #print("".join(maiparawords),end="\t\t",file=fpai)
    #print(simline.simpara,file=fpai,flush=True)
    #print("".join(maiparawords),file=fpai)
    #print("",end="\t\t\t",file=fpai)
    #print(simline.simpara,file=fpai,flush=True)
    #print("".join(maiheadwords+maiparawords),file=fpai,flush=True)
    #print(nikhead+nikpara,end="\t\t",file=fpai)
    eachlimit+=1
    #if eachlimit>=6: break
    limit+=1
    print(limit,flush=True)
    #if limit>=2000: exit()























exit()
for i,mailine in enumerate(mailines):
  if i%100==0: print(i, len(mailines),flush=True)
  columns = mailine.split("|")
  maidate=columns[0]
  maipage=columns[1]
  maihead=columns[2]
  maipara=columns[3].split("。")[0]
  maipara=maipara.split("…")[0]#first para
  if ymdmai not in maidate: continue#-5-10":
  maiheadwords=maihead.split(" ")
  maiparawords=maipara.split(" ")
  if len(maiparawords) > 30: continue#delete long sentence
  words=maiheadwords+maiparawords
  STOP=False
  for word in words:
    if word in deletewords: STOP=True#delete words
  if STOP: continue
  keywords = []
  for word in words:
    if len(word)==1: continue#delete 1 word word
    if word not in dictionary: continue
    if dictionary[word] < nwords: continue
    keywords.append(word)
  simlines = []
  for j,nikline in enumerate(niklines):
    #if j%10000==0: print(j, len(niklines))
    nikcol =nikline.split("|")
    nikdate=nikcol[0]
    nikhead=nikcol[1]
    if "。" not in nikhead: continue#choose only sentence
    nikhead=nikhead.split("。")[0]#first head
    nikpara=nikcol[2].split("。")[0]#first para
    nikwords=nikhead+nikpara
    if ymdnik not in nikdate: continue
    keycount=0
    coverwords = []
    for keyword in keywords:
      if keyword in nikwords:
        keycount+=1
        if keyword not in coverwords:
            coverwords.append(keyword)
    if keycount<6: continue#keycount
    if len(nikpara) > 80: continue#delete long sentence
    simlines.append(simformat(keycount,coverwords,nikhead,nikpara))
  simlines=sorted(simlines,key=lambda x: x.simcount,reverse=True)
  eachlimit=0
  for simline in simlines:
    print(simline.simcount,end="|",file=fpai)
    #print(simline.simwords,end="\t",file=fpai)
    #print("".join(maiheadwords),end="\t\t",file=fpai)
    #print(simline.simhead,file=fpai,flush=True)
    #print("".join(maiparawords),end="\t\t",file=fpai)
    #print(simline.simpara,file=fpai,flush=True)
    print("".join(maiparawords),file=fpai)
    print("",end="\t\t\t",file=fpai)
    print(simline.simpara,file=fpai,flush=True)
    #print("".join(maiheadwords+maiparawords),file=fpai,flush=True)
    #print(nikhead+nikpara,end="\t\t",file=fpai)
    eachlimit+=1
    if eachlimit>=3: break
    limit+=1
    print(limit)
    if limit>=100: exit()











exit()
fpai = open("pair","w")
fmai = open("./sts.UNK.100000.20062015.head.clips","r")
fnik = open("nikkei2006","r")
mailines = fmai.read().splitlines()
niklines = fnik.read().splitlines()
print(len(mailines))
for i,mailine in enumerate(mailines):
  print(i, len(mailines))
  columns = mailine.split("|")
  if columns[0]=="2015-5-10":
    keywords=columns[3].split(" ")
    for nikline in niklines:
      for keyword in keywords:
          if keyword in nikline:
              print(nikline.split("|")[1],end="\t\t\t",file=fpai)
              print(columns[3],file=fpai)



