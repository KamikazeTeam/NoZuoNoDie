import numpy as np
import collections
from collections import defaultdict
from operator import itemgetter
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

def load_sts(filename, skip_unlabeled=True):
    s0 = []
    s1 = []
    labels = []
    lines=open(filename,'r').read().splitlines()
    for line in lines:
        line = line.rstrip()
        label, s0x, s1x = line.split('\t')
        if label == '':
            if skip_unlabeled: continue
            else: labels.append(-1.)
        else: labels.append(float(label))
        s0.append(word_tokenize(s0x))
        s1.append(word_tokenize(s1x))
    return (s0, s1, np.array(labels))
def showlen(sentences):
    lenlist=[len(words) for words in sentences]
    maxlen = max(lenlist)
    print(maxlen)
    avglen = sum(lenlist)/len(lenlist)
    print(avglen)
    print(len(lenlist))
    lencount=collections.Counter(lenlist)
    #print(lencount)
    plt.hist(lenlist, bins=np.arange(0, 90, 1), normed=False,
             facecolor='r',edgecolor='r',alpha=1.0)
    plt.show()
def getcount(sentences):
    vocabulary_size=20000
    allwordlist = []
    for words in sentences:
      for word in words:
        allwordlist.append(word.lower())
    print(len(allwordlist))
    def build_dataset(words):
      count = [['UNK', -1]]
      count.extend(collections.Counter(words).most_common(vocabulary_size-1))
      dictionary = dict()
      for word, _ in count:
        dictionary[word] = len(dictionary)
      data = list()
      unk_count = 0
      for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count += 1
        data.append(index)
      count[0][1] = unk_count
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      return data, count, dictionary, reverse_dictionary
    data, count, dictionary, reverse_dictionary = build_dataset(allwordlist)
    del allwordlist  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:100])
    print('Most rare words (+UNK)', count[-100:])
    print('Sample data', data[:30], [reverse_dictionary[i] for i in data[:30]])
    print("dictionary size", len(dictionary), flush=True)
    return count
def cuscount(sentences):
    icase=True
    count_thres=1
    vocabset = defaultdict(int)
    for s in sentences:
        for t in s:
            vocabset[t if not icase else t.lower()] += 1
    #print(vocabset)
    vocab = list(map(itemgetter(0), 
                     sorted(filter(lambda k: itemgetter(1)(k) >= count_thres,
                                   vocabset.items()),
                            key=itemgetter(1, 0), reverse=True)))
    vocab_N = len(vocab)
    #vocab = vocab[:20000]
    word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
    word_idx['_PAD_'] = 0
    word_idx['_OOV_'] = 1
    print('Vocabulary of %d words (adaptable: %d)' % (vocab_N, len(word_idx)))    
    #print(vocab)
    return vocabset
def getdict():
    w = dict()
    g = []
    g.append(np.zeros(300))
    with open('glove.6B.300d.txt', 'r') as f:
        for line in f:
            l = line.split()
            word = l[0]
            w[word] = len(g)
            g.append(np.array(l[1:]).astype(float))
    g = np.array(g, dtype='float32')
    return w,g

w,_=getdict()

s0,s1,y=load_sts("train.tsv")
s=s0+s1
showlen(s)
counts = getcount(s)
dicts  = cuscount(s)

numoferr=0
for name,value in counts:
  if name not in w:
    #print(name,value)
    numoferr+=1
print(numoferr)

v0,v1,y=load_sts("valid.tsv")
v=v0+v1
showlen(v)
countv = getcount(v)

numoferr=0
for name,value in countv:
  if name not in w:# and name not in dicts:
    #print(name,value)
    numoferr+=1
print(numoferr)

t0,t1,y=load_sts("test.tsv")
t=t0+t1
showlen(t)
countt = getcount(t)

numoferr=0
for name,value in countt:
  if name not in w:# and name not in dicts:
    print(name,value)
    numoferr+=1
print(numoferr)


exit()
"""tf-idf 12
of” 6
“…is 6
úohs 5
doesn’t 4
words” 4
cˇtú 4
peropos 4
“bags 4
• 4
event’s 3
'term 3
a|b 3
héma-quebec 3
supprimerai 3
renièrent 3
b|a 3
bayón 2
fiatal 2
'forgotten 2
'justice 2
pozoblanco 2
baden-wuttenberg 2
bayesians 2
ifjú 2
frequency-inverse 2
superclasses 2
bayes’ 2
harmadik 2
pankseep 2
együtt 2
abbvie 2
kneˇzˇínková 2
décary 2
vibrants 2
plachetka 2
évezredért 2
gox 2
zawlocki 2
them. 2
józsefváros 2
chapattis 2
“prior” 2
boxeur 2
ty´den.cz 2
ashya 2
services. 2
esély 2
anti-graft 2
lebrijano 2
katerˇina 2
budapestért 2
concurrenciez 2
'absence 2
illhousiens 2
16h 2
£14 2
torrione 2
19h 2
guilvinec 2
alapítvány 2
szocialisták 2
bingleys 2
baloldal 2
minimally-effective 2
fredrico 2
path-finding 1
divulgues 1
non-assertive 1
'failed 1
septuagénaire 1
coéquipiers 1
shellac-based 1
mentally-ill 1
crisis-hit 1
binette 1
water-damaged 1
step-father- 1
orwellien 1
us-french 1
stucco-like 1
permalink 1
fan/light 1
bascula 1
controversy-ridden 1
aluminum/glass 1
naiveness 1
sub-problem 1
‘extend’ 1
waaaaay 1
£450millions 1
'- 1
sense. 1
matinier 1
ukraine-russia 1
surenchérissaient 1
best/easiest 1
google’s 1
obstruant 1
constucted 1
“extend” 1
'photoshopped 1
jirˇí 1
eacher 1
effiecent 1
'students 1
majuscules 1
ihned.cz 1
‘important’ 1
rencontrèrent 1
éditrices 1
d-mult 1
granduating 1
'worsened 1
shadowed/overridden 1
stackoverflow 1
émergée 1
absorbement 1
recherche- 1
'respond 1
'gas 1
conseillera 1
'exploits 1
primer/paint 1
beer.10 1
rejoignèrent 1
désaltérantes 1
youtube-version 1
that. 1
55bn 1
moses-inspired 1
mh17 1
allready 1
reality/history 1
'incorrect 1
'you 1
'friday 1
w/light 1
'weekend 1
demans 1
dry/cure 1
renoua 1
supercolle 1
orwelliens 1
hammerite 1
77year 1
'jihadi 1
'improved 1
beˇlohlávek 1
//www.dsattorney.com/qa-pseudonyms-in-contracts/ 1
£81 1
say/imply 1
low-hand 1
clinic. 1
finger-picking 1
'willingness 1
equipments/facilities 1
nigeria\ 1
thaï 1
hole/gap 1
non-refereed 1
réctifia 1
reward/benefits 1
diplômer 1
1mdb 1
website’s 1
'panicked 1
fabriquaient 1
re-heat 1
'concern 1
unisexe 1
re-commemorate 1
'bookkeeper 1
trouves 1
'very 1
saudi-led 1
dragooning 1
buteur 1
father- 1
stayfor 1
bloquèrent 1
sex-tape 1
delinquant 1
anti-g8 1
'special 1"""
exit()
w = dict()
g = []
g.append(np.zeros(300))
with open('glove.6B.300d.txt', 'r') as f:
    for line in f:
        l = line.split(' ')
        word = l[0]
        w[word] = len(g)
        g.append(np.array(l[1:]).astype(float))
g = np.array(g, dtype='float32')

print("cal avg",flush=True)
lenlist=[len(vec) for vec in g]
avglen = sum(lenlist)/len(lenlist)
print(avglen)



exit()
with open('glove.6B.300d.txt', 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
with open('glove.6B.300d.txt', 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

