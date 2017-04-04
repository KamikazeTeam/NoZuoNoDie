import nltk
import collections
import pickle

vocabulary_size=100000
sourcefile="./sts.UNK.100000.20062015.head.clips"
raw=open(sourcefile).read()
lines=raw.splitlines()
print(len(lines))
allwordlist = []
for line in lines:
  words = line.split()
  #if len(words) == 0: continue
  for word in words:
    allwordlist.append(word)
#print(len(allwordlist), flush=True)
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
pickle.dump(dictionary,open("./dictionary","wb"))