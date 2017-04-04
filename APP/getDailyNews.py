import nltk
import random

def getDailyNews(date):
  f = open("./sts.UNK.100000.20062015.head.clips")
  fr = open("./sts.UNK.100000.20062015.rnew","w")
  raw = f.read()
  lines = raw.splitlines()
  print(len(lines))
  collectednum = 0
  for line in lines:
    columns = line.split("|")
    if columns[0] in date:
      print(line,file=fr)
      collectednum+=1
      if collectednum >= 100: break

if __name__ == "__main__":
  import sys
  date = []
  if len(sys.argv) >= 2:
    for i in range(1,len(sys.argv)):
      date.append(sys.argv[i])
  else:
    date=["2015-9-30","2015-5-10","2015-1-12","2015-1-31","2015-11-5","2015-8-4"]
  getDailyNews(date)
