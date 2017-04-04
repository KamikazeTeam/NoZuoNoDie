#!/usr/bin/python3
#set fileencoding=utf8

import numpy as np
import sys

if __name__ == "__main__":
  logname = sys.argv[1]
  print(logname)
  fout=open(logname+'-n',"w")
  flog=open(logname,"r")
  lines=flog.read().splitlines()
  print(len(lines))
  ypred = [float(line.split("\t")[0]) for line in lines]
  ygold = [float(line.split("\t")[1]) for line in lines]
  s0 = [line.split("\t")[2] for line in lines]
  s1 = [line.split("\t")[3] for line in lines]
  normal = 'maxmin'#'maxmin''normal'
  if normal == 'normal':
    def ZScoreNormalization(x,mu,sigma):
        x = (x - mu) / sigma; 
        return x;
    ypred_mu    = np.average(ypred)
    ypred_sigma = np.std(ypred)
    ypred = [ZScoreNormalization(x,ypred_mu,ypred_sigma) for x in ypred]
    ypred = [(x*1.87+2.5) for x in ypred]
  if normal == 'maxmin':
    def MaxminNormalization(x,xmax,xmin):
        x = (x - xmin) / (xmax - xmin);
        return x;
    ypred_max = max(ypred)
    ypred_min = min(ypred)
    ypred = [MaxminNormalization(x,ypred_max,ypred_min) for x in ypred]
    ypred = [(x*5.0) for x in ypred]
  #records = [(ypred[i],ygold[i],s0[i],s1[i]) for i in range(len(lines))]
  #sortedrecords=sorted(records,key=lambda x: (float(x[0])-float(x[1])),reverse=True)
  #for record in sortedrecords:
  #  print('%.2f'%record[0],end="\t",file=fout)
  #  print('%.2f'%record[1],end="\t",file=fout)
  #  print(record[2],end="\t\t\t",file=fout)
  #  print(record[3],file=fout)
  for i in range(len(ypred)):
    print(ypred[i],file=fout)
