import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

if __name__ == "__main__":
  logname = sys.argv[1]
  flog=open(logname,"r")
  lines=flog.read().splitlines()
  print(len(lines))
  py = [(float(line.split("\t")[0]),float(line.split("\t")[1])) for line in lines]
  p0 = []
  p1 = []
  p2 = []
  p3 = []
  p4 = []
  p5 = []
  for pair in py:
    if float(pair[1])==0 or float(pair[1])<=0.5:
      p0.append(float(pair[0]))
      continue
    if float(pair[1])==1 or (float(pair[1])>0.5 and float(pair[1])<=1.5):
      p1.append(float(pair[0]))
      continue
    if float(pair[1])==2 or (float(pair[1])>1.5 and float(pair[1])<=2.5):
      p2.append(float(pair[0]))
      continue
    if float(pair[1])==3 or (float(pair[1])>2.5 and float(pair[1])<=3.5):
      p3.append(float(pair[0]))
      continue
    if float(pair[1])==4 or (float(pair[1])>3.5 and float(pair[1])<=4.5):
      p4.append(float(pair[0]))
      continue
    if float(pair[1])==5 or float(pair[1])>4.5:
      p5.append(float(pair[0]))
      continue
    print("Not integer!")
  print(len(p0))
  print(len(p1))
  print(len(p2))
  print(len(p3))
  print(len(p4))
  print(len(p5))
  print(len(p0)+len(p1)+len(p2)+len(p3)+len(p4)+len(p5))
  ypred = [float(pair[0]) for pair in py]
  ygold = [float(pair[1]) for pair in py]
  print(ypred[:10])
  print(ygold[:10])
  pr = pearsonr(ypred, ygold)[0]
  sr = spearmanr(ypred, ygold)[0]
  e  = mse(ypred, ygold)
  print('Pearson: %f' % ( pr,))
  print('Spearman: %f' % ( sr,))
  print('MSE: %f' % ( e,))
  #plt.text(0.0,0.0,s='predictions', ha='center', va='center')
  #plt.text(0.0,0.0,s='counts', ha='center', va='center', rotation='vertical')
  bins   = np.arange(-2, 7, 0.25)#fixed bin size
  normed = False
  alpha  = 0.2
  plt.figure(1)
  plt.suptitle('Org Data')
  plt.subplot(711)
  plt.hist(p0, bins=bins, normed=normed,facecolor='r',edgecolor='r',hold=0,alpha=alpha)
  plt.subplot(712)
  plt.hist(p1, bins=bins, normed=normed,facecolor='g',edgecolor='g',hold=1,alpha=alpha)
  plt.subplot(713)
  plt.hist(p2, bins=bins, normed=normed,facecolor='b',edgecolor='b',hold=2,alpha=alpha)
  plt.subplot(714)
  plt.hist(p3, bins=bins, normed=normed,facecolor='c',edgecolor='c',hold=3,alpha=alpha)
  plt.subplot(715)
  plt.hist(p4, bins=bins, normed=normed,facecolor='m',edgecolor='m',hold=4,alpha=alpha)
  plt.subplot(716)
  plt.hist(p5, bins=bins, normed=normed,facecolor='y',edgecolor='y',hold=5,alpha=alpha)#k,w
  plt.subplot(717)
  plt.hist(ypred, bins=bins, normed=normed,facecolor='k',edgecolor='k',hold=6,alpha=alpha)
  #plt.xlabel('predictions')
  #plt.ylabel('counts')
  #p=p0
  #print(len(p))
  plt.figure(2)
  plt.suptitle('Org Data')
  x = [pair[1] for pair in py]
  y = [pair[0] for pair in py]
  plt.scatter(x,y,alpha=0.061)

  ypred = [float(pair[0]) for pair in py]
  ygold = [float(pair[1]) for pair in py]
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
  py = [(ypred[i],ygold[i]) for i in range(len(ypred))]
  p0 = []
  p1 = []
  p2 = []
  p3 = []
  p4 = []
  p5 = []
  for pair in py:
    if float(pair[1])==0 or float(pair[1])<=0.5:
      p0.append(float(pair[0]))
      continue
    if float(pair[1])==1 or (float(pair[1])>0.5 and float(pair[1])<=1.5):
      p1.append(float(pair[0]))
      continue
    if float(pair[1])==2 or (float(pair[1])>1.5 and float(pair[1])<=2.5):
      p2.append(float(pair[0]))
      continue
    if float(pair[1])==3 or (float(pair[1])>2.5 and float(pair[1])<=3.5):
      p3.append(float(pair[0]))
      continue
    if float(pair[1])==4 or (float(pair[1])>3.5 and float(pair[1])<=4.5):
      p4.append(float(pair[0]))
      continue
    if float(pair[1])==5 or float(pair[1])>4.5:
      p5.append(float(pair[0]))
      continue
    print("Not integer!")
  print(len(p0))
  print(len(p1))
  print(len(p2))
  print(len(p3))
  print(len(p4))
  print(len(p5))
  print(len(p0)+len(p1)+len(p2)+len(p3)+len(p4)+len(p5))
  ypred = [float(pair[0]) for pair in py]
  ygold = [float(pair[1]) for pair in py]
  print(ypred[:10])
  print(ygold[:10])
  pr = pearsonr(ypred, ygold)[0]
  sr = spearmanr(ypred, ygold)[0]
  e  = mse(ypred, ygold)
  print('Pearson: %f' % ( pr,))
  print('Spearman: %f' % ( sr,))
  print('MSE: %f' % ( e,))
  bins = np.arange(-2, 7, 0.25)
  alpha = 0.2
  plt.figure(3)
  plt.suptitle('Nor Data')
  plt.subplot(711)
  plt.hist(p0, bins=bins, normed=normed,facecolor='r',edgecolor='r',hold=0,alpha=alpha)
  plt.subplot(712)
  plt.hist(p1, bins=bins, normed=normed,facecolor='g',edgecolor='g',hold=1,alpha=alpha)
  plt.subplot(713)
  plt.hist(p2, bins=bins, normed=normed,facecolor='b',edgecolor='b',hold=2,alpha=alpha)
  plt.subplot(714)
  plt.hist(p3, bins=bins, normed=normed,facecolor='c',edgecolor='c',hold=3,alpha=alpha)
  plt.subplot(715)
  plt.hist(p4, bins=bins, normed=normed,facecolor='m',edgecolor='m',hold=4,alpha=alpha)
  plt.subplot(716)
  plt.hist(p5, bins=bins, normed=normed,facecolor='y',edgecolor='y',hold=5,alpha=alpha)#k,w
  plt.subplot(717)
  plt.hist(ypred, bins=bins, normed=normed,facecolor='k',edgecolor='k',hold=6,alpha=alpha)
  plt.figure(4)
  plt.suptitle('Nor Data')
  x = [pair[1] for pair in py]
  y = [pair[0] for pair in py]
  plt.scatter(x,y,alpha=0.061)

  ypred = [float(pair[0]) for pair in py]
  ygold = [float(pair[1]) for pair in py]
  def Z_ScoreIntegration(x):
    if x<=0.5: return 0.0
    if (x>0.5 and x<=1.5): return 1.0
    if (x>1.5 and x<=2.5): return 2.0
    if (x>2.5 and x<=3.5): return 3.0
    if (x>3.5 and x<=4.5): return 4.0
    if x>4.5: return 5.0
  ypred = [Z_ScoreIntegration(x) for x in ypred]
  py = [(ypred[i],ygold[i]) for i in range(len(ypred))]
  p0 = []
  p1 = []
  p2 = []
  p3 = []
  p4 = []
  p5 = []
  for pair in py:
    if float(pair[1])==0 or float(pair[1])<=0.5:
      p0.append(float(pair[0]))
      continue
    if float(pair[1])==1 or (float(pair[1])>0.5 and float(pair[1])<=1.5):
      p1.append(float(pair[0]))
      continue
    if float(pair[1])==2 or (float(pair[1])>1.5 and float(pair[1])<=2.5):
      p2.append(float(pair[0]))
      continue
    if float(pair[1])==3 or (float(pair[1])>2.5 and float(pair[1])<=3.5):
      p3.append(float(pair[0]))
      continue
    if float(pair[1])==4 or (float(pair[1])>3.5 and float(pair[1])<=4.5):
      p4.append(float(pair[0]))
      continue
    if float(pair[1])==5 or float(pair[1])>4.5:
      p5.append(float(pair[0]))
      continue
    print("Not integer!")
  print(len(p0))
  print(len(p1))
  print(len(p2))
  print(len(p3))
  print(len(p4))
  print(len(p5))
  print(len(p0)+len(p1)+len(p2)+len(p3)+len(p4)+len(p5))
  ypred = [float(pair[0]) for pair in py]
  ygold = [float(pair[1]) for pair in py]
  print(ypred[:10])
  print(ygold[:10])
  pr = pearsonr(ypred, ygold)[0]
  sr = spearmanr(ypred, ygold)[0]
  e  = mse(ypred, ygold)
  print('Pearson: %f' % ( pr,))
  print('Spearman: %f' % ( sr,))
  print('MSE: %f' % ( e,))
  bins = np.arange(-2, 7, 0.25)
  alpha = 0.2
  plt.figure(5)
  plt.suptitle('Int Data')
  plt.subplot(711)
  plt.hist(p0, bins=bins, normed=normed,facecolor='r',edgecolor='r',hold=0,alpha=alpha)
  plt.subplot(712)
  plt.hist(p1, bins=bins, normed=normed,facecolor='g',edgecolor='g',hold=1,alpha=alpha)
  plt.subplot(713)
  plt.hist(p2, bins=bins, normed=normed,facecolor='b',edgecolor='b',hold=2,alpha=alpha)
  plt.subplot(714)
  plt.hist(p3, bins=bins, normed=normed,facecolor='c',edgecolor='c',hold=3,alpha=alpha)
  plt.subplot(715)
  plt.hist(p4, bins=bins, normed=normed,facecolor='m',edgecolor='m',hold=4,alpha=alpha)
  plt.subplot(716)
  plt.hist(p5, bins=bins, normed=normed,facecolor='y',edgecolor='y',hold=5,alpha=alpha)#k,w
  plt.subplot(717)
  plt.hist(ypred, bins=bins, normed=normed,facecolor='k',edgecolor='k',hold=6,alpha=alpha)
  plt.figure(6)
  plt.suptitle('Int Data')
  x = [pair[1] for pair in py]
  y = [pair[0] for pair in py]
  plt.scatter(x,y,alpha=0.061)

  diff = [(pair[0]-pair[1]) for pair in py]
  diffin0 = 0.0
  for x in diff:
    if x==0: diffin0+=1
  print(diffin0)
  print(diffin0/len(lines))
  diffin1 = 0.0
  for x in diff:
    if x>=-1 and x<=1: diffin1+=1
  print(diffin1)
  print(diffin1/len(lines))
  plt.figure(7)
  plt.suptitle('Diff Data')
  plt.hist(diff, bins=np.arange(-5.5, 5.5, 1), normed=normed,facecolor='r',edgecolor='r',hold=0,alpha=alpha)
  plt.show()







  exit()
  plt.axhline(y=0.72, linewidth=0.5, color='k')
  plt.axhline(y=0.74, linewidth=0.5, color='k')
  plt.axhline(y=0.80, linewidth=0.5, color='k')
  plt.axhline(y=0.82, linewidth=0.5, color='k')
  plt.legend(loc="lower right")
  plt.show()

  p = [float(line.split("\t")[0]) for line in lines]
  y = [float(line.split("\t")[1]) for line in lines]
  plt.scatter(y, p)
  plt.show()

  fig, ax = plt.subplots(nrows=2,ncols=2)
   
  for row in ax:
      for col in row:
          col.plot(x, y)

  plt.subplots(6)

  #fig, ax = plt.subplots(nrows=2,ncols=3)

  #plt.xlim([min(data)-5, max(data)+5])
  #plt.hist(p, bins=bins, alpha=0.5)


