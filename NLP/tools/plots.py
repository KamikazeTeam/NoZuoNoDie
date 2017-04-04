#!/usr/bin/python3
#set fileencoding=utf8

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
cycol = cycle('mbcgrk')#y

flog=open("./simplot","r")
lines=flog.read().splitlines()
for i in range(0,len(lines)):
	if len(lines[i])==0:continue
	#if lines[i][0]=="{" or lines[i][0]=="O":
	if lines[i][0]=="H":
		color=next(cycol)
		continue
	record = lines[i].split("|")
	record = record[:len(record)-1]
	#print(record)
	try:
		traind = [float(pair.split(",")[0]) for pair in record]
		#print(traind)
		plt.plot(traind,label=i,color=color,alpha=0.2)
		validd = [float(pair.split(",")[1]) for pair in record]
		plt.plot(validd,label=i,color=color,alpha=0.2)
		testd  = [float(pair.split(",")[2]) for pair in record]
		plt.plot(testd,label=i,color=color,alpha=1.0)
		#testdd = []
		#for i in range(len(testd)):
		#	if i%12==0:
		#		testdd.append(traind[i])
		#	else:
		#		testdd.append(-100.0)
		#plt.plot(testdd,label=i,color='r',alpha=1.0)
	except:# ValueError, e:
		print('error')

plt.axhline(y=0.60, linewidth=0.5, color='k')
plt.axhline(y=0.70, linewidth=0.5, color='k')
plt.axhline(y=0.72, linewidth=0.5, color='k')
plt.axhline(y=0.74, linewidth=0.5, color='k')
plt.axhline(y=0.76, linewidth=0.5, color='k')
plt.axhline(y=0.78, linewidth=0.5, color='k')
plt.axhline(y=0.80, linewidth=0.5, color='k')
plt.axhline(y=0.82, linewidth=0.5, color='k')
plt.axhline(y=0.84, linewidth=0.5, color='k')
plt.axhline(y=0.86, linewidth=0.5, color='k')
plt.axhline(y=0.88, linewidth=0.5, color='k')
plt.axhline(y=0.90, linewidth=0.5, color='k')
plt.axhline(y=0.92, linewidth=0.5, color='k')
plt.axhline(y=0.94, linewidth=0.5, color='k')
plt.axhline(y=0.96, linewidth=0.5, color='k')
plt.legend(loc="lower left", bbox_to_anchor=(-0.1,0.0))
axes = plt.gca()
axes.set_ylim([0.035,1.0])
plt.show()
