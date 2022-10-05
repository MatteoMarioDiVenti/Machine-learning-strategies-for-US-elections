# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:33:43 2021

@author: mmd2218
"""


import random
import numpy as np
from itertools import product
#
dem_states=249
rep_states=217

#32
startdifference=dem_states-rep_states

Arizona=11
Georgia=16
Pennsylvania=20
NorthCarolina=15    
Wisconsin=10

swing_states=np.array([Arizona, Georgia, Pennsylvania, NorthCarolina, Wisconsin])
states=len(swing_states)
resources= 6
status_quo=np.array([-1,-1,1,-1,1])
#create payoff and list
dec = np.array([np.array(i) for i in product(range(resources+1), repeat=states) if sum(i)==resources]);
num_actions=len(dec);
payoffT=np.zeros((num_actions,num_actions));
payoffB=np.zeros((num_actions,num_actions));


for i in range (num_actions):
    for j in range(num_actions):
        payoffB[i][j]= (startdifference+(np.dot(np.sign(status_quo+ 2*np.sign(dec[i]-dec[j])),swing_states)))>0

        payoffT[i][j]= (-startdifference+(np.dot(np.sign(2*np.sign(dec[i]-dec[j])-status_quo),swing_states)))>0

def getStrategyTable(n):
 #initialize
 regretSum=np.zeros(num_actions);
 oppRegretSum=np.zeros(num_actions);

 strategySum=np.zeros(num_actions);
 oppStrategySum=np.zeros(num_actions);


 def getStrategy(regretSum,sSum):
    normalizingSum=0;
    strategy=np.zeros(num_actions)
    Sum=sSum
    for a in range(num_actions):
     if regretSum[a]>0 :
      strategy[a]=regretSum[a] 
     else:
         strategy[a]=0
     normalizingSum += strategy[a]
    
    for a in range(num_actions):
        if normalizingSum>0 :
            strategy[a] /= normalizingSum
        else:
            strategy[a]= 1/num_actions
        Sum[a]+=strategy[a]
    sSum=Sum    
    return(strategy)
                
                
 def getAction(strategy)  :
    r=random.random()
    a=0
    cumulativeProbability=0
    while a < num_actions-1:
        cumulativeProbability+=strategy[a]
        if r < cumulativeProbability:
            break
        a+=1
    return(a)

 def Train(iterations):
    for i in range(iterations+1):
         newStrategy=getStrategy(regretSum, strategySum)
         oppNewStrategy=getStrategy(oppRegretSum, oppStrategySum)
         myAction=getAction(newStrategy)
         oppAction=getAction(oppNewStrategy)
    #get utility     
         actionUtility=payoffB[:,oppAction]
         oppActionUtility=payoffT[:,myAction]
    #get regret
         for a in range(num_actions):
          regretSum[a]+=actionUtility[a]-actionUtility[myAction]
          oppRegretSum[a]+=oppActionUtility[a]-oppActionUtility[oppAction]    

    
        
 def getAverageStrategy(straSum):
    avgStrategy=np.zeros(num_actions)
    normalizingSum=0
    for a in range(num_actions):
        normalizingSum+=straSum[a]
    for a in range(num_actions):
        if normalizingSum >0:
            avgStrategy[a]=straSum[a]/normalizingSum
        else:
            avgStrategy[a]=1.0/num_actions
    return(avgStrategy)
    
 Train(10000);
 Strategies_table=np.zeros((num_actions,7))
 Strategies_table[:,0]=getAverageStrategy(oppStrategySum)
 Strategies_table[:,1]=getAverageStrategy(strategySum)
 Strategies_table[:,2:7]=dec

 proxyT=np.zeros((num_actions,5))
 proxyB=np.zeros((num_actions,5))
 for i in range(num_actions):
    proxyT[i,:]=dec[i,:]*Strategies_table[i,0]
    proxyB[i,:]=dec[i,:]*Strategies_table[i,1]
    
 final_resultT=np.dot(np.ones([1,num_actions]),proxyT)/6
 final_resultB=np.dot(np.ones([1,num_actions]),proxyB)/6
 print(final_resultT)
 print(final_resultB)

 return(Strategies_table)