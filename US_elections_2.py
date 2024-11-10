# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:33:43 2021

@author: mmd2218
"""

#%%
import random
import numpy as np
from itertools import product
import pandas as pd
#%%
import numpy as np
from itertools import product

# US 2024 election udpate (with FT style and dead-heat swing states)
# Define swing state values
swing_states = np.array([
    11,  # Arizona
    16,  # Georgia
    19,  # Pennsylvania
    16,  # North Carolina
    10,  # Wisconsin
    15,  # Michigan
    6,   # Nevada
    7,   # New Hampshire+neb+maine
])

states = len(swing_states)
resources = 10



# Function to generate valid combinations
# recursion to maximise run time
def generate_combinations(resources, states, current_combination=None, current_index=0):
    if current_combination is None:
        current_combination = []
    if current_index == states:
        if sum(current_combination) == resources:
            yield np.array(current_combination)
        return

    for i in range(resources + 1):
        yield from generate_combinations(resources, states, current_combination + [i], current_index + 1)
# Create a generator and convert it to a NumPy array
dec = np.array(list(generate_combinations(resources, states)))


#%%
# Prepare payoff matrices
num_actions = len(dec)
payoffH = np.zeros((num_actions, num_actions))
swing_threshold = 50

# memory optimised algo as limited storage
for i in range (num_actions):
    for j in range(i,num_actions):
        if i==j:
           res = 0.5
        else:
            diff_sign = 0.5 + 0.5 * np.sign(np.array(dec[i]) - np.array(dec[j]))
            res = 0.5 + 0.5 * np.sign(np.dot(diff_sign, swing_states) - swing_threshold)
            payoffH[i][j]= 1-res
            payoffH[j][i]= res
payoffT = 0.5 + 0.5 * np.sign(-payoffH + 0.5)
#%%
 #initialize
regretSum=np.zeros(num_actions)
oppRegretSum=np.zeros(num_actions)

strategySum=np.zeros(num_actions)
oppStrategySum=np.zeros(num_actions)


def getStrategy(regretSum,sSum):
    normalizingSum=0
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
         actionUtility=payoffH[:,oppAction]
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
    
# finally train
Train(10000)
Strategies_table=np.zeros((num_actions,states+2))
Strategies_table[:,0]=getAverageStrategy(oppStrategySum)
Strategies_table[:,1]=getAverageStrategy(strategySum)
Strategies_table[:,2:]=dec

proxyT=np.zeros((num_actions,states))
proxyB=np.zeros((num_actions,states))
for i in range(num_actions):
        proxyT[i,:]=dec[i,:]*Strategies_table[i,0]
        proxyB[i,:]=dec[i,:]*Strategies_table[i,1]
    
final_resultT=np.dot(np.ones([1,num_actions]),proxyT)/states+1
final_resultB=np.dot(np.ones([1,num_actions]),proxyB)/states+1
print(final_resultT)
print(final_resultB)
df = pd.DataFrame(Strategies_table)

#%% 
# average only the last [2:] columns of df if column 0 is above its 75th percentile
# Calculate the 75th percentile of the first column
percentile = np.percentile(df.iloc[:, 0], 90)

# Filter rows where the first column is above the 75th percentile
filtered_df = df[df.iloc[:, 0] > percentile]

# Weighted average of filtered DataFrame
weighted_average = (filtered_df.iloc[:, 2:].T * filtered_df.iloc[:, 0]).T.sum() / filtered_df.iloc[:, 0].sum()
weighted_average.index = ['Arizona', 'Georgia', 'Pennsylvania', 'North Carolina', 'Wisconsin', 'Michigan', 'Nevada', 'New Hampshire+neb+maine']
print(round(weighted_average*10)) # as want to allocate 100 resources
