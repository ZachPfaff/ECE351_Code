#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:00:53 2022

@author: Zach
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
steps = 1
t = np.arange(0, 20, steps)


#Code for Step Function
def step(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    return y

#code for Ramp Function
def ramp(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]

    return y

f1 = step(t-2) - step(t-9)
f2 = np.exp(-t)*step(t)
f3 = ramp(t-2)*(step(t-2)-step(t-3)) + ramp(4-t)*(step(t-3)-step(t-4))

#Code for Convolution
def my_conv(f1, f2):
   Nf1 = len(f1)    #create new arrays with same length as input signals
   Nf2 = len(f2)
   
   f1Extended = np.append(f1, np.zeros((1, Nf2 - 1)))                                                  
   f2Extended = np.append(f2, np.zeros((1, Nf1 - 1))) 
   
   result = np.zeros(f1Extended.shape)
   for i in range(Nf2):     
       for j in range(Nf1): 
               try: result[i + j] += f1Extended[i] * f2Extended[j]
               except: print(i,j)

   return result


y = my_conv(f1, f2)
c = np.convolve(f1, f3)
z = np.convolve(f2, f3)


x = np.convolve(f1, f2)

plt.figure(figsize =(12 ,8))
plt.subplot (3,1,1)
plt.plot(x)
plt.title('Convolved Functions w/ np.conv')
plt.ylabel('f1 * f2')
plt.grid(True)


plt.subplot (3,1,2)
plt.plot(c)
plt.ylabel('f1 * f3')
plt.grid(which='both')

plt.subplot (3,1,3)
plt.plot(z)
plt.ylabel('f2 * f3') 
plt.grid(which='both')
