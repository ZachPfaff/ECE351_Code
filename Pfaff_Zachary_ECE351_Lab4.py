################
#              #
# Zach Pfaff   #
# ECE 351      #
# Lab 4        #
# Feb 15, 2022 #
#              #
################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
steps = 0.01
t = np.arange(-10, 10 + steps, steps)
tbound = np.arange(-20, 2 * t[len(t) - 1] + steps, steps)


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

f1 = np.exp(-2 * t) * (step(t) - step(t - 3))
f2 = step(t - 2) - step(t - 6)
f3 = np.cos(1.57 * t) * step(t) 

h1 = (1/2) * ((((-1 * np.exp(-2 * t) + 1) * step(t)) - (-1 * np.exp(-2 * (t - 3)) + 1) * step(t -  3)))
h2 = (t - 2) * step(t - 2) - (t - 6) * step(t - 6)
h3 = (1/1.57) * np.sin(1.57 * t) * step(t)

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

x = my_conv(f1, step(t))

y = my_conv(f2, step(t))

z = my_conv(f3, step(t))

#Part 1 graphs
plt.figure(figsize =(12 ,8))
plt.subplot (3,1,1)
plt.plot(t, f1)
plt.title('f1, f2, f3')
plt.ylabel('f1')
plt.grid(True)

plt.subplot (3,1,2)
plt.plot(t, f2)
plt.ylabel('f2')
plt.grid(which='both')

plt.subplot (3,1,3)
plt.plot(t, f3)
plt.ylabel('f3') 
plt.grid(which='both')

#Part 2 Task 1 Graphs
plt.figure(figsize =(12 ,8))
plt.subplot (3,1,1)
plt.plot(tbound, x)
plt.title('f1, f2, f3 Convolved w/ Step Function')
plt.ylabel('f1')
plt.grid(True)

plt.subplot (3,1,2)
plt.plot(tbound, y)
plt.ylabel('f2')
plt.grid(which='both')

plt.subplot (3,1,3)
plt.plot(tbound, z)
plt.ylabel('f3') 
plt.grid(which='both')

#Part 2 Task 2 Graphs
plt.figure(figsize =(12 ,8))
plt.subplot (3,1,1)
plt.plot(t, h1)
plt.title('h1, h2, h3')
plt.ylabel('h1')
plt.grid(True)

plt.subplot (3,1,2)
plt.plot(t, h2)
plt.ylabel('h2')
plt.grid(which='both')

plt.subplot (3,1,3)
plt.plot(t, h3)
plt.ylabel('h3') 
plt.grid(which='both')