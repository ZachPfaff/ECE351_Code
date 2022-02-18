################
#              #
# Zach Pfaff   #
# ECE 351      #
# Lab 5        #
# Feb 15, 2022 #
#              #
################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 0.00001
t = np.arange(0, 1.2e-3 + steps, steps)


#Code for Step Function
def step(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    return y

#Variable declarations
R = 1000
L = 0.027
C = 0.0000001
w = 18581
a = -5000
mg = 192455904
pg = 1.83

#hand calculated function
y = ((mg/w)*np.exp(a*t)*np.sin(w * t + pg)) * step(t)

num = [0, 10000, 0]
den = [1, 10000, 370370370]

#impulse response
tout, yout = sig.impulse((num, den), T = t)

#step response
xout, sout = sig.step((num, den), T = t)


#Part 1 graphs
plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(t, y)
plt.title('hand calculated h(t)')
plt.ylabel('y')
plt.grid(True)

plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(tout, yout)
plt.title('Impulse() Function')
plt.ylabel('y')
plt.grid(True)

#Part 2 graphs
plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(xout, sout)
plt.title('Step Response')
plt.ylabel('f1')
plt.grid(True)