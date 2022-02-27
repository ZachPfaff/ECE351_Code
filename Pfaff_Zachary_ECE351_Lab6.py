################
#              #
# Zach Pfaff   #
# ECE 351      #
# Lab 6        #
# Feb 22, 2022 #
#              #
################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 0.001
t = np.arange(0, 2 + steps, steps)


#Code for Step Function
def step(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    return y

#PART 1
y = ((1/2)+np.exp(-6*t) - (1/2)*np.exp(-4*t)) * step(t)

num1 = [1, 6, 12]
den1 = [1, 10, 24]
den2 = [1, 10, 24]

#step response
yout, xout = sig.step((num1, den1), T = t)

#partial fraction expansion
r1, s1, p1 = sig.residue(num1, den2)

print(r1)
print(s1)

#Part 1 graphs
plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(t, y)
plt.title('hand calculated h(t)')
plt.ylabel('y')
plt.grid(True)

plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(yout, xout)
plt.title('Step() Function')
plt.ylabel('y')
plt.grid(True)

#PART 2
t2 = np.arange(0, 4.5 + steps, steps)

num2 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250, 0]
den4 = [1, 18, 218, 2036, 9085, 25250]

r2, s2, p2 = sig.residue(num2, den3)
print(r2)
print(s2)

def cosine(r2, s2, t2):
    y = np.zeros((len(t2)))
    for i in range(len(r2)):
            y += (2 * abs(r2[i]) * np.exp(np.real(s2[i]) * t2) 
                  * np.cos(np.imag(s2[i]) * t2 + np.angle(r2[i]))) * step(t2)
        
    return y

y2 = cosine(r2, s2, t2)

yout2, xout2 = sig.step((num2, den4), T = t)

#Part 2 graphs
plt.figure(figsize = (12 ,8))
plt.subplot (1,1,1)
plt.plot(t2, y2)
plt.title('Cosine Method')
plt.ylabel('f1')
plt.grid(True)

plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(yout2, xout2)
plt.title('Step() Function')
plt.ylabel('y')
plt.grid(True)