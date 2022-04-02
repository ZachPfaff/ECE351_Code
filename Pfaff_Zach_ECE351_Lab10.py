#################
#               #
# Zach Pfaff    #
# ECE 351       #
# Lab 10        #
# March 22, 2022#
#               #
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy .signal as sig
import control as con
import math

steps = 0.1
steps2 = 0.000001
t = np.arange(0, 0.01 + steps2, steps2)
w = np.arange(10**3, 10**6 + steps, steps)
r = 1000
l = 0.027
c = 0.0000001

#-----------------Part 1------------------
num = [(1/(r*c)), 0]
den = [1, (1/(r*c)), (1/(l*c))]


mag = 20*np.log10((w/(r*c))/(np.sqrt((w**4) + ((1/(r*c))**2 - (2/(l*c)))*(w**2) + (1/(l*c))**2)))
phase = (np.pi/2) - np.arctan((w/(r*c))/((-(w**2))+(1/(l*c))))

for i in range(len(phase)):
    if  (phase[i]*(180/np.pi)) > 90:
        phase[i] = (phase[i]*(180/np.pi)) - 180
    else:
        phase[i] = phase[i]*(180/np.pi)


h = sig.bode((num, den), w)

sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, deg = True, Plot = True)


plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.semilogx(w, mag)
plt.title("Mag1")
plt.ylabel("x")
plt.grid()
plt.subplot(3, 1, 2)
plt.semilogx(w, phase)
plt.title("Phase1")
plt.ylabel("x")
plt.grid()

plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.semilogx(h[0], h[1])
plt.title("Mag2")
plt.ylabel("x")
plt.grid()
plt.subplot(3, 1, 2)
plt.semilogx(h[0], h[2])
plt.title("Phase2")
plt.ylabel("x")
plt.grid()


#--------------Part 2-----------------------------
f = 1e6
steps2 = 0.000001
t = np.arange(0, 0.01 + steps2, steps2)
xt = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

h1 = sig.bilinear(num, den, f)
yt = sig.lfilter(h1[0], h1[1], xt)

plt.figure(figsize = (15, 10))
plt.plot(t, xt)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('x')

plt.figure(figsize = (15, 10))
plt.plot(t, yt)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('x')