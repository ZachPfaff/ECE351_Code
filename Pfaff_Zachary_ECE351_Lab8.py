#################
#               #
# Zach Pfaff    #
# ECE 351       #
# Lab 8         #
# March 8, 2022 #
#               #
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 0.01
t = np.arange(0, 20 + steps, steps)
T = 8
w = (2*np.pi)/T
ak = 0

def ak(k):
        y = 0
        return y

def bk(k):
        y = (2/(k * np.pi)) * (1 - np.cos(k * np.pi))
        return y

print("ak(0): ", ak(0))
print("ak(1): ", ak(1))
print("bk(1): ", bk(1))
print("bk(2): ", bk(2))
print("bk(3): ", bk(3))

def fourier(t, n):
    y = 0
    for i in np.arange(1, n + 1):
        y = y + (2/(i * np.pi)) * (1 - np.cos(i * np.pi)) * (np.sin(i * w * t))       
    return y

F1 = fourier(t, 1)
F3 = fourier(t, 3)
F15 = fourier(t, 15)
F50 = fourier(t, 50)
F150 = fourier(t, 150)
F1500 = fourier(t, 1500)

plt.figure(figsize = (12 ,20))
plt.subplot(3, 1, 1)
plt.plot(t, F1)
plt.title("N=1")
plt.ylabel("x")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, F3)
plt.title("N=3")
plt.ylabel("x")
plt.grid()

plt.figure(figsize = (12 ,20))
plt.subplot(3, 1, 1)
plt.plot(t, F15)
plt.title("N=15")
plt.ylabel("x")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, F50)
plt.title("N=50")
plt.ylabel("x")
plt.grid()

plt.figure(figsize = (12 ,20))
plt.subplot(3, 1, 1)
plt.plot(t, F150)
plt.title("N=150")
plt.ylabel("x")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, F1500)
plt.title("N=1500")
plt.ylabel("x")
plt.grid()

