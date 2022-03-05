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
t = np.arange(0, 10 + steps, steps)


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

#for G(s)
b1 = [1, 9]
a1 = [1, -6, -16]
z1, p1, k1 = sig.tf2zpk(b1, a1)

print('G(s) numerator = ', z1)
print('G(s) denominator = ', p1)

#for A(s)
b2 = [1, 4]
a2 = [1, 4, 3]
z2, p2, k2 = sig.tf2zpk(b2, a2)

print('A(s) numerator = ', z2)
print('A(s) denominator = ', p2)

#for B(s)
x = [1, 26, 168]
v = np.roots(x)
print('B(s) factored = ' ,v)



#Task 5
num = sig.convolve([1 , 9], [1, 4])
print('Numerator = ', num)

den1 = sig.convolve(sig.convolve(sig.convolve(sig.convolve([1, 1], [1, 3]), [1, -8]), [1, 2]),[1, 4])
print('Denominator1 = ', den1)

den2 = sig.convolve(sig.convolve(sig.convolve(sig.convolve([1, 1], [1, 3]), [1, 12]), [1, 9]),[1, 14])
print('Denominator2 = ', den2)

x3 = [2, 41, 500, 2995, 6878, 4344]
v3 = np.roots(x3)
print('roots =', v3)



#plot for part 2
numr = [1, 13, 36]
denr = [2, 41, 500, 2995, 6878, 4344]

tout, yout = sig.step((numr, denr), T = t)

plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(tout, yout)
plt.title('Step Response Closed Loop')
plt.ylabel('y')
plt.grid(True)

#Plot for part 1
den3 = sig.convolve(sig.convolve(sig.convolve([1, -8], [1, 2]), [1, 1]), [1, 3])
print('Open Loop Denominator = ', den3)

numr2 = [1, 9]
denr2 = [1, -2, -37, -82, -48]
tout2, yout2 = sig.step((numr2, denr2), T = t)

plt.figure(figsize =(12 ,8))
plt.subplot (1,1,1)
plt.plot(tout2, yout2)
plt.title('Step Response Open Loop')
plt.ylabel('y')
plt.grid(True)