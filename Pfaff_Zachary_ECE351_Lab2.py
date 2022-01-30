#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:20:34 2022

@author: Zach
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

plt.rcParams.update({'font.size': 14})
"""
#Code for Cosine Function
steps = 1e-2
t = np.arange(0, 5 + steps, steps)
print('Number of elements: len(t) = ', len(t), '\nFirst Element: t[0] = ', t[0],
      ' \nLast Element: t[len(t) - 1] = ', t[len(t) - 1])
def cos(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
            y[i] = np.cos(5*t[i])
            
    return y
y = cos(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('COS Function High Resolution')
plt.xlabel('t')
"""


#Code for Step Function
steps = 0.01
t = np.arange(-5, 10, steps)
def step(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    return y

y = step(t)

"""
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('u(t)')
plt.title('Step Function')
plt.xlabel('t')

"""
#Code for Ramp Function
#steps = 0.001
#t = np.arange(-1, 1 + steps, steps)
def ramp(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]

    return y
y = ramp(t)

"""
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('r(t)')
plt.title('Ramp Function')
plt.xlabel('t')
"""

def lab(t):
    y = ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)
    
    return y

y = np.diff(lab(t))

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(y)
plt.grid()
plt.ylabel('y(t)')
plt.title('y = np.diff(lab(t))')
plt.xlabel('t')
