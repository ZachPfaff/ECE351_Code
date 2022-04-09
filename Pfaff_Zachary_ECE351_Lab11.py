##################
#                #
# Zach Pfaff     #
# ECE 351        #
# Lab 11         #
# April 5th, 2022#
#                #
##################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches

#steps = 0.1
##t = np.arange(0, 51.1 + steps, steps)

num = [2, -40]
den = [1, -10, 16]

r1, s1, p1 = sig.residue(num, den)

print(r1)

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """ 
    
    # get a figure/plot
    ax = plt.subplot(111)
    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)
    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)
    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()
    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

z, p, k = zplane(num, den)

w, m = sig.freqz(num, den, whole = True)

w = w/np.pi

plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.semilogx(w, 20*np.log10(np.abs(m)))
plt.title("Mag1")
plt.ylabel("x")
plt.grid()
plt.subplot(3, 1, 2)
plt.semilogx(w, np.angle(m))
plt.title("Phase1")
plt.ylabel("x")
plt.grid()


