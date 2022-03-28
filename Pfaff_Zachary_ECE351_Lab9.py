#################
#               #
# Zach Pfaff    #
# ECE 351       #
# Lab 9         #
# March 22, 2022#
#               #
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

fs = 100
Ts = 1/fs
t = np.arange(0, 2, Ts)

def fft1(x, fs):
    N = len(x) #find the length of the signal
    X_fft = scipy.fftpack.fft(x) #perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) #shift the zero frequency components
                                                  #to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N #fs is sampling frequency and needs to be defined
                                     #previously in code
    X_mag = np.abs(X_fft_shifted)/N #compute magnitudes of signal
    X_phi = np.angle(X_fft_shifted) #compute phases of signal
    return freq, X_mag, X_phi

def fft2(x, fs):
    N = len(x) #find the length of the signal
    X_fft = scipy.fftpack.fft(x) #perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) #shift the zero frequency components
                                                  #to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N #fs is sampling frequency and needs to be defined
                                     #previously in code
    X_mag = np.abs(X_fft_shifted)/N #compute magnitudes of signal
    X_phi = np.angle(X_fft_shifted) #compute phases of signal
    for i in range(len(X_phi)):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
    return freq, X_mag, X_phi

x1_1 = np.cos(2*np.pi*t)
x1_2 = 5*np.sin(2*np.pi*t)
x1_3 = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)**2

f1_1 = fft1(x1_1, fs)
f1_2 = fft1(x1_2, fs)
f1_3 = fft1(x1_3, fs)

f2_1 = fft2(x1_1, fs)
f2_2 = fft2(x1_2, fs)
f2_3 = fft2(x1_3, fs)

#Task 5 Code
T = 8
w = (2*np.pi)/T
t2 = np.arange(0, 16, Ts)
N = 15

def fourier(t2, n):
    y = 0
    for i in np.arange(1, n + 1):
        y = y + (2/(i * np.pi)) * (1 - np.cos(i * np.pi)) * (np.sin(i * w * t2))       
    return y

x2 = fourier(t2, N)

f3 = fft2(x2, fs)

#Task 1 Plots
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_1)
plt.title("Sig 1")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f1_1[0], f1_1[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f1_1[0], f1_1[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f1_1[0], f1_1[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f1_1[0], f1_1[2])
plt.ylabel("Phase")
plt.grid()

#Task 2 Plots
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_2)
plt.title("Sig 2")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f1_2[0], f1_2[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f1_2[0], f1_2[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f1_2[0], f1_2[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f1_2[0], f1_2[2])
plt.ylabel("Phase")
plt.grid()

#Task 3 Plots
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_3)
plt.title("Sig 3")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f1_3[0], f1_3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f1_3[0], f1_3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f1_3[0], f1_3[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f1_3[0], f1_3[2])
plt.ylabel("Phase")
plt.grid()

#------------TASK 4 PLOTS---------------

#plot from task 1
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_1)
plt.title("Sig 1")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f2_1[0], f2_1[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f2_1[0], f2_1[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f2_1[0], f2_1[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f2_1[0], f2_1[2])
plt.ylabel("Phase")
plt.grid()

#plot from task 2
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_2)
plt.title("Sig 2")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f2_2[0], f2_2[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f2_2[0], f2_2[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f2_2[0], f2_2[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f2_2[0], f2_2[2])
plt.ylabel("Phase")
plt.grid()

#plot from task 3
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t, x1_3)
plt.title("Sig 3")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f2_3[0], f2_3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-5, 5])
plt.stem(f2_3[0], f2_3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f2_3[0], f2_3[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-5, 5])
plt.stem(f2_3[0], f2_3[2])
plt.ylabel("Phase")
plt.grid()

#----------------------Task 5 Plot--------------
plt.rcParams.update({'font.size': 14})
plt.figure(figsize = (15 ,15))
plt.subplot(3, 1, 1)
plt.plot(t2, x2)
plt.title("Lab 8 Transform")
plt.ylabel("y")
plt.grid()

plt.subplot(3, 2, 3)
plt.stem(f3[0], f3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.xlim([-3, 3])
plt.stem(f3[0], f3[1])
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(f3[0], f3[2])
plt.ylabel("Phase")
plt.grid()

plt.subplot(3, 2, 6)
plt.xlim([-3, 3])
plt.stem(f3[0], f3[2])
plt.ylabel("Phase")
plt.grid()