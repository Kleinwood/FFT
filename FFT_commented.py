# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 01:43:32 2022

@author: Grandwood 
All rights reserved
"""

#%% Import packages 导入包
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

#%% Set audio file directory and name 设置音频文件位置及文件名
file_dir = 'K:\\Python\\Audio\\'
file_name = 'AK_right.wav'

#%% Read .wav audio file 读取.wav音频文件

#Note: This program only works for .wav files. Other formats need converting. 注：此程序只能读取.wav文件，其他格式需先转换

#Data as tuple
data = wavfile.read(file_dir+file_name)
#Sample rate in samples/sec
samplerate = data[0]
#Wave data stored in numpy array
wave_data = data[1]
#Duration of the audio
length = wave_data[:,0].size / samplerate #unit: sec

#%% Time domain plot 时域图

time = np.linspace(0., length, wave_data[:,0].size)

plt.figure(dpi=250)
plt.title(file_name+' Time Domain')
plt.plot(time, wave_data[:, 0], label="Left channel")
plt.plot(time, wave_data[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

#%% FFT
# Number of sample points
N = wave_data[:,0].size
# sample spacing
dT = 1.0 / samplerate
x = np.linspace(0.0, N*dT, N, endpoint=False)
y = wave_data[:, 0]
yf = fft(y)
xf = fftfreq(N, dT)[:N//2]

#%%Frequency domain. Log plot 频域图 x对数坐标
plt.figure(dpi=250)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
x_range = [20,20000]   #You can play around with the range 范围可调
plt.title(file_name+' Frequency Domain (log)')
plt.xlim(x_range)
plt.grid()
plt.xscale('log')
plt.show()

#%%Frequency domain. Linear plot 频域图 线性坐标
plt.figure(dpi=250)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
x_range = [20,2000]   #You can play around with the range 范围可调
plt.title(file_name+' Frequency Domain (linear)')
plt.xlim(x_range)
plt.grid()
plt.show()


