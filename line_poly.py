# -*- coding: utf-8 -*-

__author__ = 'user'

import numpy

import numpy as np                                                           

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore', np.RankWarning) 

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0,6.0,7.0,8.0,9.0,10.0])

y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0,-0.8,-0.7,-0.2,0.2,0.6])

a3 = np.polyfit(x, y, 3)

print (a3)

a10 = np.polyfit(x, y, 10) 

p3 = np.poly1d(a3)     

print (p3)

p10 = np.poly1d(a10)

xp=np.linspace(0,10,20)


plt.plot(x, y, ".", markersize = 10)

plt.plot(xp, p3(xp), "r--") 

plt.plot(xp, p10(xp), "b--")
plt.show()
