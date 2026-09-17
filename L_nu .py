

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.linewidth'] = 1.5

mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.markersize'] = 7

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = 8.27, 5.8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

h = 6.626e-34 
k = 1.381e-23 
nu_0 = 3.29e15


N =1000
nu = np.logspace(13,15.8,N)

def Lum(T):
    L = np.zeros(N)
    for i in range(N):
        L[i] = nu[i]**3 / (np.exp(  (h*nu[i])/(k*T) ) - 1)
    return L

L_1 = Lum(5800)
L_2 = Lum(20000)
L_3 = Lum(25000)


def find_max(L):
    max = 0
    for i in range(N):
        if L[i] > max:
            max = L[i]
    return max

L_1 = L_1/find_max(L_1)
L_2 = L_2/find_max(L_2)
L_3 = L_3/find_max(L_3)

plt.dpi = 300
plt.figure(figsize=(12, 12))
plt.xlabel("Dimensionless Frequency")
plt.ylabel("Normalised Luminosity")
plt.title("Luminosity as a function of frequency")
plt.plot(nu/nu_0, L_1, label = "5800 K")
plt.plot(nu/nu_0, L_2, label = "20000 K")
plt.plot(nu/nu_0, L_3, label = "25000 K")
plt.axvline(1, color = "black", label = r"$\nu_0$")
plt.xlim(0, 1.75)
plt.legend()
plt.show()
