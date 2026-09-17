import numpy as np
import matplotlib.pyplot as plt

N_GL = 47

nu_0 = 3.29e15
h = 6.626e-34
k = 1.381e-23
pi = np.pi
sigma = 5.67e-8
c = 2.998e8

pc = 3.086e16

n_H = 1e6
T_s = np.array([25000, 20000, 5800])
L_s = np.array([4 * 10**30, 4 * 10**29, 3.9 * 10**26])
alpha = np.array([4.54 * 10**-19, 2.59 * 10**-19, 2.52 * 10**-19])

x_GL, w_GL = np.polynomial.laguerre.laggauss(N_GL)


# get radius from SB law
def get_R_star(L_star, T_star):
    return (np.sqrt(L_star/(4 * pi * sigma * T_star**4))) # m

R_star = np.zeros(3)
for i in range(3):
    R_star[i] = get_R_star(L_s[i], T_s[i])


def get_x0(T_s):
    x_0 = (h*nu_0)/(k*T_s)
    return x_0

def f_x_IntergralGL(b, x_0):
    F = (b+x_0)**2 * ( np.exp(x_0)  - np.exp(-b) )**(-1)
    return F

def Intergral_GL(x_0):
    Sum = 0
    for i in range(N_GL):
        Sum += w_GL[i] * f_x_IntergralGL(x_GL[i], x_0)
    return Sum

def get_C(n_star):
    C = 8 * pi**2 * R_star[n_star]**2 * (k*T_s[n_star])**3 / (h**3 * c**2)
    return C

def get_Q(n_star):
    x_0 = get_x0(T_s[n_star])
    C = get_C(n_star)
    Q = C * Intergral_GL(x_0)
    return Q

def get_r_SR(Q, alpha):
    r_SR = ( 3*Q/ (4 * pi * alpha *n_H**2) )**(1/3)
    return r_SR

Q_s = np.zeros(3)
for i in range(3):
    Q_s[i] = get_Q(i)
    print(Q_s[i], "for T =", T_s[i], "K")

r_SR = np.zeros(3)
for i in range(3):
    r_SR[i] = get_r_SR(Q_s[i], alpha[1])
    print(r_SR[i]/pc, "pc for T =", T_s[i], "K")
