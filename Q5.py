import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.markersize'] = 7

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = 8.27, 5.8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1


c = 2.998 * 10**8 
h = 6.626 * 10**-34 
k = 1.381 * 10**-23 
pi = np.pi
e = np.e
sigma = 5.67 * 10**-8 

a_nu0 = 6.3 * 10**-22
nu_0 = 3.29 * 10**15
n = 10**6

lyr = 9.461 * 10**15 # m

alpha = 2.59 * 10**-19


N_int_r = 100000
N_GL = 47
tau_max =  5
tau_min = -18
#in log space


tau_nu0 = np.linspace(tau_min, tau_max, N_int_r) 

x_GL, w_GL = np.polynomial.laguerre.laggauss(N_GL)


#define intergrand
def f_x_IntergralGL(b, tau_nu0):
    F = 1/(10*b+ 1)**5.4 * np.exp(-tau_nu0 * ( 1/(10*b+1)  )**3)
    return F

def Intergral_GL(tau_nu0):
    Sum = 0
    for i in range(N_GL):
        Sum += w_GL[i]*f_x_IntergralGL(x_GL[i], tau_nu0)
    return Sum




# wrap all this into a single function

def get_C():
    C = a_nu0 * 10 **24  * 10 / ( n * alpha * 4 * pi * h * e ** (0.1)  )
    return C


def do_intergration():
    
    #initialise
    tau_nu0 = np.logspace(tau_min, tau_max, N_int_r) # logspace to get better resolution at low tau
    r= np.zeros(N_int_r)
    f= np.zeros(N_int_r)
    fminus1 = np.zeros(N_int_r)
    r[0] = 10**18 
    f[0] = 0.999 #start from fully ionised  not1 to avoid first step being infty
    # experiment with this to find best
    fminus1[0] = 1 - f[0]
    C = get_C()

    # is this the right order 
    for i in range(N_int_r-1):
        delta_tau = tau_nu0[i+1] - tau_nu0[i]
        r[i+1] = r[i] + delta_tau / (a_nu0  * n *(fminus1[i])) # getting runtime error will give infinite first step start with f=1 
        I_GL = Intergral_GL(tau_nu0[i])
        F = C/r[i+1]**2 * I_GL
        #f[i+1] = (-F + np.sqrt(F**2 +4*F))/2 # take positive root for f>0
        fminus1[i+1] = 2/(2+F+ np.sqrt(F**2 + 4*F)) # rearranged to avoid numerical issues when F is small
        f[i+1] = 1-fminus1[i+1]
    return r, f, tau_nu0




#find nearest tau_nu0 to stromgren radius and plot vertical line there
def find_nearest(array, value):
    array = np.asarray(array)
    i = (np.abs(array - value)).argmin()
    return i




r, f, tau_nu0 = do_intergration()

plt.figure(1)
plt.figure(figsize=(12, 12))
plt.axvline(tau_nu0[find_nearest(f, 0.5)], color = 'red', label=(r"$r_1(\tau_{\nu_0})$"))
plt.plot(tau_nu0, f, color = 'blue', label=(r"$f(\tau_{\nu_0})$"))
#plt.axvline(tau_nu0[find_nearest(r, lyr*30000)], color = 'green', label = 'Radius of Galaxy')
plt.xlim(0, 20000)
plt.ylim(0, 1.1)
plt.xlabel(r"$\tau_{\nu_0}$")
plt.ylabel(r"$f$")
plt.legend()
plt.title("Ionisation fraction as function of optical depth for a Quasar")


print("r_1", r[find_nearest(f, 0.5)]/lyr, "lyr")


plt.figure(2)
plt.figure(figsize=(12, 12))
plt.axvline(r[find_nearest(f, 0.5)]/lyr, color = 'red', label=(r"$r_1$"))
plt.plot(r/lyr, f, color = 'blue', label=(r"$f(r)$"))
plt.axvline(30000, color = 'green', label = 'Radius of Galaxy')
plt.xlim(0, r[N_int_r-1]/lyr)
plt.ylim(0, 1.1)
plt.xlabel(r"$r [lyr]$")
plt.ylabel(r"$f$")
plt.legend()
plt.title("Ionisation fraction as function of radius for a Quasar")
plt.show()

