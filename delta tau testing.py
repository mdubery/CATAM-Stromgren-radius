import numpy as np
import matplotlib.pyplot as plt

c = 2.998e8 
h = 6.626e-34
k = 1.381e-23
pi = np.pi
sigma = 5.67e-8

a_nu0 = 6.3e-22
nu_0 = 3.29e15
n = 1e6

pc = 3.086 * 10**16

T_s = np.array([25000, 20000, 5800])
L_s = np.array([4 * 10**30, 4 * 10**29, 3.9 * 10**26])
T_i = np.array([5000, 10000, 20000])
alpha = np.array([4.54 * 10**-19, 2.59 * 10**-19, 2.52 * 10**-19])



# get radius from SB law
def get_R_star(L_star, T_star):
    return (np.sqrt(L_star/(4 * pi * sigma * T_star**4))) # m

R_star = np.zeros(3)
for i in range(3):
    R_star[i] = get_R_star(L_s[i], T_s[i])


N_int_r = 20000 #20,000
N_GL = 47

tau_min = -18
#in log space

x_GL, w_GL = np.polynomial.laguerre.laggauss(N_GL)

#define intergrand

def get_x0(i):
    x_0 = (h*nu_0)/(k*T_s[i])
    return x_0

def f_x_IntergralGL(b, x_0, tau_nu0):
    F = (b+x_0)**-1 * ( np.exp(x_0)  - np.exp(-b) )**-1 * np.exp(  -tau_nu0 * (  x_0 / (b+x_0)  ) **3)
    return F

def Intergral_GL(x_0, tau_nu0):
    Sum = 0
    for i in range(N_GL):
        Sum += w_GL[i] * f_x_IntergralGL(x_GL[i], x_0, tau_nu0)
    return Sum

# wrap all this into a single function

def get_C(R_s, alpha):
    C = (2 * pi * R_s**2  *a_nu0 * nu_0**3 / (c**2 * alpha * n))
    return C


def do_intergration(n_star, n_alpha, tau_max, N_int_r):
    
    #initialise
    tau_nu0 = np.logspace(tau_min, tau_max, N_int_r) # logspace to get better resolution at low tau
    r= np.zeros(N_int_r)
    f= np.zeros(N_int_r)
    fminus1 = np.zeros(N_int_r)
    r[0] = R_star[n_star]
    f[0] = 0.999 #start from fully ionised  not1 to avoid first step being infty
    # experiment with this to
    fminus1[0] = 1 - f[0]
    C = get_C(R_star[n_star], alpha[n_alpha]) # to avoid numerical issues with small C
    x_0 = get_x0(n_star)

    # is this the right order 
    for i in range(N_int_r-1):
        delta_tau = tau_nu0[i+1] - tau_nu0[i]
        r[i+1] = r[i] + delta_tau / (a_nu0  * n *(fminus1[i])) # getting runtime error will give infinite first step start with f=1 
        I_GL = Intergral_GL(x_0, tau_nu0[i])
        F = C/r[i+1]**2 * I_GL
        fminus1[i+1] = 2/(2+F+ np.sqrt(F**2 + 4*F)) # rearranged to avoid numerical issues when F is small
        f[i+1] = 1-fminus1[i+1]
    return r, f, tau_nu0


#define f(b)
def f_x_Intergral_SR(b, x_0):
    F = (b+x_0)**2 * ( np.exp(x_0)  - np.exp(-b) )**-1
    return F

# do GL intergral
def Intergral_GL_SR(x_0):
    Sum = 0
    for i in range(N_GL):
        Sum += w_GL[i] * f_x_Intergral_SR(x_GL[i], x_0)
    return Sum

def get_C_SR(n_star):
    C = 8 * pi**2 * R_star[n_star]**2 * (k*T_s[n_star])**3 / (h**3 * c**2)
    return C

def get_Q(n_star):
    x_0 = get_x0(n_star)
    C = get_C_SR(n_star)
    Q = C * Intergral_GL_SR(x_0)
    return Q

Q = np.zeros(3)
for i in range(3):
    Q[i] = get_Q(i)
print("Q values", Q)


def get_r_SR(n_star, n_alpha):
    Q = get_Q(n_star)
    r_1 = ( 3*Q/ ( 4 * pi * alpha[n_alpha] * n**2)) **(1/3)
    return r_1
    

#find nearest tau_nu0 to stromgren radius and plot vertical line there
def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


#ionisation fraction as func of tau

N_trials = 100
r_1 = np.zeros(N_trials)
N_r_trials = np.zeros(N_trials)
#ref_1overN = np.zeros(N_trials)

for i in range(N_trials):
    N_r_trials[i] = 10000+ 1000*(i)
    r, f, tau_nu0 = do_intergration(0 ,0 , 2.5, int(N_r_trials[i]))
    r_1[i] = r[find_nearest(f, 0.5)]
    
    print(i)


error = abs(1 - r_1[N_trials-1]/r_1)*100
oneoverN = error[0]/ N_r_trials * N_r_trials[0]

print(N_r_trials)
plt.figure(figsize=(12, 12))
plt.plot(N_r_trials, error, color = 'blue', label = 'error')
plt.plot(N_r_trials, oneoverN, linestyle = '--', color = 'red', label = '1/N')
plt.dpi = 300
plt.title(r"Error in $r_1$ as a function of Integration steps")
plt.xlabel("Number of Integration steps")
plt.ylabel("Error %")
plt.legend()
plt.xlim(0, 100000)
plt.ylim(0, 0.25)
plt.show()