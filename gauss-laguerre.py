import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return x**3/(np.expm1(x))

true = np.pi**4/15

def g(x):
    return np.exp(x)*x**3/(np.expm1(x))

best_N = 1
N_max = 180
err = np.zeros(N_max)


for j in range(1, N_max):
    N = j
    x_LG, w_LG = np.polynomial.laguerre.laggauss(N)
    ans = 0
    for i in range(N):
        ans += w_LG[i]*g(x_LG[i])
    err[j] = (true - ans)/true
    
    if abs(err[j]) < abs(err[best_N]):
        print("for N =", N, " error is ", err[j])
        print("=", ans )
        best_N = j

print("Best N is ", best_N, " with error ", err[best_N])
print("true value is", np.pi**4/15)

plt.plot(range(1, N_max), np.log10(np.abs(err[1:N_max])))
plt.xlabel("Number of points N")
plt.ylabel("Log_10 of abs error")
plt.xlim(0, N_max)
plt.ylim(-16,0)     
plt.title("Error as a function of N")
plt.show()
