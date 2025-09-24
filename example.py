import numpy as np
import RK6

def fit_power_law(x,y):
    # Take logs
    logx = np.log(x)
    logy = np.log(y)

    # Linear fit: log(y) = log(a) + b*log(x)
    power, _ = np.polyfit(logx, logy, 1)
    return power

#As a simple example, consider the Lorenz attractor.
sigma = 10
rho = 28
beta = 8/3
v = lambda x: np.array([sigma*(x[1] - x[0]), x[0]*(rho -x[2]) - x[1], x[0]*x[1] - beta*x[2]])

x = np.array([1.0,1.0,1.0])

#Integrate a transient to get onto the attractor
t = 10
steps = 512
x = RK6.rk6(x, t, steps, v)

#Do convergence testing to determine if the method is sixth order
t = 2.0
ns = [32, 64, 128, 256, 512]
n_fine = 1024*8 #fine resolution to compare against as truth
numerical_error = np.zeros( len(ns) )
x_fine = RK6.rk6(x,t,n_fine,v)



print("Testing RK6 with no exponentials. This is standard Runge-Kutta.")
for i, n in enumerate(ns):
    xf = RK6.rk6(x,t,n,v)
    numerical_error[i] = np.linalg.norm(xf - x_fine)
power = fit_power_law(ns, numerical_error)
print(f"Numerical errors are {numerical_error}.")
print(f"Fitted power law is {power:.3f}.\n\n")




print("Testing Lawson RK6 taking A to be a diagonal linear operator.")
g = lambda x: np.array([sigma*(x[1]), x[0]*(rho -x[2]), x[0]*x[1]])
A = np.array([-sigma, -1, -beta])
for i, n in enumerate(ns):
    xf = RK6.lawson_rk6(x,t,n,g,A,diagonal=True)
    numerical_error[i] = np.linalg.norm(xf - x_fine)
power = fit_power_law(ns, numerical_error)
print(f"Numerical errors are {numerical_error}.")
print(f"Fitted power law is {power:.3f}.\n\n")




print("Testing Lawson RK6 taking A to be a nondiagonal linear operator.")
g = lambda x: np.array([0, x[0]*(rho -x[2]), x[0]*x[1]])
A = np.array([ [-sigma, sigma, 0], [0,-1,0], [0,0,-beta] ])
for i, n in enumerate(ns):
    xf = RK6.lawson_rk6(x,t,n,g,A,diagonal=False)
    numerical_error[i] = np.linalg.norm(xf - x_fine)
power = fit_power_law(ns, numerical_error)
print(f"Numerical errors are {numerical_error}.")
print(f"Fitted power law is {power:.3f}.\n\n")