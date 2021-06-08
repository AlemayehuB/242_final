from math import exp, sqrt
import random as rnd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

#k_B = 1, h_bar = 1, a = 1, b = 0

T = 0.001
def time(T, dt = 0.5):
    # input: Temp. output: time, N
    tot = 1/T
    return (tot, tot/dt)

def ener(pos, ep,tim):
    # energy of given zig zag path
    E = 0
    n = len(pos)
    for i in range(1, n):
        E += ((0.5/ep**2 *(pos[i] - pos[i-1])**2) + pos[i]**4)
    E = E/tim
    return E

def a_prob(pos, pos1, pos2, ep, tim, T):
    #calculate acceptance probability
    # k = 1, m = 1, w = 1

    pr_1 = np.exp(-((0.5/ep *(pos1 - pos)**2) + ep*(pos1**4))/T)
    pr_2 = np.exp(-((0.5/ep *(pos2 - pos)**2) + ep*(pos2**4))/T)

    A = min(1.0, pr_2/pr_1)

    return A

def qm_MC(MCcycles):

    #initial zig zag distribution
    tot, N = time(T)
    dt = tot/N
    xs = np.zeros(int(N))
    step = 0.
    n = 0
    E = np.array([ener(xs, dt,tot)])
    #Loop over MC MCcycles
    for MC in range(MCcycles):1
        for i in range(1, len(xs)):

            a = xs[i]
            xs_tmp = cp.copy(xs)
            xs_tmp[i] = a + rnd.uniform(-1*step, step)
            if i == (len(xs)-1):
                xs_tmp[0] = xs_tmp[len(xs) -1]
            #Metropolis criteria
            if (rnd.random() < a_prob(xs[i-1], xs[i], xs_tmp[i], dt, tot, T)):
                n += 1
                xs = xs_tmp
        if MC> 1000:
            E = np.append(E,ener(xs, dt, tot))

    print("acceptance rate:" + str(n/((MCcycles-1000)*len(xs))))
    print("Total time:" + str(tot) + " , N:" + str(N) + " dt:" + str(dt))
    print()
    return(xs,E)

xs ,E = qm_MC(5000)
print(np.mean(E))
# plt.plot(xs,np.linspace(0,time(0.1)[0],int(time(0.1)[1])), 'r')
# plt.scatter(xs,np.linspace(0,time(0.1)[0],int(time(0.1)[1])))
# plt.show()
plt.plot(np.linspace(0,len(E),len(E)), E)
plt.show()
