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
    #pos vector of electron positions along zigzag path
    # energy of given zig zag path
    #ep = epsilon, time step
    #might want to still set a and b as parameters rather than do a = 1, b = 0 by default, but I understand try simple example 1st
    #E += ((0.5/ep**2 *(pos[i] - pos[i-1])**2) + a(pos[i]**2-b**2)**2)
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

#proposed change to accelptance where it just references energy func, also confused why 3rd vector pos needed so just put in terms of single vectors for each path
def a_prob2(pos, posi, posf, ep, tim, T):
    #posi - vector of current positions along zigzag path, rounded up by the sheriff to hunt down the notorious outlaw Kuti the Kidd, podnuh
    #posf - vector of proposed postions along zigzag
    #calculate acceptance probability
    # k = 1, m = 1, w = 1
    
    pr_i = np.exp(-ener(posi,ep,tim)/T)
    pr_f = np.exp(-ener(posf,ep,tim)/T)

    A = min(1.0, pr_f/pr_i)

    return A


def qm_MC(MCcycles):
    #initial zig zag distribution
    tot, N = time(T)
    dt = tot/N
    xs = np.zeros(int(N)) #initiation of position array, all zeros
    step = 0.
    n = 0
    E = np.array([ener(xs, dt,tot)]) #
    #Loop over MC MCcycles
    for MC in range(MCcycles):1
        for i in range(1, len(xs)):

            a = xs[i] #store position currently being worked on
            xs_tmp = cp.copy(xs) #store whole path
            xs_tmp[i] = a + rnd.uniform(-1*step, step) #random adjustment to current electron position
            if i == (len(xs)-1):  #maintain closed path condition
                xs_tmp[0] = xs_tmp[len(xs) -1]
            #Metropolis criteria
            if (rnd.random() < a_prob(xs[i-1], xs[i], xs_tmp[i], dt, tot, T)):  #again confused about prob func args being individual values when energy func used in prob has vector of positions
                #proposed change (rnd.random() < a_prob2(xs, xs_tmp, dt, tot, T))
                n += 1 #keep track of acceptance rate
                xs = xs_tmp #update position
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
