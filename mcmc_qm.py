from math import exp, sqrt
import random as rnd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

#k_B = 1, h_bar = 1, a = 1, b = 0
def time(T, dt = 0.5):
    # input: Temp. output: time, N
    tot = 1/T
    return (tot, tot/dt)

def ener(pos, ep,tim, a = 1, b = 0):
    # energy of given zig zag path
    E = 0
    n = len(pos)
    for i in range(1, n):
        E += ((0.5/ep**2 *(pos[i] - pos[i-1])**2) + a*(pos[i]**2 - b**2)**2)
    E = E/tim
    return E

def a_prob(x1, x2, i, ep, tim, T, a = 1, b = 0):
    #calculate acceptance probability
    # k = 1, m = 1, w = 1

    # pr_1 = np.exp(-((0.5/ep *(pos1 - pos)**2) + ep*a*(pos1**2 - b**2)**2)/T)
    # pr_2 = np.exp(-((0.5/ep *(pos2 - pos)**2) + ep*a*(pos2**2 - b**2)**2)/T)
    if i ==(len(x1)-1):
        pr_1 = np.exp(-((0.5/ep *((x1[i] - x1[i-1])**2 + (x1[1] - x1[i])**2)) + ep*a*(x1[i]**2 - b**2)**2)/T)
        pr_2 = np.exp(-((0.5/ep *((x2[i] - x2[i-1])**2 + (x2[1] - x2[i])**2)) + ep*a*(x2[i]**2 - b**2)**2)/T)
    else:
        pr_1 = np.exp(-((0.5/ep *((x1[i] - x1[i-1])**2 + (x1[i+1] - x1[i])**2)) + ep*a*(x1[i]**2 - b**2)**2)/T)
        pr_2 = np.exp(-((0.5/ep *((x2[i] - x2[i-1])**2 + (x2[i+1] - x2[i])**2)) + ep*a*(x2[i]**2 - b**2)**2)/T)
    A = min(1.0, pr_2/pr_1)

    return A

def MC(MCcycles, step = 1, dt = 0.5, a = 1, b = 0):
    ''' Goes through 'MCcyles' iteration and returns (pos on the chain, energy, and acceptance rate)'''
    #initial zig zag distribution
    tot, N = time(T, dt)
    xs = np.zeros(int(N))
    n = 0
    E = np.array([ener(xs, dt,tot, b =b)])
    #Loop over MC MCcycles
    for MC in range(MCcycles):
        for i in range(1, len(xs)):

            a = xs[i]
            xs_tmp = cp.copy(xs)
            xs_tmp[i] = a + rnd.uniform(-1*step, step)
            if i == (len(xs)-1):
                xs_tmp[0] = xs_tmp[len(xs) -1]
            #Metropolis criteria
            if (rnd.random() < a_prob(xs, xs_tmp, i, dt, tot, T,b = b)):
                n += 1
                xs = xs_tmp
        if (MCcycles <  1000) or (MC > 1000):
            E = np.append(E,ener(xs, dt, tot, b = b ))
    ar = n/((MCcycles)*len(xs))

    return (xs, E, ar)


def qm_MC(MCcycles, step = 1, dt = 0.5, a = 1, b = 0):
    '''Runs Monte Carlo code until optimum acceptance rate reached'''

    accept = False
    xs , E, ar = MC(MCcycles, step = step, dt = dt, b = b)
    tot, N = time(T, dt)
    while accept == False:
        print("acceptance rate: " + str(ar))
        print("step size: "+ str(step))
        print("Total time: " + str(tot) + " , N:" + str(N) + " dt:" + str(dt))
        if (ar < 0.45):
            step = step - 0.01
            xs, E, ar = MC(MCcycles, step = step , dt = dt, b = b)

        elif (ar > 0.55):
            step = step + 0.01
            xs, E, ar = MC(MCcycles, step = step, dt = dt, b = b)

        else:
            accept = True
            print("Success")
            print("********************************************************************")
    return(xs,E)


if __name__ == '__main__':
#### Problem 1#################################################################
    # N = 100
    # Temp = np.linspace(0,1,100)
    # Es = np.array([])
    # for i in range(1,len(Temp)):
    #     T = Temp[i]
    #     tot = time(T)[0]
    #     ep = tot/N
    #     print("Temp: "+ str(T))
    #     xs ,E = qm_MC(3000, step =0.2, dt = ep)
    #     Es = np.append(Es, np.mean(E))
    #     print("E:" + str(np.mean(E)))
    # plt.plot(Temp[1:],Es)
    # plt.xlabel("Temp")
    # plt.ylabel("<E>")
    # plt.yscale("log")
    # plt.title("Energy at Zero-Temp Limit")
    # plt.show()

######Zig Zag Path ############################################################
    # T = 0.1
    # t1 =time(T)
    # ep = t1[0]/100
    # xs ,E = qm_MC(3000, step = 0.2, dt = ep, b = 0)
    # plt.plot(xs,np.linspace(0, t1[0],100), 'r')
    # plt.scatter(xs,np.linspace(0,t1[0],100))
    # plt.xlabel("x")
    # plt.ylabel("Time")
    # plt.title("Zig-Zag Path: a = 1, b = 0, T = 0.1")
    # plt.show()
    
##### Problem 2#################################################################
    # T = 0.001
    # xs, E = qm_MC(3000, step = 0.1,dt = 15, b = 1)
    #
    # print(np.mean(E))
    # #plt.plot(xs,np.linspace(0,time(T)[0],int(time(T)[1])))
    # counts, bins = np.histogram(xs, bins = 30,)
    # plt.hist(bins[:-1], bins, weights=counts, label = "Ground State probability" )
    # plt.title("Ground State Probability: b = 1")
    # plt.ylabel('Counts')
    # plt.xlabel('x')
    # plt.xticks(np.arange(-int(max(xs)),int(max(xs))+1,step = 1.0))
    # # plt.show()
    # plt.savefig("prob_hist_b1")

##### Problem 3#################################################################
    # Temp =np.linspace(1,20,20)
    # Es = np.array([])
    # dt = 0.005
    # for i in range(1,len(Temp)):
    #     T = Temp[i]
    #     print("Temp: "+ str(T))
    #     xs ,E = qm_MC(3000, step =0.2, dt = dt/i)
    #     Es = np.append(Es, np.mean(E))
    #     print("E:" + str(np.mean(E)))
    # plt.plot(Temp[1:],Es)
    # plt.title("Average Thermal Energy of Electron vs Temp")
    # plt.xlabel("Temp")
    # plt.ylabel("<E>")
    # plt.savefig("therm_E")
