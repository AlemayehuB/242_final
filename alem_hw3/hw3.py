from math import exp, sqrt
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

num_sims = 20000
step = 2.5
T = 1.0

def ener_cycle(xp):
    #m = 1 , w = 1
    return 0.5 * np.sum(xp**2)

def a_prob(xp1, xp2):
    #calculate acceptance probability
    # k = 1, m = 1, w = 1
    pr_1 = np.exp(-ener_cycle(xp1)/T)
    pr_2 = np.exp(-ener_cycle(xp2)/T)

    A = min(1.0, pr_2/pr_1)

    return A

def MC(MCcycles):
    i = 0
    init = np.array([rnd.randrange(-10,10), rnd.randrange(-10,10), rnd.randrange(-10,10), rnd.randrange(-10,10)])
    curr_xp = init

    E = np.array([])
    E2 = np.array([])
    # x = np.array([curr_xp[0]])
    # p = np.array([curr_xp[1]])
    #Loop over MC MCcycles
    for MC in range(MCcycles):
        new_xp = np.array([curr_xp[0] + step * (rnd.random() - .5), curr_xp[1] + step * (rnd.random() - .5),\
            curr_xp[2] + step * (rnd.random() - .5), curr_xp[3] + step * (rnd.random() - .5)])
        #Metropolis criteria
        if (rnd.random() < a_prob(curr_xp, new_xp)):
            i+=1
            curr_xp = new_xp
            if (MC > 1000):
                E = np.append(E,ener_cycle(curr_xp))
                E2 = np.append(E2,ener_cycle(curr_xp)**2)
                # x = np.append(x, curr_xp[0])
                # p = np.append(p, curr_xp[1])

    print("acceptance rate:" + str(i/num_sims))
    return(E, E2)

# Running Monte Carlo for various Temps
# avg_E = np.array([])
# avg_E2 = np.array([])
# for i in range(1,11):
#     T = i
#     E, E2= MC(num_sims)
#     avg_E = np.append(avg_E,np.sum(E)/len(E))
#     avg_E2 = np.append(avg_E2,np.sum(E2)/len(E2))
#     step = step + 0.6
#
# # Plotting
# fig, ax = plt.subplots(2,1)
# ax[0].plot(np.linspace(1,10,10),avg_E, label = "<E(T)>",linestyle="dotted")
# ax[0].plot(np.linspace(1,10,10), 2*(np.linspace(1,10,10)), label= r"$2k_{B}T$")
# ax[1].plot(np.linspace(1,10,10),avg_E2, label = r"$<E^{2}(T)>$")
# ax[1].plot(np.linspace(1,10,10), 4*(np.linspace(1,10,10)**2), label= r"$4(k_{B}T)^{2}$")
# ax[0].legend()
# ax[1].legend()
# ax[1].set_xlabel("Temp")
# ax[0].set_ylabel("E")
# ax[1].set_ylabel("E")
#
# fig.suptitle("Average <E(T)> and <$E^{2}(T)>$")
#
# # E2,X2,P2 = MC(num_sims)
# # print(np.sum(E)/len(E))
# # print(np.sum(E2)/len(E2))
# # plt.scatter(X,P, label = "Phase Space")
# # plt.plot(np.linspace(1,len(E)+1,len(E)),E, label ="<E>: T= 1.0")
# # plt.plot(np.linspace(1,len(E2)+1,len(E2)),E2, label ="<E>: T= 2.0")
# # plt.legend()
# fig.savefig("cl_ho.png")
# plt.show()
##############################################################################
# Problem 2

T = 0.25
def qm_MC(MCcycles):
    # h = 1, m = 1, w = 1

    # n start
    n = rnd.randrange(0,50)
    print("start: " + str(n))
    step = 1
    a = 0
    ns = np.array([n])
    avg_E = np.array([n+0.5])
    avg_E2 = np.array([(n+0.5)**2])

    for i in range(20000):
        if n == 0:
            new_n = 1
        else:
            new_n = n + rnd.choice([-1, 1])
        A = min(1., np.exp(-((new_n+0.5)/T))/np.exp(-((n+0.5)/T)))
        #print(A)
        if (rnd.random() < A):
            #print(A)
            a += 1
            n = new_n
            #ns = np.append(ns, n)
            if i > 1000:
                avg_E = np.append(avg_E, n + 0.5 )
                avg_E2 = np.append(avg_E2, (n + 0.5)**2)

    print("acceptance rate:" + str(a/num_sims))
    return  (avg_E, avg_E2)
# fig.savefig("cl_ho.png")


# Finding Energy over Temp
E, E2 = qm_MC(num_sims)
avg_E = np.array([np.sum(E)/len(E)])
print("avg_E: " + str(avg_E))
avg_E2 = np.array([np.sum(E2)/len(E2)])

# Specific Heat
q = np.array([(np.sum(E2)/len(E2) - (np.sum(E)/len(E))**2)/T**2])

for i in range(1,16):
    T += 0.25
    E, E2= qm_MC(num_sims)
    print("T:" + str(T)+ "E:" + str(np.sum(E)/len(E)))
    avg_E = np.append(avg_E,np.sum(E)/len(E))
    avg_E2 = np.append(avg_E2,np.sum(E2)/len(E2))
    q = np.append(q, (np.sum(E2)/len(E2) - (np.sum(E)/len(E))**2)/T**2 )

# Plotting
# fig, ax = plt.subplots(2,1)
# ax[0].plot(np.linspace(0.1,1,10),avg_E, label = "<E(T)>",linestyle="dotted")
# ax[0].plot(np.linspace(0.1,1,10), np.linspace(0.1,1,10), label= r"$k_{B}T$")
# ax[1].plot(np.linspace(0.1,1,10),avg_E2, label = r"$<E^{2}(T)>$")
# ax[1].plot(np.linspace(0.1,1,10), 2*(np.linspace(0.1,1,10)**2), label= r"$2(k_{B}T)^{2}$")
# ax[0].legend()
# ax[1].legend()
# ax[1].set_xlabel("Temp")
# ax[0].set_ylabel("E")
# ax[1].set_ylabel("E")
fig2, ax2 = plt.subplots(1,1)
# print(np.mean(q))
ax2.plot(np.linspace(0.25,8.25,16), q, label = "Specific Heat")
ax2.plot(np.linspace(0.25,8.25,16), np.ones(16), label = r"$k_{B}$")
fig2.suptitle("Heat Capacity")
ax2.set_xlabel("T")
ax2.set_ylabel("C")
ax2.legend()
# fig.suptitle("Quantum Electron: Average <E(T)> and <$E^{2}(T)>$")
#fig.savefig("qm_ho.png")

plt.show()
fig2.savefig("hc")
