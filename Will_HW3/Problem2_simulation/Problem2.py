import numpy as np
import matplotlib.pyplot as plt
import os

#----------------PROBLEM SETUP--------------
def E(n):
    return n+0.5

def AnalyticalSolution(Tlist):
    '''
    Analytical function of average energy
    '''
    return 1/(np.exp(1/Tlist)-1) + 0.5
#-----------------GENERAL CALCULATIONS-------------------

def Boltzmann(n,temperature):
    '''
    numerical weight of a given state for system following Boltzmann distribution (un-normalized)
    '''
    return np.exp( -1*E(n)/temperature ) 

#---------------PLOTTING------------------------

def plot_E_vs_T(Elist,E2list,Tlist):
    fig,ax = plt.subplots(2,1)
    fig.set_size_inches(10,10)

    ax[0].plot(Tlist, AnalyticalSolution(Tlist),color='red',label=r'$\frac{1}{e^{1/T}-1} + \frac{1}{2}$')
    ax[0].plot(Tlist,Elist,color='k')
    ax[0].set_xlabel('T',fontsize=30)
    ax[0].set_ylabel(r'$\epsilon(T)$',fontsize=30)
    ax[0].set_xlim(0,max(Tlist))
    ax[0].tick_params(labelsize=20)
    ax[0].legend(loc='upper left',fontsize=20,frameon=False)

    ax[1].plot(Tlist,E2list,color='k')
    ax[1].set_xlabel('T',fontsize=30)
    ax[1].set_ylabel(r'$\epsilon_2(T)$',fontsize=30)
    ax[1].set_xlim(0,max(Tlist))
    ax[1].tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig(f'E_E2_vs_T.png',bbox_inches='tight')
    plt.close()
    
#---------------MONTE CARLO STUFF------------------------


def GetTrialCoordinates(n1):
    '''
    Randomly step up or down an energy state. Stay at n=0 if proposed jump is 0 -> -1
    '''
    n2 = n1 + (-1)**np.random.randint(2)
    if n2 == -1:
        # If trying to go below the minimum energy state, just stay in the same state
        n2 = 0
    return n2

def AcceptCoordinates(n1,n2,temperature):
    '''
    whether or not proposed step is accepted.   
    '''

    if Boltzmann(n2,temperature) >= Boltzmann(n1,temperature):
        return True
    else:
        #If new state is less probable than old one, roll the dice
        return np.random.rand() < Boltzmann(n2,temperature)/Boltzmann(n1,temperature)

def UpdateCoordinates(n,temperature):
    n_ = GetTrialCoordinates(n)
    while not AcceptCoordinates(n,n_,temperature):
        n_ = GetTrialCoordinates(n)
    
    return n_

#---------------SIMULATION MAIN--------------------

def SimulationStart(temperature, Nsteps=10000):
    n = 0
    Elist = []

    for N in range(Nsteps):
        Elist.append(E(n))
        n = UpdateCoordinates(n,temperature)
    Elist = np.array(Elist)
    return np.mean(Elist), np.mean(Elist**2)


#----------------RUN SIMULATION REPEATEDLY--------------

def E_vs_T(Tlist,Nsteps=10000):
    Elist  = []
    E2list = []

    for temperature in Tlist:
        E, E2 = SimulationStart(temperature,Nsteps=Nsteps)
        Elist.append(E)
        E2list.append(E2)
        print(f'E/T = {E/temperature}', f'E^2/T = {E2/temperature}')

    plot_E_vs_T( np.array(Elist),np.array(E2list),np.array(Tlist) )



if __name__ == '__main__':

    #parameters
    hbar   = 1
    m      = 1
    w      = 1
    Nsteps = 100000

    E_vs_T(range(1,25), Nsteps=Nsteps)