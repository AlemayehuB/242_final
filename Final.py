import numpy as np
import matplotlib.pyplot as plt
import os


def derivative(r1,r2,dt):
    return (r2-r1)/dt

#----------------PROBLEM SETUP--------------
def V(r):
    '''
    harmonic oscillator potential
    '''
    return a*(r**2 - b**2)**2

def T(rdot):
    '''
    kinetic energy
    '''
    return 0.5 * rdot**2

def H(r,rdot):
    '''
    Hamiltonian
    '''
    return T(rdot) + V(r)

def Boltzmann(r,rdot,temperature):
    '''
    numerical weight of a given state for system following Boltzmann distribution (un-normalized)
    '''
    return np.exp( -1*H(r,rdot)/temperature )

#---------------PLOTTING------------------------

def plot_random_walk(rlist,temperature):
    plt.figure(figsize=(10,10))

    for i in range(len(rlist)):
        plt.scatter(rlist[i],np.full_like(rlist[i],i),color='k',s = 10,alpha = 0.1)

    plt.xlabel('x',fontsize=30)
    plt.ylabel('time',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.savefig('MCMC_double_well.png')
    plt.close()

def plot_distribution(r):
    plt.figure(figsize=(10,10))
    plt.hist(r,bins=20,density=True)
    plt.xlabel('x',fontsize=30)
    plt.ylabel('PDF',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('pdf_double_well.png',bbox_inches='tight')

    
#---------------MONTE CARLO STUFF------------------------

def InitializeSystem(rbounds,N):
    '''
    Set positions
    '''
    return np.linspace(rbounds[0],rbounds[1],N)

def GetTrialCoordinates(r,dr):
    '''
    Randomly generate new coordinates dr away from current position
    '''
    r_ = r + (-1)**np.random.randint(2,size=len(r))*dr 

    # if new coordinates are out of bounds, r_new = r_old
    r_[rbounds[1] <= r_] = r[rbounds[1] <= r_]
    r_[r_ <= rbounds[0]] = r[r_ <= rbounds[0]]

    return r_

def AcceptCoordinates(r1,r2,temperature):
    '''
    whether or not proposed step is accepted. Metropolis algorithm
    '''
    rdot = derivative(r1,r2,epsilon)
    accept_list = Boltzmann(r2,rdot,temperature) >= Boltzmann(r1,rdot,temperature)
    false_idx = np.nonzero( ~accept_list )[0]
    #If new state is less probable than old one, roll the dice
    if false_idx.size > 0:
        accept_list[false_idx][np.random.rand(len(false_idx)) < \
        Boltzmann(r2[false_idx],rdot[false_idx],temperature)/Boltzmann(r1[false_idx],rdot[false_idx],temperature)] \
        = True

    return accept_list

def UpdateCoordinates(r,temperature,dr):
    r_ = GetTrialCoordinates(r,dr)
    accept_list = AcceptCoordinates(r,r_,temperature)
    false_idx = np.nonzero( ~accept_list[0] )[0]

    # Get new coordinates for all trial states that were declined
    # repeat process until all states are accepted
    while false_idx.size > 0:
        r_f = GetTrialCoordinates(r_[false_idx],dr)

        accept_list[false_idx] = AcceptCoordinates(r[false_idx],r_f,temperature)
        false_idx = np.nonzero( ~accept_list )[0]

    return r_

#---------------SIMULATION MAIN--------------------

def SimulationStart(temperature,N=100,dr=1, Nsteps=1000, plot_paths = False):
    r = InitializeSystem(rbounds,N=N)
    rlist = []
    Elist = []

    for i in range(Nsteps):
        if i % 100 == 0: 
            print(f'{i}/{Nsteps}')
        rlist.append(r)

        r_ = UpdateCoordinates(r,temperature,dr)

        rdot = derivative(r,r_,epsilon)
        Elist.append(H(r,rdot))

        r=r_

        #print(np.mean(Elist))
    Elist = np.array(Elist)
    plot_distribution(r)
    
    if plot_paths:
        plot_random_walk(rlist,temperature)

    return np.mean(Elist)



if __name__ == '__main__':

    #parameters
    w      = 1
    dr = 0.01
    a,b = 1,1
    Nsteps = 1000
    N=100
    epsilon = 1/N
    temperature = 0.01 * w

    rbounds = [-2,2]

    print('Energy = ',SimulationStart(temperature,N=N,dr=dr, Nsteps=Nsteps))
