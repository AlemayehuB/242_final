import numpy as np
import matplotlib.pyplot as plt
import os


def derivative(r,dt):
    return np.concatenate( ([0],0.5*(r[2:]-r[:-2])/dt,[0]) )
#----------------PROBLEM SETUP--------------
def V(r):
    '''
    double well potential
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

def InitializeSystem():
    '''
    Set positions
    '''
    return np.full(N,x_initial)

def GetTrialCoordinates(r):
    '''
    Randomly generate new coordinates dr away from current position
    '''
    return r + (-1)**np.random.randint(2,size=len(r))*dr 

def AcceptCoordinates(r1,r2,r1dot,r2dot,temperature):
    '''
    whether or not proposed step is accepted. Metropolis algorithm
    '''
    accept_list = Boltzmann(r2,r2dot,temperature) >= Boltzmann(r1,r1dot,temperature)
    false_idx = np.nonzero( ~accept_list )[0]
    #If new state is less probable than old one, roll the dice
    if false_idx.size > 0:
        accept_list[false_idx] = np.random.rand(len(false_idx)) <= \
        Boltzmann(r2[false_idx],r2dot[false_idx],temperature)/Boltzmann(r1[false_idx],r1dot[false_idx],temperature)

        #print(f'acceptance probability = {np.mean(Boltzmann(r2[false_idx],r2dot[false_idx],temperature)/Boltzmann(r1[false_idx],r1dot[false_idx],temperature))}')
    return accept_list

def UpdateCoordinates(r1,temperature):
    r2 = GetTrialCoordinates(r1)
    #keep endpoints fixed
    r2[0] = r1[0]
    r2[-1] = r1[-1]

    r1dot = derivative(r1,epsilon)
    r2dot = derivative(r2,epsilon)
    accept_list = AcceptCoordinates(r1,r2,r1dot,r2dot,temperature)
    false_idx = np.nonzero( ~accept_list )[0]

    # Get new coordinates for all trial states that were declined
    # repeat process until all states are accepted
    while false_idx.size > 0:
        r2_f = GetTrialCoordinates(r2[false_idx])

        #calculate r2dot using new coordinates
        r2_copy = np.copy(r2)
        r2_copy[false_idx] = r2_f
        r2dot = derivative(r2_copy,epsilon)
        
        accept_list[false_idx] = AcceptCoordinates(r1[false_idx],r2_f,r1dot[false_idx],r2dot[false_idx],temperature)
        false_idx = np.nonzero( ~accept_list )[0]

        #print(accept_list[false_idx])

        r2[accept_list] = r2_copy[accept_list] #update positions

    return r2

#---------------SIMULATION MAIN--------------------

def SimulationStart(temperature):
    r = InitializeSystem()
    rlist = []
    Elist = []

    for i in range(Nsteps):
        if i % 100 == 0: 
            print(f'{i}/{Nsteps}')
        rlist.append(r)

        r_ = UpdateCoordinates(r,temperature)

        rdot = derivative(r_,epsilon)
        Elist.append(H(r,rdot))

        r=r_

        #print(np.mean(Elist))
    Elist = np.array(Elist)
    plot_distribution(r)
    
    if plot_paths:
        plot_random_walk(rlist,temperature)

    return np.mean(Elist)



if __name__ == '__main__':

    #Global variables
    w      = 1
    dr = 0.01
    a,b = 1,1
    Nsteps = 500
    N=10000

    #dt = 1 / b**2 #dt = tunneling time
    epsilon = 1#dt/N #epsilon should be much less than the tunneling time (t_tunnel ~ pi/epsilon)
    temperature = 0.01 * w

    x_initial = 0
    plot_paths = False

    print(f'Energy = {SimulationStart(temperature)}')