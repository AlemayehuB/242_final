import numpy as np
import matplotlib.pyplot as plt
import os


def derivative(r,dt):
    return np.concatenate( ([0.5*(r[1] - r[-2])], 0.5*(r[2:]-r[:-2]), [0.5*(r[1] - r[-2])]) ) / dt

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
    return np.longdouble(np.exp( -1*H(r,rdot)/temperature ))

#---------------PLOTTING------------------------

def plot_random_walk(rlist,temperature):
    plt.figure(figsize=(10,10))

    for i in range(len(rlist)):
        plt.scatter(rlist[i],np.full_like(rlist[i],i),color='k',s = 10,alpha = 0.1)

    plt.xlabel('x',fontsize=30)
    plt.ylabel('iteration',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.savefig('MCMC_double_well.png')
    plt.close()

def plot_distribution(r,savename='pdf_double_well'):
    plt.figure(figsize=(10,10))
    plt.hist(r,bins=100,density=True)
    r_ = np.linspace(-2*b, 2*b,100)
    plt.plot(r_,V(r_),color='k')
    plt.xlabel('x',fontsize=30)
    #plt.ylabel('PDF',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig(f'{savename}.png',bbox_inches='tight')
    plt.close()

def plot_E_vs_T(Elist,Tlist,suffix='lowtemp'):
    plt.figure(figsize=(10,10))

    plt.plot(Tlist,Elist,c='k')
    plt.xlabel('Temperature',fontsize=30)
    plt.ylabel('<E>',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'E_vs_T_{suffix}.png',bbox_inches='tight')
    plt.close()
    
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
    accept_list = H(r2,r2dot) <= H(r1,r1dot) #Boltzmann(r2,r2dot,temperature) >= Boltzmann(r1,r1dot,temperature)
    false_idx = np.nonzero( ~accept_list )[0]
    #If new state is less probable than old one, roll the dice
    if false_idx.size > 0:
        accept_list[false_idx] = np.random.rand(len(false_idx)) <= \
        Boltzmann(r2[false_idx],r2dot[false_idx],temperature)/Boltzmann(r1[false_idx],r1dot[false_idx],temperature)

        #print(f'acceptance probability = {np.mean(Boltzmann(r2[false_idx],r2dot[false_idx],temperature)/Boltzmann(r1[false_idx],r1dot[false_idx],temperature))}')
    return accept_list

def UpdateCoordinates(r1,temperature):
    r2 = GetTrialCoordinates(r1)

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

    #keep endpoints together (must have closed path)
    #Endpoints can move around still, they just need to be the same
    r2[-1] = r2[0]

    return r2

#---------------SIMULATION MAIN--------------------

def SimulationStart(temperature,temp_iter=0):
    r = InitializeSystem()
    rlist = []
    Elist = []

    for i in range(Nsteps):
        if i % 100 == 0: 
            print(f'{i}/{Nsteps}')
            if plot_pdf:
                plot_distribution(r,savename=f'pdf_evolution_{i:04d}')
        rlist.append(r)

        r_ = UpdateCoordinates(r,temperature)

        rdot = derivative(r_,epsilon)
        Elist.append(H(r,rdot))

        r=r_

        #print(np.mean(Elist))
    Elist = np.array(Elist)

    if plot_pdf:
        plot_distribution(r)
    if plot_paths:
        plot_random_walk(rlist,temperature)

    return np.mean(Elist)


#----------------RUN SIMULATION REPEATEDLY--------------

def E_vs_T(Tlist,Tlimit='low'):
    Elist  = []

    global dt
    global epsilon
    for i,temperature in enumerate(Tlist):
        dt = 1/temperature
        epsilon = dt/N
        
        E = SimulationStart(temperature,temp_iter=i)
        Elist.append(E)
        print(f'<E> = {E}')

    plot_E_vs_T( np.array(Elist),np.array(Tlist),f'{Tlimit}temp' )



if __name__ == '__main__':

    #Global variables
    w      = 1
    a,b = 2,1

    epsilon = 1 #epsilon should be much less than the tunneling time
    x_initial = 0

    # ---------- NUMBER 1 + 2 -----------------

    #temperature = 0.001
    #dt = 1/temperature #total time for each path
    #Nsteps = 1000 #number of timesteps
    #N=100 #number of walkers
    #dr = b/N
    #epsilon = dt/N 

    temperature = 0.01
    epsilon = 1
    Nsteps = 2000
    N = 100000
    dr = 0.01



    plot_pdf = True
    plot_paths = False

    SimulationStart(temperature)

    plot_pdf = False
    plot_paths = False

    #Tlist = np.linspace(0.005,0.05,10)
    #Nsteps = 2000 #number of timesteps
    #N=10000 #number of walkers
    
    #temp_iter = 0
    #E_vs_T(Tlist,'low')

    # ------------- NUMBER 3 ----------------
    #Tlist = np.linspace(1,20,10)
    #E_vs_T(Tlist,'high')