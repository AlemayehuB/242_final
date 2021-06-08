import numpy as np
import matplotlib.pyplot as plt
import os

#----------------PROBLEM SETUP--------------
def V(r):
    '''
    harmonic oscillator potential
    '''
    return 0.5*np.linalg.norm(r)**2

def T(p):
    '''
    kinetic energy
    '''
    return 0.5*np.linalg.norm(p)**2

def H(r,p):
    '''
    Hamiltonian
    '''
    return T(p) + V(r)

def AnalyticalSolution(Tlist):
    '''
    Analytical function of average energy
    '''
    return 2*Tlist

#-----------------GENERAL CALCULATIONS-------------------

def Boltzmann(r,p,temperature):
    '''
    numerical weight of a given state for system following Boltzmann distribution (un-normalized)
    '''
    return np.exp( -1*H(r,p)/temperature )

#---------------PLOTTING------------------------

def plot_random_walk(rlist,plist,temperature):
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(10,10)

    ax[0].plot(rlist[:,0],rlist[:,1],color='k')
    ax[0].set_xlabel('x',fontsize=30)
    ax[0].set_ylabel('y',fontsize=30)
    ax[0].tick_params(labelsize=20)
    ax[0].axis('equal')
    ax[0].set(xlim=(-20,20),ylim=(-30,30))

    ax[1].plot(plist[:,0],plist[:,1],color='k')
    ax[1].set_xlabel(r'$p_x$',fontsize=30)
    ax[1].set_ylabel(r'$p_y$',fontsize=30)
    ax[1].tick_params(labelsize=20)
    ax[1].axis('equal')
    ax[1].set(xlim=(-20,20),ylim=(-30,30))

    plt.suptitle(f'T={temperature}',fontsize=30)

    plt.tight_layout()
    plt.savefig(f'tracks/random_walk_2D_T_{temperature:03d}.png')
    plt.close()

def plot_E_vs_T(Elist,E2list,Tlist):
    fig,ax = plt.subplots(2,1)
    fig.set_size_inches(10,10)

    ax[0].plot(Tlist, AnalyticalSolution(Tlist),color='red',label=r'$2T$')
    ax[0].plot(Tlist,Elist,color='k')
    ax[0].set_xlabel('T',fontsize=30)
    ax[0].set_ylabel(r'$\epsilon(T)$',fontsize=30)
    ax[0].set_xlim(0,max(Tlist))
    ax[0].tick_params(labelsize=20)

    ax[1].plot(Tlist,E2list,color='k')
    ax[1].set_xlabel('T',fontsize=30)
    ax[1].set_ylabel(r'$\epsilon_2(T)$',fontsize=30)
    ax[1].set_xlim(0,max(Tlist))
    ax[1].tick_params(labelsize=20)

    ax[0].legend(loc='upper left',fontsize=20,frameon=False)

    plt.tight_layout()
    plt.savefig(f'E_E2_vs_T.png',bbox_inches='tight')
    plt.close()
    
#---------------MONTE CARLO STUFF------------------------


def GetTrialCoordinates(r,p,dr=1,dp=1):
    '''
    Randomly generate new coordinates in phase space (dr,dp) away from current position
    '''
    
    if len(r) == 1:
        return r + (-1)**np.random.randint(2)*dr,v + -1**np.random.randint(2)*dv

    elif len(r) == 2:
        theta_r,theta_p = np.random.rand(2) * 2*np.pi
        
        dx,dy = np.cos(theta_r), np.sin(theta_r)
        dx *= dr/(dx**2 + dy**2)
        dy *= dr/(dx**2 + dy**2)

        dpx,dpy = np.cos(theta_p), np.sin(theta_p)
        dpx *= dp/(dpx**2 + dpy**2)
        dpy *= dp/(dpx**2 + dpy**2)     
        return r+np.array([dx,dy]), p+np.array([dpx,dpy])

def AcceptCoordinates(r1,p1,r2,p2,temperature):
    '''
    whether or not proposed step is accepted.   
    '''
    if Boltzmann(r2,p2,temperature) >= Boltzmann(r1,p1,temperature):
        return True
    else:
        
        #If new state is less probable than old one, roll the dice
        #print(Boltzmann(r2,p2,temperature)/Boltzmann(r1,p1,temperature))
        return np.random.rand() < Boltzmann(r2,p2,temperature)/Boltzmann(r1,p1,temperature)

def UpdateCoordinates(r,p,temperature,dr=1,dp=1):
    r_,p_ = GetTrialCoordinates(r,p,dr,dp)
    while not AcceptCoordinates(r,p,r_,p_,temperature):
        r_,p_ = GetTrialCoordinates(r,p,dr,dp)
    
    return r_,p_

#---------------SIMULATION MAIN--------------------

def SimulationStart(temperature,Ndim=2,dr=1,dp=1, Nsteps=10000):
    r,p = np.zeros(Ndim),np.zeros(Ndim)
    rlist,plist = r[:],p[:]

    Elist = []

    for N in range(Nsteps):
        r,p = UpdateCoordinates(r,p,temperature)

        rlist = np.vstack((rlist,r))
        plist = np.vstack((plist,p))
        Elist.append(H(r,p))

    Elist = np.array(Elist)
    plot_random_walk(rlist,plist,temperature)
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
    dr,dp = 1,1
    Nsteps = 10000

    E_vs_T(range(1,25), Nsteps=Nsteps)