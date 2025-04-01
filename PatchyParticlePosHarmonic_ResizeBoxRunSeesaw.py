import sys

import os
import argparse
import numpy as np
from pyquaternion import Quaternion
import hoomd
import hoomd.md
import gsd.hoomd
import math
import scipy.integrate as integrate
from scipy.optimize import fsolve

"""
Script that takes an equilibrated starting trajectory and apply's a stronger harmonic trap force. 
"""
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu)
#sim = hoomd.Simulation(device = hoomd.device.CPU(), seed = 1234)

kT = 1.0 #the thermal energy is taken to be the unit of energy
sigma = 1.0 #the LJ diameter is taken to be the unit of length
sigma_p = 0.1*sigma #the diameter of the patch particles. set as 1/10 the unit of length
zeta = 1.0 #the drag coefficient is taken to be unity (this is equivalent to defining the unit time tau = 1.0, see next line)
tau = sigma*sigma*zeta/kT

# Parameters in the simulation that will be held constant (and therefore hardcoded in this script include:
epsilon = 100*kT #this large LJ energy ensures near hard-sphere statistics with diameter 2^{1/6}\sigma)
zeta_R = zeta*sigma*sigma #the rotational drag coefficient
radius = 2**(1/6)*sigma/2
radius_p = 2**(1/6)*sigma_p/2

#user_args = hoomd.option.get_user()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--N") #total number of core particles to simulate
parser.add_argument("-theta", "--theta") # seesaw angle
parser.add_argument("-e_p", "--eps_p") #patch interaction strength (in units of kT)
parser.add_argument("-k", "--k") #Harmonic trap energy (units of kT)
parser.add_argument("-delh","--delh") #height of wave
parser.add_argument("-q", "--period") #number of periods of the wave in the box length
parser.add_argument("-even", "--even") #should the patches be evenly distributed?
parser.add_argument("-c", "--phi") #total area fraction of spheres computed as N*\pi*(2^{1/6}\sigma)^2/(4A)
parser.add_argument("-t", "--time") #simulation duration time (in units of ideal self-diffusion time \sigma^2*\zeta/kT)
args = parser.parse_args()

N     = int(args.N)
theta     = float(args.theta)
eps_p = float(args.eps_p)
k = float(args.k)
delh = float(args.delh)
period = float(args.period)
even  = bool(args.even)
phi   = float(args.phi)
time  = float(args.time)

eps_p = eps_p*kT
k = k*kT
delh = delh*sigma
time = time*tau

# write output
log_file = "./logfiles/Seesaw/PosHarmonic2DWavePatchyParticlesSeesaw_N_" + str(N) + "_phi_" + str(phi) + "_theta_" + str(theta) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_q_"+str(period)+ "_even_" + str(even) + "_.gsd"
gsd_file = "./trajectories/Seesaw/PosHarmonic2DWavePatchyParticlesSeesaw_N_" + str(N) + "_phi_" + str(phi) + "_theta_" + str(theta) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_q_"+str(period)+  "_even_" + str(even) + "_.gsd"
restart_file = "./restartfiles/Seesaw/restart_PosHarmonic2DWavePatchyParticlesSeesaw_N_" + str(N) + "_phi_" + str(phi) + "_theta_" + str(theta) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_q_"+str(period)+  "_even_" + str(even) + "_.gsd"
equil_file = "./equilfiles/Seesaw/PosHarmonic2DWavePatchyParticlesSeesaw_N_" + str(N) + "_phi_" + str(phi) + "_theta_" + str(theta) + "_epsP_" + str(eps_p) + "_k_" + str(20.0) + "_delh_" + str(delh) + "_q_"+str(period)+  "_even_" + str(even) + "_.gsd"


def seesaw( sigma, radius, theta): 
    return np.asarray( [(0, radius, 0 ), ((radius**2 - (radius*np.cos(theta*np.pi/180))**2)**(1/2), radius*np.cos(theta*np.pi/180), 0 ), (0, 0 ,radius), (0,0,-radius) ] )



def func(x, delh , q):
    return (1+ delh**2 * q**2 *np.sin(q*x)**2)**(1/2)

def func2(x, delh, period, L):
    return(L**2/x - integrate.quad(func, -x/2, x/2, args=(delh,2*np.pi/x*period))[0])


restart_done= False

if restart_done== False and os.path.isfile(restart_file) == True:

    sim.create_state_from_gsd(restart_file, frame=-1 )
    print('restart from restartfile')

    xyzPatches = seesaw(sigma, radius, theta)



    rigid = hoomd.md.constrain.Rigid()
    rigid.body['C'] = {
            "constituent_types": ['P']*4,
            "positions":xyzPatches,
            "orientations":[(0,0,0,0)]*4,
            "charges":[0]*4,
            "diameters":[sigma_p]*4
            }
    restart_done=True
else:
    sim.create_state_from_gsd(equil_file, frame=-1)

    xyzPatches = seesaw(sigma, radius, theta)

            #else: sys.exit('P-value not supported! 1<= P <=6')

    rigid = hoomd.md.constrain.Rigid()
    rigid.body['C'] = {
            "constituent_types": ['P']*4,
            "positions":xyzPatches,
	    "orientations":[(1,0,0,0)]*4,
	    "charges":[0]*4,
	    "diameters":[sigma_p]*4
            }
    restart_done = True

#print(snapshot.particles.position)
#print(snapshot.particles.typeid)




all = hoomd.filter.All() #all particles are included in this group
groupC = hoomd.filter.Type('C') #Core particles are groupC
groupP = hoomd.filter.Type('P') #Patch particles are groupP
rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))

# define neighbor list
#nl = hoomd.md.nlist.Cell()
nl=hoomd.md.nlist.Cell(buffer = 0.4, exclusions = ['body'])

#WCA Potential between all particles
lj = hoomd.md.pair.LJ(default_r_cut=2.0**(1./6.)*sigma, nlist=nl, mode='shift')
lj.params[('C','C')] = {'sigma': sigma, 'epsilon': epsilon}
lj.r_cut[('C', 'C')] = 2.0**(1./6.)*sigma
lj.params[('C','P')] = {'sigma': (sigma+sigma_p)/2, 'epsilon': epsilon}
lj.r_cut[('C', 'P')] = 2.0**(1./6.)*(sigma+sigma_p)/2
lj.params[('P','P')] = {'sigma': sigma_p, 'epsilon': eps_p}
lj.r_cut[('P', 'P')] = 2.0**(1./6.)*sigma_p*2.5
#lj.set_params(mode='shift')
#lj.pair_coeff.set('C', 'C', epsilon=epsilon, sigma=sigma, r_cut = 2.0**(1./6.)*sigma)
#lj.pair_coeff.set('P', 'C', epsilon=epsilon, sigma=(sigma+sigma_p)/2, r_cut = 2.0**(1/6)*(sigma+sigma_p)/2)
#lj.pair_coeff.set('P', 'P', epsilon= eps_p, sigma= sigma_p, r_cut=2.5*sigma_p)

boxlength = sim.state.box.Lx
#add Harmonic external force
harmonic = hoomd.md.external.field.Harmonic()
harmonic.params['C'] = dict(k = k, delh=delh, q= 2*np.pi/(boxlength) * period, dim=2)
harmonic.params['P'] = dict(k = 0, delh=0, q=0, dim=2)


# define integrator (brownian dynamics)
time_step = 1e-3*tau #timestep based on ideal self-diffusion time
num_steps = time/time_step
output = 1000.0/time_step #output every 10 units of time (increase to reduce trajectory size)
output_log = 1000.0/time_step #how often thermodynamic properties are logged, every unit of time

integrator = hoomd.md.Integrator(dt=time_step,  integrate_rotational_dof=True)
integrator.rigid = rigid
integrator.forces.append(lj)
integrator.forces.append(harmonic)
#hoomd.md.integrate.mode_standard(dt=time_step, aniso=True)
lg = hoomd.md.methods.Langevin(filter=rigid_centers_and_free, kT=kT) #underdamped dynamics
#bd.gamma('C', zeta) #set translational drag coefficient
#bd.gamma_r('C', zeta_R) #set rotational drag coefficient
#bd.gamma('P', zeta) #set translational drag coefficient
#bd.gamma_r('P', zeta_R) #set rotational drag coefficient

integrator.methods.append(lg)
sim.operations.integrator = integrator
sim.state.thermalize_particle_momenta(filter = rigid_centers_and_free, kT = kT)


thermo_properties = hoomd.md.compute.ThermodynamicQuantities(filter = rigid_centers_and_free)
sim.operations.computes.append(thermo_properties)
logger = hoomd.logging.Logger()
logger.add(thermo_properties)
logger.add(sim, quantities=['timestep', 'walltime', 'tps'])

gsd_writer = hoomd.write.GSD(filename=gsd_file,trigger=hoomd.trigger.Periodic(int(output)),mode='wb',filter=hoomd.filter.All(),dynamic=['property', 'momentum'])
gsd_writer_log = hoomd.write.GSD(filename=log_file, trigger=hoomd.trigger.Periodic(int(output_log)), mode='wb', filter = hoomd.filter.Null())
gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.On(int(num_steps)), mode='wb', filter=hoomd.filter.All(), dynamic=['property', 'momentum'])

#gsd_writer.log = logger
sim.operations.writers.append(gsd_writer_restart)
sim.operations.writers.append(gsd_writer)
sim.operations.writers.append(gsd_writer_log)
gsd_writer_log.log=logger

sim.run(num_steps)
