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
parser.add_argument("-p", "--P") # number of patches per particle
parser.add_argument("-e_p", "--eps_p") #patch interaction strength (in units of kT)
parser.add_argument("-k", "--k") #Harmonic trap energy (units of kT)
parser.add_argument("-delh","--delh") #height of wave
parser.add_argument("-l", "--wavelength") #wavelength
parser.add_argument("-c", "--phi") #total area fraction of spheres computed as N*\pi*(2^{1/6}\sigma)^2/(4A)
parser.add_argument("-t", "--time") #simulation duration time (in units of ideal self-diffusion time \sigma^2*\zeta/kT)
args = parser.parse_args()

P     = int(args.P)
eps_p = float(args.eps_p)
k = float(args.k)
delh = float(args.delh)
wavelength = float(args.wavelength)
phi   = float(args.phi)
time  = float(args.time)


eps_p = eps_p*kT
k = k*kT
delh = delh*sigma
time = time*tau


def OnePatch(P, sigma, radius): return  np.asarray([(radius,0,0)])
def dimer(P, sigma, radius): return  np.asarray([(0, 0, radius), (0, 0, -radius)])
def trigonalPlanar(P, sigma, radius): return np.asarray( [(0, radius, 0 ), (- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 )   ])
def tetrahedron(P, sigma, radius): return np.asarray( [(0, 0, radius ), (( (8/9)**(0.5))*radius, 0, -1/3*radius ), (-((2/9)**(0.5))*radius, ((2/3)**(0.5))*radius, -1/3*radius), (-((2/9)**(0.5))*radius, -((2/3)**(0.5))*radius, -1/3*radius)    ])
def trigonalBiPyramidal(P, sigma, radius): return np.asarray( [(0, radius, 0 ), (- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 ), (0, 0 ,radius), (0,0,-radius) ] )
def seesaw(P, sigma, radius): return np.asarray( [(- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 ), (0, 0 ,radius), (0,0,-radius) ] )
def octahedron(P, sigma, radius): return np.asarray( [(radius, 0,0), (-radius, 0,0), (0, radius, 0), (0, -radius, 0), (0 ,0 , radius), (0, 0, -radius) ]   )
def doubleTrigonalPlanar(P, sigma, radius): return np.asarray([(0, radius, 0 ), (- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 ), (0, 0, radius ), (- (3**(0.5)) /2*radius, 0, -1/2*radius ), ((3**(0.5)) /2*radius, 0, -1/2*radius)])
def cube( P, sigma, radius):
    L= 2/(3**(0.5))*radius
    return np.asarray([(L/2, L/2, L/2),
                       (L/2, L/2, -L/2 ),
                       (L/2, -L/2, L/2 ),
                       (L/2, -L/2, -L/2 ),
                       (-L/2, L/2, L/2 ),
                       (-L/2, L/2, -L/2 ),
                       (-L/2, -L/2, L/2 ),
                       (-L/2, -L/2, -L/2 )])


def func(x,y, delh , q):
    return (1 + (delh**2 * q**2 *np.sin(q*x)**2)/4 + (delh**2 * q**2 *np.sin(q*y)**2)/4)**(1/2)

def func2(x, delh, period, L):
    return(L**2 - integrate.dblquad(func, -x/2, x/2, -x/2, x/2, args=(delh, 2*np.pi/x*period))[0])


#A_int = (radius**2) * np.arccos((2*radius**2 - radius_p**2)/(2*radius**2)) - (2*radius**2 - radius_p**2)/(2*radius) * np.sqrt(radius**2 - ((2*radius**2 - radius_p**2)/(2*radius))**2) + radius_p**2 *np.arccos((radius_p**2)/(2*radius*radius_p)) - (radius_p**2)/(2*radius)*np.sqrt(radius_p**2 - (radius_p**2/(2*radius))**2)
#A_int = (radius**2) * np.arccos((2*radius**2 - radius_p**2)/(2*radius**2)) +radius_p**2 *np.arccos((radius_p**2)/(2*radius*radius_p)) -1/2*np.sqrt(radius_p**2 * (4*radius**2 - radius_p**2))
#unit_cell = np.sqrt((np.pi*(radius**2) + P*(np.pi*(radius_p**2) - A_int))/phi) #unit cell length dimension for square lattice



#surface area of a cell
SA_cell = integrate.dblquad(func, -wavelength/2, wavelength/2, -wavelength/2, wavelength/2, args=(delh, (2*np.pi)/wavelength ))[0]

n_cell = int(phi*SA_cell /(np.pi*(radius**2)) )# number of particles in a cell

print(SA_cell)
print(n_cell)

# phi_new = int(n_cell)*(np.pi*(radius**2))/SA_cell
# print(phi_new)

N_cell = int(10000/n_cell) # number of cells needed
print(N_cell)

N_cell = int(np.sqrt(N_cell))**2
print('Final N_cell')
print(N_cell)

TotalSurfaceArea = N_cell*SA_cell

num_particles = math.ceil(phi*TotalSurfaceArea)
print('Final num particles')
print(num_particles)

phi_new = num_particles*(np.pi*(radius**2))/(SA_cell*N_cell)
print('Final phi')
print(phi_new) 

Lprime = np.sqrt(N_cell)*wavelength
print('Final Length of the box')
print(Lprime)

unit_cell = np.sqrt((np.pi*(radius**2)  )/(phi_new)) #unit cell length dimension for square lattice


ncopy = math.ceil(np.sqrt(num_particles))

L= ncopy*unit_cell
print('Initial Length of box')
print(L)

# print((num_particles* np.pi*radius**2 )/L**2)
# print((num_particles* np.pi*radius**2  )/ (SA_cell*N_cell))

print('Area Difference')
print(L**2 -  SA_cell*N_cell)
# print(unit_cell)

print('number of periods')
print(Lprime/wavelength)

#print(ncopy)
#print(L)
N = num_particles 






# write output
log_file = "./logfiles/FixLambda/PosHarmonic2DWavePatchyParticlesFixLambda_N_" + str(N) + "_phi_" + str(phi_new) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh)  + "_lambda_" + str(wavelength) +"_.gsd"
gsd_file = "./trajectories/FixLambda/PosHarmonic2DWavePatchyParticlesFixLambda_N_" + str(N) + "_phi_" + str(phi_new) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_lambda_" + str(wavelength) + "_.gsd"
restart_file = "./restartfiles/FixLambda/restart_PosHarmonic2DWavePatchyParticlesFixLambda_N_" + str(N) + "_phi_" + str(phi_new) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_lambda_" + str(wavelength) + "_.gsd"
equil_file = "./equilfiles/FixLambda/PosHarmonic2DWavePatchyParticlesFixLambda_N_" + str(N) + "_phi_" + str(phi_new) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(20.0) + "_delh_" + str(delh) + "_lambda_" + str(wavelength) + "_.gsd"



restart_done= False

if restart_done== False and os.path.isfile(restart_file) == True:

    sim.create_state_from_gsd(restart_file, frame=-1 )
    print('restart from gsdfile')

    if P==1: xyzPatches = OnePatch(P, sigma, radius)
    if P==2: xyzPatches = dimer(P, sigma, radius)
    if P==3: xyzPatches = trigonalPlanar(P, sigma, radius)
    if P==4: xyzPatches = tetrahedron(P, sigma, radius)
    if P==5: xyzPatches = trigonalBiPyramidal(P, sigma, radius)
    if P==6: xyzPatches = octahedron(P, sigma, radius)
    if P==7: xyzPatches = doubleTrigonalPlanar(P, sigma, radius)
    if P==8: xyzPatches = cube(P,sigma, radius)


    rigid = hoomd.md.constrain.Rigid()
    rigid.body['C'] = {
            "constituent_types": ['P']*P,
            "positions":xyzPatches,
            "orientations":[(0,0,0,0)]*P,
            "charges":[0]*P,
            "diameters":[sigma_p]*P
            }
    restart_done=True
else:
    sim.create_state_from_gsd(equil_file, frame=-1)

    if P==1: xyzPatches = OnePatch(P, sigma, radius)
    if P==2: xyzPatches = dimer(P, sigma, radius)
    if P==3: xyzPatches = trigonalPlanar(P, sigma, radius)
    if P==4: xyzPatches = tetrahedron(P, sigma, radius)
    if P==5: xyzPatches = trigonalBiPyramidal(P, sigma, radius)
    if P==6: xyzPatches = octahedron(P, sigma, radius)
    if P==7: xyzPatches = doubleTrigonalPlanar(P, sigma, radius)
    if P==8: xyzPatches = cube(P,sigma, radius)

            #else: sys.exit('P-value not supported! 1<= P <=6')

    rigid = hoomd.md.constrain.Rigid()
    rigid.body['C'] = {
            "constituent_types": ['P']*P,
            "positions":xyzPatches,
	    "orientations":[(1,0,0,0)]*P,
	    "charges":[0]*P,
	    "diameters":[sigma_p]*P
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
nl=hoomd.md.nlist.Cell(buffer = 0.3, exclusions = ['body'])

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
harmonic.params['C'] = dict(k = k, delh=delh, q= 2*np.pi/(wavelength), dim=2)
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


"""
for buffer in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    nl.buffer = buffer
    print(num_steps + sim.timestep)
    sim.run(num_steps)
    print(sim.final_timestep)


    #device.notice(f'buffer={buffer}: TPS={sim.tps:0.3g}, '
    #              f'num_builds={nl.num_builds}')
    print('buffer:' +str(buffer))
    print('TPS:' +str(sim.tps))



"""
thermo_properties = hoomd.md.compute.ThermodynamicQuantities(filter = rigid_centers_and_free)
sim.operations.computes.append(thermo_properties)
logger = hoomd.logging.Logger()
logger.add(thermo_properties)
logger.add(sim, quantities=['timestep', 'walltime', 'tps'])

gsd_writer = hoomd.write.GSD(filename=gsd_file,trigger=hoomd.trigger.Periodic(int(output)),mode='wb',filter=hoomd.filter.All(),dynamic=['property', 'momentum'])
gsd_writer_log = hoomd.write.GSD(filename=log_file, trigger=hoomd.trigger.Periodic(int(output_log)), mode='wb', filter = hoomd.filter.Null())
gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.On(int(num_steps) + int(sim.timestep)), mode='wb', filter=hoomd.filter.All(), dynamic=['property', 'momentum'])

#gsd_writer.log = logger
sim.operations.writers.append(gsd_writer_restart)
sim.operations.writers.append(gsd_writer)
sim.operations.writers.append(gsd_writer_log)
gsd_writer_log.log=logger

sim.run(num_steps)
print('TPS:'+str(sim.tps))

