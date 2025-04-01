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
Run this script to generate an equilibrated starting trajectory for a specific kind of curvature.
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
parser.add_argument("-theta", "--theta") # seesaw angle
parser.add_argument("-e_p", "--eps_p") #patch interaction strength (in units of kT)
parser.add_argument("-k", "--k") #Harmonic trap energy (units of kT)
parser.add_argument("-delh","--delh") #height of wave
parser.add_argument("-l", "--wavelength") #wavelength
parser.add_argument("-c", "--phi") #total area fraction of spheres computed as N*\pi*(2^{1/6}\sigma)^2/(4A)
parser.add_argument("-t_e", "--timeEquil") #simulation equilibration time
args = parser.parse_args()

P     = int(args.P)
theta     = float(args.theta)
eps_p = float(args.eps_p)
k = float(args.k)
delh = float(args.delh)
wavelength = float(args.wavelength)
phi   = float(args.phi)
timeEquil = float(args.timeEquil)


eps_p = eps_p*kT
k = k*kT
delh = delh*sigma
timeEquil = timeEquil*tau


def OnePatch(P, sigma, radius): return  np.asarray([(radius,0,0)])
def dimer(P, sigma, radius): return  np.asarray([(0, 0, radius), (0, 0, -radius)])
def trigonalPlanar(P, sigma, radius): return np.asarray( [(0, radius, 0 ), (- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 )   ])
def tetrahedron(P, sigma, radius): return np.asarray( [(0, 0, radius ), (( (8/9)**(0.5))*radius, 0, -1/3*radius ), (-((2/9)**(0.5))*radius, ((2/3)**(0.5))*radius, -1/3*radius), (-((2/9)**(0.5))*radius, -((2/3)**(0.5))*radius, -1/3*radius)    ])
def trigonalBiPyramidal(P, sigma, radius): return np.asarray( [(0, radius, 0 ), (- (3**(0.5)) /2*radius, -1/2*radius, 0 ), ((3**(0.5)) /2*radius, -1/2*radius, 0 ), (0, 0 ,radius), (0,0,-radius) ] )
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
def seesaw( sigma, radius, theta):
    return np.asarray( [(0, radius, 0 ), ((radius**2 - (radius*np.cos(theta*np.pi/180))**2)**(1/2), radius*np.cos(theta*np.pi/180), 0 ), (0, 0 ,radius), (0,0,-radius) ] )


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
num_periods = Lprime/wavelength


#print(ncopy)
#print(L)
N = num_particles 

# write output
#log_file = "./logfiles/PosHarmonicPatchyParticles_N_" + str(N) + "_phi_" + str(phi) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_q_"+str(period)+ "_even_" + str(even) + "_.gsd"
#gsd_file = "./trajectories/PosHarmonicPatchyParticles_N_" + str(N) + "_phi_" + str(phi) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_q_"+str(period)+  "_even_" + str(even) + "_.gsd"
#restart_file = "./restartfiles/restart_PosHarmonicPatchyParticles_N_" + str(N) + "_phi_" + str(phi) + "_P_" + str(P) + "_epsP_" + str(eps_p) + "_k_" + str(k*10) + "_delh_" + str(delh) + "_q_"+str(period)+  "_even_" + str(even) + "_.gsd"
equil_file = "./equilfiles/Seesaw/FixLambda/PosHarmonic2DWavePatchyParticlesSeesawFixLambda_N_" + str(N) + "_phi_" + str(phi_new) + "_P_" + str(P) + "_theta_" + str(theta) + "_epsP_" + str(eps_p) + "_k_" + str(k) + "_delh_" + str(delh) + "_lambda_" + str(wavelength) + "_.gsd"


N_C = int(N) #number of core particles
N_P = P*N_C #number of patch particles
#print(N_C)
#print(N_P)
#print(P)
numPperC = P + 1
#   snapshot = hoomd.data.make_snapshot(N=N_C + N_P, box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L, dimensions=3), particle_types=['C', 'P'] )
#   system = hoomd.init.read_snapshot(snapshot)
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_C
snapshot.configuration.box = [L, L, L, 0, 0, 0]
snapshot.particles.types = ['C', 'P']

#Particle ID's
pid = np.zeros(N_C ) #this array will hold the particle id's. '0' (or 'C') are core particles
body_id = np.zeros(N_C )

#Particle positions
xyz = np.zeros((N_C , 3))
diameter= np.zeros(N_C )

orientation = np.zeros(((N_C ),4))
inertia = np.zeros((N_C , 3))


for i in range(ncopy):
    for j in range(ncopy): #arrange core particles in a cubic lattice
        #first, define rigid center of mass at the center of the unit cell
        particleid = (i + ncopy*j )
        if particleid+1 > N: break
        xyz[particleid, 0] = -1*L/2 + unit_cell/2 + i*unit_cell
        xyz[particleid, 1] = -1*L/2 + unit_cell/2 + j*unit_cell
        #xyz[particleid, 2] = delh*np.cos(2*np.pi/(Lprime) * period * xyz[particleid, 0])
        pid[particleid]  = 0
        diameter[particleid] = radius*2
        central_id = particleid
        body_id[particleid] = central_id
        #orientationval = 2.0 * np.pi * np.random.rand(1)
        #orientation[particleid] = (np.sin(orientationval), np.sin(orientationval), np.sin(orientationval), np.cos(orientationval));
        orientation[particleid] = (1, 0, 0, 0);

if P==1: xyzPatches = OnePatch(P, sigma, radius)
if P==2: xyzPatches = dimer(P, sigma, radius)
if P==3: xyzPatches = trigonalPlanar(P, sigma, radius)
if P==4: xyzPatches = seesaw( sigma, radius, theta)
if P==5: xyzPatches = trigonalBiPyramidal(P, sigma, radius)
if P==6: xyzPatches = octahedron(P, sigma, radius)
if P==7: xyzPatches = doubleTrigonalPlanar(P, sigma, radius)
if P==8: xyzPatches = cube(P,sigma, radius)
        #else: sys.exit('P-value not supported! 1<= P <=6')


#print(xyz)
#print(pid)
snapshot.particles.position = xyz

snapshot.particles.typeid = pid #load particle id's into snapshot

snapshot.particles.diameter = diameter #load particle diameter into snapshot (for visualization purposes)

snapshot.particles.charge = np.zeros(N_C) #set the charge to be 0.0 to neglect electrostatic interactions when invoking the dipole potential

snapshot.particles.moment_inertia = np.ones((3,N_C)) #spherical particles. inertia will be neglected anyways
#snapshot.particles.body = body_id

#np.random.seed(123)
#orientation = 2.0 * np.pi * np.random.rand(snapshot.particles.N) #each particle has a random 2D orientation (angle in radians)
#orient_quat = [(np.sin(orientation[i]), 0.0, 0.0, np.cos(orientation[i])) for i in range(snapshot.particles.N)] #convert the angle to a quaternion, required by hoomd
snapshot.particles.orientation = orientation; #load quaternion into snapshot


init_position = snapshot.particles.position;
#system.restore_snapshot(snapshot)
sim.create_state_from_snapshot(snapshot )
#system.particles.types.add('P') #patch particle
#with gsd.hoomd.open(name = 'Harmonic_initialState.gsd', mode = 'wb') as f:
    #f.append(snapshot)

#os.listdir(".")
#sim.create_state_from_gsd(filename = 'Harmonic_initialState.gsd')

rigid = hoomd.md.constrain.Rigid()
rigid.body['C'] = {
        "constituent_types": ['P']*P,
        "positions":xyzPatches,
        "orientations":[(1,0,0,0)]*P,
        "charges":[0]*P,
        "diameters":[sigma_p]*P
        }
#    rigid.set_param('C', types=['P']*P, positions = xyzPatches[:] );
rigid.create_bodies(sim.state)


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
lj.r_cut[('P', 'P')] = 2.0**(1./6.)*sigma_p
# define integrator (brownian dynamics)
time_step = 5e-5*tau #timestep based on ideal self-diffusion time
num_equil_steps = timeEquil/time_step

integrator = hoomd.md.Integrator(dt=time_step,  integrate_rotational_dof=True)
integrator.rigid = rigid
integrator.forces.append(lj)
#integrator.forces.append(harmonic)
#hoomd.md.integrate.mode_standard(dt=time_step, aniso=True)
lg = hoomd.md.methods.Langevin(filter=rigid_centers_and_free, kT=kT) #underdamped dynamics

integrator.methods.append(lg)
sim.operations.integrator = integrator
sim.state.thermalize_particle_momenta(filter = rigid_centers_and_free, kT = kT)

initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_box.Lx = Lprime
final_box.Ly = Lprime
final_box.Lz = L
box_resize_trigger = hoomd.trigger.Periodic(10)
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=int(num_equil_steps))
box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
output=10/time_step
gsd_writer_equil = hoomd.write.GSD(filename = equil_file, trigger=hoomd.trigger.Periodic(int(output)), mode='xb', filter=hoomd.filter.All(), dynamic=['property'])
sim.operations.writers.append(gsd_writer_equil)
sim.run(num_equil_steps)

#might need to add a harmonic trap at low strap strength as another equilibiration step
#for operation in sim.operations:
#    print(operation)
#sim.operations.remove(integrator)

boxlength = sim.state.box.Lx
num_periods = int(boxlength/wavelength)
#add Harmonic external force
harmonic = hoomd.md.external.field.Harmonic()
harmonic.params['C'] = dict(k = k/2, delh=delh, q= 2*np.pi/(wavelength) , dim=2)
harmonic.params['P'] = dict(k = 0, delh=0, q=0, dim=2)

integrator.forces.append(harmonic)
lj.r_cut[('P', 'P')] = 2.0**(1./6.)*sigma_p*2.5
#sim.operations.add(integrator)

sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k, delh=delh, q= 2*np.pi/wavelength, dim=2)
sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k*2, delh=delh, q= 2*np.pi/(wavelength), dim=2)
sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k*4, delh=delh, q= 2*np.pi/(wavelength) , dim=2)
#gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.Periodic(int(num_steps)), mode='wb', filter=hoomd.filter.All(), dynamic=['property'])
#sim.operations.writers.append(gsd_writer_restart)
sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k*8, delh=delh, q= 2*np.pi/(wavelength), dim=2)
#gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.Periodic(int(num_steps)), mode='wb', filter=hoomd.filter.All(), dynamic=['property'])
#sim.operations.writers.append(gsd_writer_restart)
sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k*16, delh=delh, q= 2*np.pi/(wavelength), dim=2)
#gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.Periodic(int(num_steps)), mode='wb', filter=hoomd.filter.All(), dynamic=['property'])
#sim.operations.writers.append(gsd_writer_restart)
sim.run(num_equil_steps)

harmonic.params['C']= dict(k=k*25, delh=delh, q= 2*np.pi/(wavelength), dim=2)
#gsd_writer_restart = hoomd.write.GSD(filename = restart_file, trigger = hoomd.trigger.Periodic(int(num_steps)), mode='wb', filter=hoomd.filter.All(), dynamic=['property'])
#sim.operations.writers.append(gsd_writer_restart)
sim.run(num_equil_steps)
