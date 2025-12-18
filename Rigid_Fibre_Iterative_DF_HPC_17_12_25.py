# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:45:03 2025

@author:
"""

import numpy as np
import time
import math
import os
import sys
import pickle
import tempfile
import signal
from numba import njit, prange


# Safety parameters
MAX_ACCELERATION = 1000  
WALL_PROXIMITY_THRESHOLD = 5  
MAX_FIBRES = 4


if len(sys.argv) > 1:
    fibre_id = int(sys.argv[1])
    output_dir = sys.argv[2]
else:
    fibre_id = 1
    output_dir = "default_output"


os.makedirs(output_dir, exist_ok=True)

# Atomic saving
def atomic_pickle_dump(obj, path):
    dirn = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix='.tmp_pickle_')
    try:
        with os.fdopen(fd, 'wb') as ftmp:
            pickle.dump(obj, ftmp, protocol=pickle.HIGHEST_PROTOCOL)
            ftmp.flush(); os.fsync(ftmp.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def atomic_save_npz(path, **kwargs):
    dirn = os.path.dirname(path) or '.'
    tmpf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=dirn, suffix='.npz') as tf:
            tmpf = tf.name
        np.savez_compressed(tmpf, **kwargs)
        os.replace(tmpf, path)
    finally:
        if tmpf and os.path.exists(tmpf):
            try: os.remove(tmpf)
            except Exception: pass

def atomic_save_npy(path, arr):
    dirn = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix='.tmp_npy_', suffix='.npy')
    try:
        os.close(fd)
        np.save(tmp, arr)
        # flush to disk
        with open(tmp, 'rb') as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# SIGTERM handler
terminate_requested = False
def _sigterm_handler(signum, frame):
    global terminate_requested
    print(f"Signal {signum} received: will attempt to save and exit.", flush=True)
    terminate_requested = True

signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)   

# Fibre configs
def get_fibre_parameters(fibre_id):
    configurations = [
        # Format: [rhofibre_phys, Lfibre_phys, Dfibre_phys, theta_c]
        [1632.7, 0.0005, 0.000015, 0],           # Fiber 2
        #[1432.0, 0.004232, 0.0009, np.pi/5],    # Fiber 51

    ]

    if 1 <= fibre_id <= len(configurations):
        return configurations[fibre_id-1]
    else:
        print(f"All {MAX_FIBRES} fibres completed. Exiting.")
        sys.exit(0)

# Get parameters for this specific fiber
rhofibre_phys, Lfibre_phys, Dfibre_phys, theta_c = get_fibre_parameters(fibre_id)

print(f"Fibre {fibre_id} parameters:")
print(f"  Density: {rhofibre_phys} kg m")
print(f"  Length: {Lfibre_phys} m")
print(f"  Diameter: {Dfibre_phys} m")
print(f"  Orientation: {theta_c:.4f} rad ({theta_c*180/np.pi:.1f} )")



print(f"Starting simulation for Fiber {fibre_id}")
#print(f"Parameters: L={Lfibre_phys}, D={Dfibre_phys}, rho={rhofibre_phys}, theta={theta_c}")


# timing for efficiency
class Timer:
    def __init__(self, name):
        self.name = name
        self.total_time = 0.0
        self.calls = 0
        self.last_duration = 0.0

    def start(self):
        self._start_time = time.time()

    def stop(self):
        dur = time.time() - self._start_time
        self.last_duration = dur
        self.total_time += dur
        self.calls += 1

    def average_time(self):
        return self.total_time / max(1, self.calls)

    def reset_last(self):
        self.last_duration = 0.0

# Create timers for each major step
timers = {
    'total_iteration': Timer("Total Iteration"),
    'streaming': Timer("Streaming"),
    'boundary_conditions': Timer("Boundary Conditions"),
    'unforced_velocity': Timer("Unforced Velocity"),
    'IB_loop': Timer("IB Loop"),
    'particle_dynamics': Timer("Particle Dynamics"),
    'equilibrium': Timer("Equilibrium Distribution"),
    'collision': Timer("Collision Step"),
    'wall_check': Timer("Wall Proximity Check")
}



# Physical parameters
Vfibre_phys = Lfibre_phys * Dfibre_phys
Lx_phys = 8 * Lfibre_phys
Ly_phys = 16 * Lfibre_phys
rhofluid_phys = 1000

# Simulation parameters 
#dim = 2
#Re = 50
tau = 0.6025
#tol = 1e-20
#l2error = 1
cssq = 1/3
omega = 1/tau
nu = cssq * (tau - 0.5)

# Physical Steps
nu_phys = 1e-6
dx_phys = Dfibre_phys/10
dy_phys = dx_phys
dt_phys = (dx_phys)**2 * nu / nu_phys
drho_phys = rhofluid_phys
dx =1

#print(f"dx = {dx_phys}")

# Lattice size
nx = int(Lx_phys / dx_phys) + 1
ny = int(Ly_phys / dx_phys) + 1


umax = 0.1
umax_phys = umax * dx_phys / dt_phys

# Fibre parameters in lattice units
Lfibre_lat = Lfibre_phys / dx_phys
rhofibre = rhofibre_phys / drho_phys
Vfibre = Vfibre_phys / dx_phys**2
ds_phys = dx_phys
ds = dx
Dfibre_lat = Dfibre_phys / dx_phys
perimeter = 2 * (Lfibre_phys + Dfibre_phys)
Nb = int(round(perimeter / dx_phys))

print(f'Fibre {fibre_id}: Nb = {Nb}, nx = {nx}, ny = {ny}')

# Initial position
x_c = nx // 2
y_c = ny * 0.9

# Initialise arrays
x_b = np.zeros(Nb)
y_b = np.zeros(Nb)
rx = np.zeros(Nb) #relative positions
ry = np.zeros(Nb)
u_b = np.zeros((Nb, 2))
u_des = np.zeros((Nb, 2))

# Arrange 2D Lagrangian points 
for i in range(Nb):
    s = i * ds
    if s < Lfibre_lat:
        rx[i] = s - 0.5 * Lfibre_lat
        ry[i] = -0.5 * Dfibre_lat
    elif s < Lfibre_lat + Dfibre_lat:
        rx[i] = 0.5 * Lfibre_lat
        ry[i] = (s - Lfibre_lat) - 0.5 * Dfibre_lat
    elif s < 2*Lfibre_lat + Dfibre_lat:
        rx[i] = 0.5 * Lfibre_lat - (s - Lfibre_lat - Dfibre_lat)
        ry[i] = 0.5 * Dfibre_lat
    else:
        rx[i] = -0.5 * Lfibre_lat
        ry[i] = 0.5 * Dfibre_lat - (s - 2*Lfibre_lat - Dfibre_lat)

    x_b[i] = x_c + rx[i] * np.cos(theta_c) - ry[i] * np.sin(theta_c)
    y_b[i] = y_c + rx[i] * np.sin(theta_c) + ry[i] * np.cos(theta_c)

# Physics calculations
g = -9.81 * dt_phys**2 / dx_phys
M_p = (rhofibre * Vfibre)
M_f = 1 * Vfibre
I_p = 1/4*M_p*(Dfibre_lat)**2 + 1/12*M_p*(Lfibre_lat)**2
I_f = 1/4*M_f*(Dfibre_lat)**2 + 1/12*M_f*(Lfibre_lat)**2

# Initialize velocities and forces
u_cx, u_cy = 0, 0
u_cx_old, u_cy_old = 0, 0
omega_c, omega_c_old = 0, 0
F_b = np.zeros((Nb, 2))
F_ib = np.zeros((2, nx, ny))
F_grav = (M_p - M_f) * g
F_bsum    = np.zeros((Nb, 2))   
F_ibsum   = np.zeros((2, nx, ny))

#IB iteration
m_max= 5
IB_tol= 1e-10

# D2Q9 lattice arrangement
ndir = 9
halfdir = 5
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cx = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1])
c = np.vstack((cx,cy))
opp = [0,2,1,4,3,7,5,8,6]

# reshape cx, cy for broadcasting: (ndir,1,1)
cx_b = cx[:, None, None]
cy_b = cy[:, None, None]
w_b  = w[:, None, None]

# Simulation initialization
rho = np.ones((nx, ny))
ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))
uxold = np.zeros((nx, ny))
uyold = np.zeros((nx, ny))
feq = np.zeros((ndir, nx, ny))
u_unforced_x = np.zeros((nx, ny))
u_unforced_y = np.zeros((nx, ny))

for k in range(ndir):
    feq[k, :, :] = w[k]

f = np.copy(feq)
fold = np.copy(feq)
fneq = np.copy(feq)

# IB Interpolation functions

@njit(fastmath=True)
def phi4(r):
    ar = abs(r) #dx=1 so not normalised by dx 
    if ar <= 1.0:
        return 0.125 * (3 - 2*ar + math.sqrt(1 + 4*ar - 4*ar*ar))
    elif ar <= 2.0:
        return 0.125 * (5 - 2*ar - math.sqrt(-7 + 12*ar - 4*ar*ar))
    else:
        return 0.0


@njit(parallel=True, fastmath=True)
def interpolate(Nb, x_b, y_b, nx, ny, ux, uy):
    ux_b = np.zeros(Nb)
    uy_b = np.zeros(Nb)
    sum_w = np.zeros(Nb)

    for b in prange(Nb):
        i0 = int(x_b[b])      # floor 
        j0 = int(y_b[b])

        sw  = 0.0
        uxb = 0.0
        uyb = 0.0

        for i in (i0-1, i0, i0+1, i0+2):
            if i < 0 or i >= nx:
                continue
            dx = x_b[b] - i
            wx = phi4(dx)

            for j in (j0-1, j0, j0+1, j0+2):
                if j < 0 or j >= ny:
                    continue
                dy = y_b[b] - j
                w = wx * phi4(dy)

                uxb += ux[i, j] * w
                uyb += uy[i, j] * w
                sw  += w

        if sw > 1e-16:
            ux_b[b] = uxb / sw
            uy_b[b] = uyb / sw
            sum_w[b] = sw

    return ux_b, uy_b, sum_w

@njit(parallel=True, fastmath=True)
def spread(Nb, x_b, y_b, F_bx, F_by, nx, ny, F_ib, ds):
    for b in prange(Nb):
        i0 = int(x_b[b])
        j0 = int(y_b[b])

        for i in (i0-1, i0, i0+1, i0+2):
            if i < 0 or i >= nx:
                continue
            dx = x_b[b] - i
            wx = phi4(dx)

            for j in (j0-1, j0, j0+1, j0+2):
                if j < 0 or j >= ny:
                    continue
                dy = y_b[b] - j
                w = wx * phi4(dy)

                F_ib[0, i, j] += F_bx[b] * w * ds #ds^2 for 3d
                F_ib[1, i, j] += F_by[b] * w * ds

    return F_ib

#Equilibirum and Collision 

@njit(fastmath=True)
def equilibrium(rho, ux, uy, feq, cx_b, cy_b, w_b, cssq):
    udotu = ux*ux + uy*uy        # shape (nx, ny)
    cdotu = cx_b*ux + cy_b*uy    # shape (ndir, nx, ny)
    feq[:] = w_b * rho * (1 + cdotu/cssq + cdotu**2/(2*cssq**2) - udotu/(2*cssq))

@njit(fastmath=True)
def collision(f, fold, feq, ux, uy, F_ib_x, F_ib_y, omega, cx_b, cy_b, w_b, cssq):
    cdotu = cx_b*ux + cy_b*uy
    F_i = (1 - 0.5*omega) * w_b * (
        (cx_b - ux)/cssq * F_ib_x +
        (cy_b - uy)/cssq * F_ib_y +
        (cx_b*F_ib_x + cy_b*F_ib_y) * cdotu/(cssq**2)
    )
    f[:] = (1 - omega)*fold + omega*feq + F_i #dt=1 


# Boundary conditions
def halfway_bounce_back(f, fold, nx, ny):
    # West wall x=0
    fold[2, 0, 1:ny-1] = f[1, 0, 1:ny-1]
    fold[6, 0, 1:ny-1] = f[5, 0, 1:ny-1]
    fold[7, 0, 1:ny-1] = f[8, 0, 1:ny-1]

    # East wall x=nx-1
    fold[1, nx-1, 1:ny-1] = f[2, nx-1, 1:ny-1]
    fold[5, nx-1, 1:ny-1] = f[6, nx-1, 1:ny-1]
    fold[8, nx-1, 1:ny-1] = f[7, nx-1, 1:ny-1]

    # North wall y=ny-1
    fold[4, 1:nx-1, ny-1] = f[3, 1:nx-1, ny-1]
    fold[7, 1:nx-1, ny-1] = f[5, 1:nx-1, ny-1]
    fold[8, 1:nx-1, ny-1] = f[6, 1:nx-1, ny-1]

    # South wall y=0
    fold[3, 1:nx-1, 0] = f[4, 1:nx-1, 0]
    fold[5, 1:nx-1, 0] = f[7, 1:nx-1, 0]
    fold[6, 1:nx-1, 0] = f[8, 1:nx-1, 0]

    # Southwest corner (0,0)
    fold[2, 0, 0] = f[1, 0, 0]   
    fold[3, 0, 0] = f[4, 0, 0]   
    fold[6, 0, 0] = f[5, 0, 0]   

    # Northwest corner (0, ny-1)
    fold[2, 0, ny-1] = f[1, 0, ny-1]  
    fold[4, 0, ny-1] = f[3, 0, ny-1]  
    fold[7, 0, ny-1] = f[8, 0, ny-1]  

    # Southeast corner (nx-1, 0)
    fold[1, nx-1, 0] = f[2, nx-1, 0]  
    fold[3, nx-1, 0] = f[4, nx-1, 0]  
    fold[8, nx-1, 0] = f[7, nx-1, 0]  

    # Northeast corner (nx-1, ny-1)
    fold[1, nx-1, ny-1] = f[2, nx-1, ny-1]  
    fold[4, nx-1, ny-1] = f[3, nx-1, ny-1]  
    fold[5, nx-1, ny-1] = f[6, nx-1, ny-1] 
    

# Data storage for PP

def prepare_timing_diagnostics_for_save(simulation_data):
    timings_np = {k: np.array(v) for k, v in simulation_data['timings'].items()}
    diagnostics_np = {k: np.array(v) for k, v in simulation_data['diagnostics'].items()}
    meta = {'avg_interval': avg_interval, 'save_interval': save_interval, 'nx': nx, 'ny': ny, 'Nb': Nb}
    return timings_np, diagnostics_np, meta

simulation_data = {
    'times_phys': [],
    'u_cy_phys_hist': [],
    'u_cx_phys_hist': [],
    'u_mag_phys_hist': [],
    'F_tot_phys_hist': [],
    'theta_hist': [],
    'fibre_positions': [],
    'velocity_fields': [],
    'velocity_magnitude': [],
    'rho_fields': [],
    'ux_fields': [],
    'uy_fields': [],
    'fibre_id': fibre_id,
    'parameters': {
        'Lfibre_phys': Lfibre_phys,
        'Dfibre_phys': Dfibre_phys,
        'rhofibre_phys': rhofibre_phys,
        'theta_initial': theta_c,
        'nx': nx,
        'ny': ny,
        'dx_phys': dx_phys,
        'dt_phys': dt_phys
    },
    'completion_status': 'completed',
    'stop_reason': 'reached_max_time'
}

avg_interval = 10  # average window = save interval

# arrays to hold averaged time-series
simulation_data['timings'] = {
    'total_iteration': [],
    'streaming': [],
    'boundary_conditions': [],
    'unforced_velocity': [],
    'IB_loop': [],
    'particle_dynamics': [],
    'equilibrium': [],
    'collision': [],
    'data_saving': [],
    'wall_check': []
}

simulation_data['diagnostics'] = {
    'time_phys': [],
    'avg_sum_w': [],
    'min_sum_w': [],
    'max_sum_w': [],
    'vel_error_mean_start': [],
    'vel_error_mean_end': [],
    'vel_error_max_start': [],
    'vel_error_max_end': [],
    'force_norm_start': [],
    'force_norm_end': [],
    'F_norm_diff': [],
    'F_fluid_x': [],
    'F_fluid_y': [],
    'acc_mag': [],
    'acc_ratio': [],
    'Ma_max': [],       
    'Ma_avg': [],
    'ax_tot': [], 
    'ay_tot': []
}

# temporary accumulators (reset every avg_interval steps)
_accum = {k: 0.0 for k in simulation_data['timings'].keys()}
_step_count = 0

# Wall proximity check
def check_wall_proximity(x_c, y_c, x_b, y_b, nx, ny, threshold=WALL_PROXIMITY_THRESHOLD):

    # Check center position
    if (x_c < threshold or x_c > nx - threshold or
        y_c < threshold or y_c > ny - threshold):
        return True

    # Check boundary points
    if (np.any(x_b < threshold) or np.any(x_b > nx - threshold) or
        np.any(y_b < threshold) or np.any(y_b > ny - threshold)):
        return True

    return False

# Main simulation loop
tmax = 200000000
t = 0
save_interval = 10  # Save flow field every 1000 steps to reduce storage

print(f'Starting simulation Loop for fibre {fibre_id}')
print(f'Simulation Time Step {dt_phys}')
print()

start_time = time.time()
simulation_stopped = False
stop_reason = "reached_max_time"

while t < tmax:
  #  print(f"terminate_requested = {terminate_requested}", flush=True)
    timers['total_iteration'].start()

    # Streaming
    timers['streaming'].start()
    for k in range(ndir):
	    fold[k] = np.roll(f[k], shift=(cx[k], cy[k]), axis=(0, 1))
    timers['streaming'].stop()
    


    # Domain BCs
    timers['boundary_conditions'].start()
    halfway_bounce_back(f, fold, nx, ny)
    timers['boundary_conditions'].stop()

    #Immersed Boundary

    # Compute unforced velocity
    timers['unforced_velocity'].start()

    rho = np.sum(fold, axis=0)
    u_unforced_x = np.sum(cx[:, None, None] * fold, axis=0) /rho
    u_unforced_y = np.sum(cy[:, None, None] * fold, axis=0) /rho

    #bug fixing 
    print(f"time step {t}:")
    print(f"min rho: {rho.min():.6e}, max rho: {rho.max():.6e}")

    u_mag = np.sqrt(u_unforced_x**2 + u_unforced_y**2)

    u_max = np.max(u_mag)
    idx = np.unravel_index(np.argmax(u_mag), u_mag.shape)

    print(f"u_max = {u_max:.6e} at cell {idx}")
    print(f"ux, uy at that cell = {ux[idx]:.6e}, {uy[idx]:.6e}")
    print()

    timers['unforced_velocity'].stop()

    #print('rho min/max =', rho.min(), rho.max())

    #loop
    timers['IB_loop'].start()

    F_ibsum.fill(0)  
    F_bsum.fill(0)
    force_norms_inner = []

    vel_diff_start = 0.0
    vel_diff_max_start = 0.0
    vel_diff_end = 0.0      
    vel_diff_max_end = 0.0      
    
    for m in range(m_max):
                  
        #interpolate velocity
        ux_b, uy_b, sum_w = interpolate(Nb, x_b, y_b, nx, ny, u_unforced_x, u_unforced_y)
        u_b[:,0] = ux_b
        u_b[:,1] = uy_b

        err_vec = np.sqrt((u_des[:,0]-u_b[:,0])**2 + (u_des[:,1]-u_b[:,1])**2)
        error = err_vec.max()
        loc_error_max = np.argmax(err_vec)
        if error > 0.1:
            print(f"WARNING: Large IB velocity error = {error:.6e} at iteration {m}, time {t}")
            print("location of max error:", {loc_error_max})
            print("desired velocity:", u_des[loc_error_max])
            print("interpolated velocity:", u_b[loc_error_max])
            print()
        #if m == m_max-1:
           # print(f'error= {error}')
           # print(t)
            
        # Boundary force evaluation
        F_b  = np.zeros((Nb, 2)) 
        F_b[:,0] = 2.0 * 1 *  (u_des[:,0] - ux_b) 
        F_b[:,1] = 2.0 * 1 *  (u_des[:,1] - uy_b) 
        
        #zero_markers = debug_weights(Nb, x_b, y_b, nx, ny)
        #print('zero marker', zero_markers)

        #vel err and diff calc for debugging    
        current_vel_error = np.sqrt((u_des[:,0] - ux_b)**2 + (u_des[:,1] - uy_b)**2)
        current_mean_error = current_vel_error.mean()
        current_max_error = current_vel_error.max()
        
        if m == 0:
            vel_diff_start = current_mean_error
            vel_diff_max_start = current_max_error
        vel_diff_end = current_mean_error
        vel_diff_max_end = current_max_error
            
        # Force Spread 
        F_ib =  np.zeros((2, nx, ny))
        F_ib = spread(Nb, x_b, y_b, F_b[:,0], F_b[:,1], nx, ny, F_ib, ds)
         
        #Update velocity
        u_unforced_x += 1/(2*rho) * F_ib[0]
        u_unforced_y += 1/(2*rho) * F_ib[1]
        
        #bug fixing force convergence
        force_norm = np.sqrt(F_b[:,0]**2 + F_b[:,1]**2).mean()
        force_norms_inner.append(force_norm)
        #print(f"IB iter {m}: avg force = {force_norm:.6e}")
        
        # Check if forces are converging
        if len(force_norms_inner) > 1 and force_norms_inner[-1] > force_norms_inner[0] * 10:
            print("WARNING: Forces diverging in IB iterations")
            print(f"Initial avg force: {force_norms_inner[0]:.6e}, Current avg force: {force_norms_inner[-1]:.6e}")
            if len(force_norms_inner) >= 3:
                print("Recent force norms:", ", ".join(f"{fn:.6e}" for fn in force_norms_inner[-3:]))
            print()

        
        # Total forces per point
        F_ibsum += F_ib
        F_bsum += F_b
        

        if np.max(np.sqrt((u_des[:,0]-u_b[:,0])**2 + (u_des[:,1]-u_b[:,1])**2)) < IB_tol:
            break

        
    timers['IB_loop'].stop()

    #Debug interpolate/spread
    # Total Lagrangian force across all IB iterations
    F_b_total = np.sum(F_bsum, axis=0)
    # Total Eulerian force across all IB iterations
    F_ib_total = np.array([np.sum(F_ibsum[0]), np.sum(F_ibsum[1])])

    # Difference
    diff = F_b_total - F_ib_total

    print(f"Cumulative Lagrangian force: Fx = {F_b_total[0]:.6e}, Fy = {F_b_total[1]:.6e}")
    print(f"Cumulative Eulerian force: Fx = {F_ib_total[0]:.6e}, Fy = {F_ib_total[1]:.6e}")
    print(f"Difference: Fx = {diff[0]:.6e}, Fy = {diff[1]:.6e}")
    print()

    #Fibre dynamics

    timers['particle_dynamics'].start()

    # Total hydrodynamic force on the particle:
    F_fluid_x =  np.sum(F_bsum[:,0])*ds**2
    F_fluid_y =  np.sum(F_bsum[:,1])*ds**2

    #Update to forced velocity Field
    ux = u_unforced_x
    uy = u_unforced_y


    #print(f"F_fluid_x = {F_fluid_x:.6e}, F_fluid_y = {F_fluid_y:.6e}")
    #print(f"M_p = {M_p:.6e}, M_f = {M_f:.6e}")
    #print(f"Acceleration: ax = {F_fluid_x/M_p:.6e}, ay = {F_fluid_y/M_p:.6e}")


    # Calculate accelerations for check
    ax = F_fluid_x/M_p
    ay = F_fluid_y/M_p

    # SAFETY CHECK: Stop if accelerations are unreasonable
    if abs(ax) > MAX_ACCELERATION or abs(ay) > MAX_ACCELERATION:
        print(f"WARNING: Unreasonably large acceleration detected at t={t}!")
        print(f"ax = {ax:.6e}, ay = {ay:.6e} (threshold = {MAX_ACCELERATION})")
        print(f"Stopping fibre {fibre_id} and moving to next fibre.")
        simulation_data['completion_status'] = 'stopped_early'
        simulation_data['stop_reason'] = 'excessive_acceleration'
        try:
            atomic_pickle_dump(simulation_data, os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl'))
            # small npz summary
            last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)
            atomic_save_npz(os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz'),
                            times_phys=np.array(simulation_data['times_phys']),
                            u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
                            u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
                            F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
                            theta_hist=np.array(simulation_data['theta_hist']),
                            fibre_pos_x=(np.array(last_pos[0]) if last_pos[0] is not None else np.array([])),
                            fibre_pos_y=(np.array(last_pos[1]) if last_pos[1] is not None else np.array([])),
                            fibre_id=fibre_id)
            print("Saved checkpoint due to acceleration stop.", flush=True)
        except Exception as e:
            print(f"WARNING: failed to save checkpoint on acceleration stop: {e}", flush=True)
        simulation_stopped = True
        break


    #print('F fluid x', F_fluid_x)
   # print('F fluid y', F_fluid_y)
   # print(f"u_x range: [{ux.min():.6e}, {ux.max():.6e}]")
   # print(f"u_y range: [{uy.min():.6e}, {uy.max():.6e}]")
   # print(f"rho range: [{rho.min():.6e}, {rho.max():.6e}]")



    # Velocities update
    u_cx_old_old = u_cx_old
    u_cx_old = u_cx
    u_cy_old_old = u_cy_old
    u_cy_old = u_cy

    u_cx = u_cx_old - 1/M_p *F_fluid_x + M_f/M_p *(u_cx_old - u_cx_old_old)
    u_cy = u_cy_old - 1/M_p *F_fluid_y + 1/M_p* F_grav + M_f/M_p *(u_cy_old - u_cy_old_old)

    #Torque
    T = -np.sum(rx * F_b[:,1] - ry * F_b[:,0]) * ds**2

    #Angular velocity updates
    omega_c_old_old = omega_c_old
    omega_c_old   = omega_c

    omega_c = omega_c_old + T/I_p + I_f/I_p *(omega_c_old - omega_c_old_old)


    #Central Position
    x_c = x_c +0.5*( u_cx + u_cx_old)
    y_c = y_c +0.5*( u_cy + u_cy_old)

    theta_c = theta_c + 0.5*(omega_c + omega_c_old)

    #Boundary Points
    for b in range(Nb):
        x_b[b] = x_c + rx[b] * np.cos(theta_c) - ry[b] *np.sin(theta_c)
        y_b[b] = y_c + rx[b] * np.sin(theta_c) + ry[b] *np.cos(theta_c)

    # Boundary Velocity update
    for b in range(Nb):
        rx[b] = x_b[b] - x_c
        ry[b] = y_b[b] - y_c
        u_des[b, 0] = u_cx - omega_c * ry[b]
        u_des[b, 1] = u_cy + omega_c * rx[b]


    timers['particle_dynamics'].stop()

    # Equilibrium distribution function
    timers['equilibrium'].start()

    equilibrium(rho, ux, uy, feq, cx_b, cy_b, w_b, cssq)

    #print(f"feq: [{feq.min():.6e}, {feq.max():.6e}]")
    timers['equilibrium'].stop()

    #Collision step
    timers['collision'].start()

    collision(f, fold, feq, ux, uy, F_ibsum[0], F_ibsum[1], omega, cx_b, cy_b, w_b, cssq)

    timers['collision'].stop()

    #Check wall proximity 
    timers['wall_check'].start()

    if check_wall_proximity(x_c, y_c, x_b, y_b, nx, ny):
        print(f"Fibre {fibre_id} too close to wall at t={t}. Stopping simulation and moving to next fibre.")
        simulation_data['completion_status'] = 'stopped_early'
        simulation_data['stop_reason'] = 'wall_proximity'
        try:
            atomic_pickle_dump(simulation_data, os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl'))
            # small npz summary
            last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)
            atomic_save_npz(os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz'),
                            times_phys=np.array(simulation_data['times_phys']),
                            u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
                            u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
                            F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
                            theta_hist=np.array(simulation_data['theta_hist']),
                            fibre_pos_x=(np.array(last_pos[0]) if last_pos[0] is not None else np.array([])),
                            fibre_pos_y=(np.array(last_pos[1]) if last_pos[1] is not None else np.array([])),
                            fibre_id=fibre_id)
            print("Saved checkpoint due to boundary stop.", flush=True)
        except Exception as e:
            print(f"WARNING: failed to save checkpoint on aboundary stop: {e}", flush=True)
        simulation_stopped = True
        break

    timers['wall_check'].stop()

    timers['total_iteration'].stop()



    #timing data
    _accum['total_iteration'] += timers['total_iteration'].last_duration
    for name, timer in timers.items():
        if name != 'total_iteration':
            _accum[name] += timer.last_duration
    _step_count += 1

    # Save averages
    if _step_count >= avg_interval:
        for key in simulation_data['timings'].keys():
            simulation_data['timings'][key].append(_accum[key] / avg_interval)
            _accum[key] = 0.0
        _step_count = 0

    #periodic save
    if t % save_interval == 0:

        # Store data for post-processing
        time_phys = t * dt_phys
        velocity_magnitude = np.sqrt(ux**2 + uy**2) * (dx_phys / dt_phys)
        u_cy_phys = u_cy * (dx_phys / dt_phys)
        u_cx_phys = u_cx * (dx_phys / dt_phys)
        u_mag_phys = np.sqrt(u_cx_phys**2 + u_cy_phys**2) 
        F_tot_x = -F_fluid_x + M_f * (u_cx_old - u_cx_old_old)
        F_tot_y = -F_fluid_y + F_grav + M_f * (u_cy_old - u_cy_old_old)
        F_tot_phys = np.sqrt(F_tot_x**2 + F_tot_y**2) * (drho_phys * dx_phys**2 / dt_phys**2)
        ax_tot = F_tot_x / M_p
        ay_tot = F_tot_y / M_p
        acc_mag = math.sqrt(ax_tot**2 + ay_tot**2)
        cs = np.sqrt(cssq) 
        u_mag = np.sqrt(ux**2 + uy**2) 
        Ma = u_mag / cs
        Ma_max = Ma.max()
        Ma_avg = Ma.mean()
        

        
        F_norm_start = force_norms_inner[0]
        F_norm_end   = force_norms_inner[-1]
        F_norm_diff = abs(force_norms_inner[0] - force_norms_inner[-1]) 
        
        simulation_data['times_phys'].append(time_phys)
        simulation_data['u_cy_phys_hist'].append(u_cy_phys)
        simulation_data['u_cx_phys_hist'].append(u_cx_phys)
        simulation_data['u_mag_phys_hist'].append(u_mag_phys)
        simulation_data['F_tot_phys_hist'].append(F_tot_phys)
        simulation_data['theta_hist'].append(theta_c)
        simulation_data['fibre_positions'].append((x_b.copy(), y_b.copy()))
        simulation_data['velocity_magnitude'].append(velocity_magnitude)

        simulation_data['diagnostics']['time_phys'].append(time_phys)
        simulation_data['diagnostics']['avg_sum_w'].append(np.mean(sum_w))
        simulation_data['diagnostics']['min_sum_w'].append(np.min(sum_w))
        simulation_data['diagnostics']['max_sum_w'].append(np.max(sum_w))
        simulation_data['diagnostics']['vel_error_mean_start'].append(vel_diff_start)
        simulation_data['diagnostics']['vel_error_mean_end'].append(vel_diff_end)
        simulation_data['diagnostics']['vel_error_max_start'].append(vel_diff_max_start)
        simulation_data['diagnostics']['vel_error_max_end'].append(vel_diff_max_end)
        simulation_data['diagnostics']['force_norm_start'].append(F_norm_start)
        simulation_data['diagnostics']['force_norm_end'].append(F_norm_end)
        simulation_data['diagnostics']['F_norm_diff'].append(F_norm_diff)
        simulation_data['diagnostics']['F_fluid_x'].append(F_fluid_x)
        simulation_data['diagnostics']['F_fluid_y'].append(F_fluid_y)
        simulation_data['diagnostics']['ax_tot'].append(ax_tot)
        simulation_data['diagnostics']['ay_tot'].append(ay_tot)
        simulation_data['diagnostics']['acc_mag'].append(acc_mag)
        simulation_data['diagnostics']['acc_ratio'].append(acc_mag / abs(g))
        simulation_data['diagnostics']['Ma_max'].append(Ma_max)
        simulation_data['diagnostics']['Ma_avg'].append(Ma_avg)

        


        # save frames
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # filenames (zero-padded for easy ordering)
        fname_vel = os.path.join(frames_dir, f'vel_{t:08d}.npy')
        fname_ux  = os.path.join(frames_dir, f'ux_{t:08d}.npy')
        fname_uy  = os.path.join(frames_dir, f'uy_{t:08d}.npy')
        fname_rho = os.path.join(frames_dir, f'rho_{t:08d}.npy')

        timings_np, diagnostics_np, meta = prepare_timing_diagnostics_for_save(simulation_data)
        last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)

        try:
            # use atomic saver for each large array
            atomic_save_npy(fname_vel, velocity_magnitude)
            atomic_save_npy(fname_ux, ux.copy() * (dx_phys / dt_phys))
            atomic_save_npy(fname_uy, uy.copy() * (dx_phys / dt_phys))
            atomic_save_npy(fname_rho, rho.copy())
        except Exception as e:
            print(f"WARNING: per-frame save failed at t={t}: {e}", flush=True)

        # Periodic save
        try:
            pkl_path = os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl')
            npz_path = os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz')

            atomic_pickle_dump(simulation_data, pkl_path)

            last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)
            atomic_save_npz(
                npz_path,
                times_phys=np.array(simulation_data['times_phys']),
                u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
                u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
                F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
                theta_hist=np.array(simulation_data['theta_hist']),
                fibre_pos_x=(np.array(last_pos[0]) if last_pos[0] is not None else np.array([])),
                fibre_pos_y=(np.array(last_pos[1]) if last_pos[1] is not None else np.array([])),
                timings=timings_np,
                diagnostics=diagnostics_np,
                timing_meta=meta,
                fibre_id=fibre_id,
                parameters=simulation_data['parameters']
            )
            print(f"Checkpoint + frames saved at t={t} -> {output_dir}", flush=True)
        except Exception as e:
            print(f"WARNING: periodic checkpoint failed at t={t}: {e}", flush=True)




    #print progress
    if t % 1000 == 0:
        elapsed = time.time() - start_time
        print(f'Fibre {fibre_id}: t = {t}, Time = {elapsed:.1f}s, Y-vel = {u_cy_phys:.4f} m/s')

    t += 1



    if terminate_requested:
        simulation_data['completion_status'] = 'terminated_by_signal'
        simulation_data['stop_reason'] = 'SIGTERM_received'
        try:
            atomic_pickle_dump(simulation_data, os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl'))
            last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)
            atomic_save_npz(os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz'),
                            times_phys=np.array(simulation_data['times_phys']),
                            u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
                            u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
                            F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
                            theta_hist=np.array(simulation_data['theta_hist']),
                            fibre_pos_x=(np.array(last_pos[0]) if last_pos[0] is not None else np.array([])),
                            fibre_pos_y=(np.array(last_pos[1]) if last_pos[1] is not None else np.array([])),
                            fibre_id=fibre_id)
            print("Saved checkpoint after SIGTERM; exiting.", flush=True)
        except Exception as e:
            print(f"WARNING: failed to save on SIGTERM: {e}", flush=True)
        sys.exit(0)

if not simulation_stopped and t >= tmax:
    simulation_data['completion_status'] = 'completed'
    simulation_data['stop_reason'] = 'reached_max_time'


output_file = os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl')
try:
    atomic_pickle_dump(simulation_data, output_file)
    # compact summary
    last_pos = simulation_data['fibre_positions'][-1] if simulation_data['fibre_positions'] else (None, None)
    timings_np, diagnostics_np, meta = prepare_timing_diagnostics_for_save(simulation_data)
    atomic_save_npz(
        os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz'),
        times_phys=np.array(simulation_data['times_phys']),
        u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
        u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
        F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
        theta_hist=np.array(simulation_data['theta_hist']),
        velocity_magnitude=np.array(simulation_data.get('velocity_magnitude', [])),
        timings=timings_np,
        diagnostics=diagnostics_np,
        timing_meta=meta,
        fibre_id=fibre_id,
        parameters=simulation_data['parameters'],
        completion_status=simulation_data['completion_status'],
        stop_reason=simulation_data['stop_reason']
    )
except Exception as e:
    print(f"Final save failed: {e}", flush=True)


# Save all data for post-processing
output_file = os.path.join(output_dir, f'fibre_{fibre_id}_data.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(simulation_data, f)

# Also save as numpy arrays for easy access
np.savez_compressed(
    os.path.join(output_dir, f'fibre_{fibre_id}_summary.npz'),
    times_phys=np.array(simulation_data['times_phys']),
    u_cy_phys_hist=np.array(simulation_data['u_cy_phys_hist']),
    u_mag_phys_hist=np.array(simulation_data['u_mag_phys_hist']),
    F_tot_phys_hist=np.array(simulation_data['F_tot_phys_hist']),
    theta_hist=np.array(simulation_data['theta_hist']),
    velocity_magnitude=np.array(simulation_data['velocity_magnitude']),
    rho_fields=np.array(simulation_data['rho_fields']),
    ux_fields=np.array(simulation_data['ux_fields']),
    uy_fields=np.array(simulation_data['uy_fields']),
    fibre_id=fibre_id,
    parameters=simulation_data['parameters'],
    completion_status=simulation_data['completion_status'],
    stop_reason=simulation_data['stop_reason']
)

print(f'Completed fibre {fibre_id}. Data saved to {output_dir}')
print(f'Completion status: {simulation_data["completion_status"]}')
print(f'Stop reason: {simulation_data["stop_reason"]}')
print(f'Total simulation time: {time.time() - start_time:.2f} seconds')