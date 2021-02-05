## This code is for plotting numerical solutions to ODE systems in a 2D phase-plane as well as the corresponding vector field. 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

## User input. Currently set up for numpy notation only.
print("\n\n Welcome! In the following prompts use x[0] and x[1] for the 1st and 2nd state variables respectively of your ODE system.\n")
user_RHS_1 = input("Enter RHS of 1st Eq. with numpy notation: ")
user_RHS_2 = input("Enter RHS of 2nd Eq. with numpy notation: ")
user_IC_t0 = input("Enter the t_0 value: ")
user_IC_X01 = input("Enter the 1st X_0 value: ")
user_IC_X02 = input("Enter the 2nd X_0 value: ")

## Example ODE: Duffing Equation.
t_0 = float(user_IC_t0) # Initial conditions  
X_0 = np.array([float(user_IC_X01),float(user_IC_X02)])
def RHS_ODE(t,x):
    # alpha = 0.5 # specific parameters for Duffing Equation. 
    # beta = 2/3
    # gamma = 5/6 
    # delta = 1/8
    # omega = 10

    # f1 = x[1] # RHS of first ODE in the system.
    # f2 = -delta*x[1]-beta*x[0]**3-alpha*x[0] # RHS of the second.
    f1 = eval(user_RHS_1)
    f2 = eval(user_RHS_2) 
    return np.array([f1,f2])
## Numerical Solution as a python function
h = 0.005 # Step-size
def x_sol(time, X_naught):
    x = X_naught # Initialize I.C.
    sol =[]
    for s in time:
        k1 = RHS_ODE(s,x) # Setting up the RK recursions
        k2 = RHS_ODE(s+h/2,np.add(x,(h/2)*k1))
        k3 = RHS_ODE(s+h/2,np.add(x,(h/2)*k2))
        k4 = RHS_ODE(s+h,np.add(x,h*k3))
        x = x+(h/6)*(k1+2*k2+2*k3+k4)
        sol.append(np.array(x).tolist()) # Solution!
    return np.array(sol)

T = 14*np.pi # End of desired solution interval 
t = np.arange(t_0, T+h, h) # Solution interval. 

## Function that creates each little line on a mesh_grid according the vector field defined by the ODE system. 
def vector_line(t,p_x1,p_x2,mesh_radius):
    point = np.array([p_x1,p_x2]) # Midpoint for the line
    angle = np.arctan(RHS_ODE(t,point)[1]/RHS_ODE(t,point)[0]) # angle of v.f. with x1-axis.
    x1_pol_comp = mesh_radius*np.cos(angle) # coordinates given by v.f. on the circle centered at point. 
    x2_pol_comp = mesh_radius*np.sin(angle)
    circle_pt = np.array([x1_pol_comp,x2_pol_comp]) # Point on circle.
    arrow_start = np.add(point,-circle_pt) # Starting end of line.
    arrow_end = np.add(point,circle_pt) # Ending end of line.
    arrow = np.array([arrow_start,arrow_end]) # The line!
    return np.transpose(arrow)[0],np.transpose(arrow)[1]
## Function to determine color of vectors/lines based on vector magnitude.
#def line_color(t,p_x1,p_x2,mag_max):
    
## Now to make a meshgrid adapted to the size of the solution curve
mesh_density = 30
x1_val = np.linspace(np.min(x_sol(t,X_0)[:,0]),np.max(x_sol(t,X_0)[:,0]),mesh_density)
x2_val = np.linspace(np.min(x_sol(t,X_0)[:,1]),np.max(x_sol(t,X_0)[:,1]),mesh_density)
m_radius = (x1_val[1]-x1_val[0])/2 # Half the distance between horizontal meshpoints
print(2*m_radius)
m_x1, m_x2 = np.meshgrid(x1_val,x2_val)
mesh_full = np.array([m_x1,m_x2]) # Full mesh.

mag_max = np.max(RHS_ODE(t,mesh_full)) # Need this for color scaling.
print(mag_max)
## Setting up the plots
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8,8)) # Fixing a square scaling
ax.plot(x_sol(t,X_0)[:,0],x_sol(t,X_0)[:,1], color='tab:orange') # Solution curve plot
ax.plot(m_x1,m_x2, marker='.', color='w', linestyle='none') # Plotting the meshgrid

for p in x1_val: # This double for-loop plots each arrow on the meshgrid.
    for q in x2_val:
        ax.plot(vector_line(t,p,q,m_radius)[0],vector_line(t,p,q,m_radius)[1], color='c')
        print(np.linalg.norm(np.transpose(vector_line(t,p,q,m_radius))[1]-np.transpose(vector_line(t,p,q,m_radius))))
## Show-off time I guess
plt.show(fig)
