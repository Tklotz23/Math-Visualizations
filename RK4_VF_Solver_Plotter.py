## This code is for plotting numerical solutions to ODE systems in a 2D phase-plane as well as the corresponding vector field. 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

## User input. Currently set up for numpy notation only.
print("\n\n Welcome! In the following prompts use x[0] and x[1] for the 1st and 2nd state variables respectively of your ODE system.\n")
test_case = input("Use test case? y/n ")
if test_case == 'y':
        ## Test case
    t_0 = 0 # Initial conditions  
    X_0 = np.array([0,1])
    T = 14*np.pi # End of desired solution interval 
    def RHS_ODE(t,x):
        # alpha = 0.5 # specific parameters for Duffing Equation. 
        # beta = 2/3
        # gamma = 5/6 
        # delta = 1/8
        # omega = 10

        # f1 = x[1] # RHS of first ODE in the system.
        # f2 = -delta*x[1]-beta*x[0]**3-alpha*x[0] # RHS of the second.
        f1 = x[1]**3-x[0]
        f2 = np.sin(x[0])
        return np.array([f1,f2])
else:
    user_RHS_1 = input("Enter RHS of 1st Eq. with numpy notation: ")
    user_RHS_2 = input("Enter RHS of 2nd Eq. with numpy notation: ")
    user_IC_t0 = input("Enter the t_0 value: ")
    user_IC_X01 = input("Enter the 1st X_0 value: ")
    user_IC_X02 = input("Enter the 2nd X_0 value: ")
    user_T = input("Enter the upper bound for the t-interval: ")
    
    ## Example ODE: Duffing Equation.
    t_0 = float(user_IC_t0) # Initial conditions  
    X_0 = np.array([float(user_IC_X01),float(user_IC_X02)])
    T = float(user_T)
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
    overflow_risk = False
    for s in time:
        if not overflow_risk:
            k1 = RHS_ODE(s,x) # Setting up the RK recursions
            k2 = RHS_ODE(s+h/2,np.add(x,(h/2)*k1))
            k3 = RHS_ODE(s+h/2,np.add(x,(h/2)*k2))
            k4 = RHS_ODE(s+h,np.add(x,h*k3))
            x = x+(h/6)*(k1+2*k2+2*k3+k4)
            for x_comp in x:
                if np.abs(x_comp) >=1E+30:
                    overflow_risk = True
                else: 
                    sol.append(np.array(x).tolist()) # Solution!
        else:
            print("Solution getting too large! Happens at time t=",s)
            break
    return np.array(sol)

t = np.arange(t_0, T+h, h) # Solution interval.  

## Function that creates each little line on a mesh_grid according the vector field defined by the ODE system. 
def vector_line(t,p_x1,p_x2,r1,r2):
    point = np.array([p_x1,p_x2]) # Midpoint for the line
    if np.abs(RHS_ODE(t,point)[0])<=1E-14 and RHS_ODE(t,point)[0]>0:
        angle = np.pi/2
    elif np.abs(RHS_ODE(t,point)[0])<=1E-14 and RHS_ODE(t,point)[0]<0:
        angle = -np.pi/2
    else:
        angle = np.arctan(r1*RHS_ODE(t,point)[1]/(r2*RHS_ODE(t,point)[0])) # angle of v.f. with x1-axis.
    x1_ellipse = r1*np.cos(angle) # Components of point on ellipse.
    x2_ellipse = r2*np.sin(angle)
    
    x1_ellipse_anti = r1*np.cos(angle+np.pi) # Components of antipodal point on ellipse.
    x2_ellipse_anti = r2*np.sin(angle+np.pi)    

    ellipse_pt = np.array([x1_ellipse,x2_ellipse]) # Point on ellipse.
    ellipse_pt_anti = np.array([x1_ellipse_anti,x2_ellipse_anti]) # Antipodal point on ellipse. 
    arrow_start = point+ellipse_pt_anti # Starting end of line.
    arrow_end = point+ellipse_pt # Ending end of line.
    arrow = np.array([arrow_start,arrow_end]) # The line! (Or data for matplotlib to plot the line. Needs to be transposed upon function return for correct plot). 
    return np.transpose(arrow)[0],np.transpose(arrow)[1]
## Function to determine color of vectors/lines based on vector magnitude.
#def line_color(t,p_x1,p_x2,mag_max):
    
## Now to make a meshgrid adapted to the size of the solution curve
mesh_density = 30

x1_sol_min = np.min(x_sol(t,X_0)[:,0])
x1_sol_max = np.max(x_sol(t,X_0)[:,0])

x2_sol_min = np.min(x_sol(t,X_0)[:,1])
x2_sol_max = np.max(x_sol(t,X_0)[:,1])

x1_val = np.linspace(x1_sol_min,x1_sol_max,mesh_density)
x2_val = np.linspace(x2_sol_min,x2_sol_max,mesh_density)

m_radius_1 = (x1_val[1]-x1_val[0])/2 # Half the distance between horizontal meshpoints
m_radius_2 = (x2_val[1]-x2_val[0])/2 # Half the distance between the vertical meshpoints

m_x1, m_x2 = np.meshgrid(x1_val,x2_val)
mesh_full = np.array([m_x1,m_x2]) # Full mesh.

mag_max = np.max(RHS_ODE(t,mesh_full)) # Need this for color scaling later.
print(mag_max)
## Setting up the plots
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8,8)) # Fixing a square scaling
ax.plot(x_sol(t,X_0)[:,0],x_sol(t,X_0)[:,1], color='tab:orange') # Solution curve plot
ax.plot(m_x1,m_x2, marker='.', color='w', linestyle='none') # Plotting the meshgrid
for p in x1_val: # This double for-loop plots each arrow on the meshgrid.
    for q in x2_val:
        ax.plot(vector_line(t,p,q,m_radius_1,m_radius_2)[0],vector_line(t,p,q,m_radius_1,m_radius_2)[1], color='c')
## Show-off time I guess
plt.show(fig)
