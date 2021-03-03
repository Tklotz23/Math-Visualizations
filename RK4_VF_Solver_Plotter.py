## This code is for plotting numerical solutions to ODE systems in a 2D phase-plane as well as the corresponding vector field. Everything is adapted to the size of the solution curve.
## WARNING: This program currently uses the python eval() function. DO NOT run with untrusted input. This also means you SHOULD NOT include stupidly big numbers in the RHS of your ODE input. You will use up all your memory! 
## Finally, the program itself will become more organized overtime, even with newer features.

import numexpr as ne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

## These are functions for example ODEs that are stored in a dictonary. Choices are called by user. 
def RHS_ODE1(t,x):
    alpha = 0.5 # specific parameters for Duffing Equation. 
    beta = 2/3
    gamma = 5/6 
    delta = 1/8
    omega = 10
    f1 = np.sin(x[1]) # RHS of first ODE in the system.
    f2 = -delta*x[0]-alpha*np.cos(x[1]) # RHS of the second.
    return np.array([f1,f2])

def RHS_ODE2(t,x):
    f1 = x[1]**3-x[0]
    f2 = np.cos(x[0])
    return np.array([f1,f2])

def RHS_ODE3(t,x):
    f1 = A[0,0]*x[0]+A[0,1]*x[1]
    f2 = A[1,0]*x[0]+A[1,1]*x[1]
    return np.array([f1,f2])

def RHS_ODE4(t,x):
    f1 = eval(user_RHS_1)
    f2 = eval(user_RHS_2)
    return np.array([f1,f2])

RHS_ODE = {'1': RHS_ODE1,'2': RHS_ODE2,'3': RHS_ODE3,'4': RHS_ODE4} # ODE dictionary. 

## The following function is for setting the paramters of the linear ODE (example 3). 
def lin_coeff_matrix():
    pre_e_vec11=input("Enter first component of first desired real eigenvector as a float:\n ")
    pre_e_vec12=input("Enter second component of first desired real eigenvector as a float:\n ")
    pre_e_vec21=input("Enter first component of second desired real eigenvector as a float:\n ")
    pre_e_vec22=input("Enter second component of second desired real eigenvector as a float:\n ")
    e_vec11 = float(pre_e_vec11)
    e_vec12 = float(pre_e_vec12)
    e_vec21 = float(pre_e_vec21)
    e_vec22 = float(pre_e_vec22)

    P = np.array([[e_vec11,e_vec12],[e_vec21,e_vec22]])
    Pinv = np.linalg.inv(P)

    r_c_choice = input("For real, nondefective, eigenvectors enter 'r', for complex enter 'c', and for repeated, defective, enter 'rr':\n")
    if r_c_choice == 'r':
        pre_e_val1 = input("Enter first desired eigenvalue as float:\n")
        pre_e_val2 = input("Enter second desired eigenvalue as float:\n")
        e_val1 = float(pre_e_val1)
        e_val2 = float(pre_e_val2)
        D = np.array([[e_val1,0],[0,e_val2]])
        A = P @ D @ Pinv
        t_0 = -1/(np.abs(min([e_val1,e_val2]))) # Start of solution interval. 
        T = 1/np.abs(max([e_val1,e_val2])) # End of solution interval.
    elif r_c_choice == 'c':
        pre_e_re = input("Enter real part of eigenvalue as float:\n")
        pre_e_im = input("Enter imaginary part of eigenvalue as float:\n")
        e_re = float(pre_e_re)
        e_im = float(pre_e_im)
        C = np.array([[e_re,e_im],[-e_im,e_re]])
        A = P @ C @ Pinv
        t_0 = -4*np.pi # Start of solution interval.
        T = 4*np.pi # End of solution interval.
    elif r_c_choice == 'rr':
        pre_e_val = input("Enter defective eigenvalue as float:\n")
        e_val = float(pre_e_val)
        J = np.array([[e_val,1],[0,e_val]])
        A = P @ J @ Pinv
        t_0 = -1/(np.abs(min([e_val,1]))) # Start of solution interval. 
        T = 1/np.abs(max([e_val,1])) # End of solution interval.
    else: 
        print("Invalid entry! Please enjoy a linear swirly example for your troubles.")
        C = np.array([[-2,4],[-4,-2]])
        A = P @ C @ Pinv
        t_0 = -4*np.pi # Start of solution interval.
        T = 4*np.pi # End of solution interval.
    return A,t_0,T,np.transpose(P)[0],np.transpose(P)[1]

def solution_number(ex_choice):
    sol_number = input("\n\n How many solutions would you like to plot with the example ODE system? (REMARK: 10+ solutions may result in significantly longer run times): ")
    if sol_number == 1:
        X_0 = np.array([[0,1]])
    else:
        dimension = (int(sol_number),2)
        scaling_d = {'1': 3,'2': 3.2, '3': 0.5}
        X_0 = scaling_d[ex_choice]*np.random.uniform(-1,1,dimension)
    return X_0

## User input. Currently set up for numpy notation only.
print("\n\n Welcome! In the following prompts use x[0] and x[1] for the 1st and 2nd state variables respectively of your ODE system.\n")
example = input("Use an example ODE? enter 'y' for yes or anything else for no: ")
if example == 'y':
    ## Example parameters defined by user. 
    ex_choice = input("Select Example (enter the listed integer). Options are:\n\n 1: nonlinear Ex 1\n 2: nonlinear Ex 2\n 3: linear \n")
    if ex_choice == '1':
        d_ind = ex_choice
        t_0 = 0 # Initial conditions
        T = 10*np.pi # End of desired solution interval 
        X_0 = solution_number(ex_choice)
    elif ex_choice == '2':
        d_ind = ex_choice
        t_0 = -14*np.pi # Initial conditions
        T = 14*np.pi # End of desired solution interval 
        X_0 = solution_number(ex_choice)
    elif ex_choice == '3':
        d_ind = ex_choice
        A,t_0,T,v1,v2 = lin_coeff_matrix()
        X_0 = solution_number(ex_choice)
    else: 
        print("Incorrect input! Please enjoy nonlinear example 2 for your troubles.")
        d_ind = '2'
        t_0 = 0 # Initial conditions
        T = 14*np.pi # End of desired solution interval 
        X_0 = solution_number('2')
    print("Initial values randomly selected for your solution curve(s): ")
    print(X_0) 
else:
    # User defined parameters including RHS of ODE systems, initial condition, and time interval. 
    user_RHS_1 = input("Enter RHS of 1st Eq. with numpy notation: ")
    user_RHS_2 = input("Enter RHS of 2nd Eq. with numpy notation: ")
    user_IC_t0 = input("Enter the t_0 value: ")
    user_IC_X01 = input("Enter the 1st X_0 value: ")
    user_IC_X02 = input("Enter the 2nd X_0 value: ")
    user_T = input("Enter the upper bound for the t-interval: ")
    
    t_0 = float(user_IC_t0) # Initial conditions  
    X_0 = np.array([[float(user_IC_X01),float(user_IC_X02)]])
    T = float(user_T)
    d_ind = '4'
## Numerical Solution as a python function
h = 0.005 # Step-size
def x_sol(time, X_naught):
    x = X_naught # Initialize I.C.
    sol = []
    overflow_risk = False
    for s in time:
        if not overflow_risk:
            k1 = RHS_ODE[d_ind](s,x) # Setting up the RK recursions
            k2 = RHS_ODE[d_ind](s+h/2,np.add(x,(h/2)*k1))
            k3 = RHS_ODE[d_ind](s+h/2,np.add(x,(h/2)*k2))
            k4 = RHS_ODE[d_ind](s+h,np.add(x,h*k3))
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
    if np.abs(RHS_ODE[d_ind](t,point)[0])<=1E-10 and RHS_ODE[d_ind](t,point)[0]>0:
        angle = np.pi/2
    elif np.abs(RHS_ODE[d_ind](t,point)[0])<=1E-10 and RHS_ODE[d_ind](t,point)[0]<0:
        angle = -np.pi/2
    else:
        angle = np.arctan2(r1*RHS_ODE[d_ind](t,point)[1],(r2*RHS_ODE[d_ind](t,point)[0])) # angle of v.f. with x1-axis. NOTE: arctan2 eats in two values whereas arctan takes in a a single argument (usually y/x). 
    x1_ellipse = r1*np.cos(angle) # Components of point on ellipse.
    x2_ellipse = r2*np.sin(angle)
    
    x1_ellipse_anti = r1*np.cos(angle+np.pi) # Components of antipodal point on ellipse.
    x2_ellipse_anti = r2*np.sin(angle+np.pi)    

    ellipse_pt = np.array([x1_ellipse,x2_ellipse]) # Point on ellipse.
    ellipse_pt_anti = np.array([x1_ellipse_anti,x2_ellipse_anti]) # Antipodal point on ellipse. 
    arrow_start = point+ellipse_pt_anti # Starting end of line.
    arrow_end = point+ellipse_pt # Ending end of line.
    arrow = np.array([arrow_start,arrow_end]) # The line! (Or data for matplotlib to plot the line. Needs to be transposed upon function return for correct plot). 

    pct = 0
    arrow_tip_start = point*(1-pct)+arrow_end*pct
    arrow_tip_diff = arrow_end-arrow_tip_start
    
    return np.transpose(arrow)[0],np.transpose(arrow)[1],arrow_tip_start,arrow_tip_diff

## Now to make a meshgrid adapted to the size of the solution curves.
mesh_density = 32
## initializing lists to select largest and smallest coordinate values among all solution curves. 
min_list_1 = []
max_list_1 = []
min_list_2 = []
max_list_2 = []
for x0 in X_0:
    min_list_1.append(np.min(x_sol(t,x0)[:,0]))
    max_list_1.append(np.max(x_sol(t,x0)[:,0]))
    min_list_2.append(np.min(x_sol(t,x0)[:,1]))
    max_list_2.append(np.max(x_sol(t,x0)[:,1]))

x1_sol_min = min(min_list_1)
x1_sol_max = max(max_list_1)
x2_sol_min = min(min_list_2)
x2_sol_max = max(max_list_2)

x1_val = np.linspace(x1_sol_min,x1_sol_max,mesh_density)
x2_val = np.linspace(x2_sol_min,x2_sol_max,mesh_density)

m_radius_1 = (x1_val[1]-x1_val[0])/2 # Half the distance between horizontal meshpoints
m_radius_2 = (x2_val[1]-x2_val[0])/2 # Half the distance between the vertical meshpoints

m_x1, m_x2 = np.meshgrid(x1_val,x2_val)
mesh_full = np.array([m_x1,m_x2]) # Full mesh.

## Function to determine color of vectors/lines based on vector magnitude. Function returns rgb-alpha values in a tuple for use in the color option of a plot. 

color_choice = 'winter' # Choosing the coloring for the vector field. Helpful later too.
mag_max = np.max(np.sqrt(RHS_ODE[d_ind](t,mesh_full)[0]**2+RHS_ODE[d_ind](t,mesh_full)[1]**2)) # max magnitude of vf over whole mesh grid. 
mag_min = np.min(np.sqrt(RHS_ODE[d_ind](t,mesh_full)[0]**2+RHS_ODE[d_ind](t,mesh_full)[1]**2)) # min magnitude of vf over whole mesh grid.

def line_color(t,p_x1,p_x2):
    vf_coloring = cm.get_cmap(color_choice) # Assign a colormap to float values between 0 and 1. 
    point = np.array([p_x1,p_x2]) # Point in the mesh.
    v_p = RHS_ODE[d_ind](t,point) # Vector at the point in the mesh.
    v_p_mag = np.sqrt(v_p[0]**2+v_p[1]**2) # Magnitude of said vector. 
    v_p_mag_scaled = (v_p_mag-mag_min)/(mag_max-mag_min) # Linear scaling of vf magnitude on mesh grid to the interval [0,1]. 
    return vf_coloring(v_p_mag_scaled), v_p_mag

## A function to set up the plots.
def plotting_time():
    plt.style.use('dark_custom1')
    fig, ax = plt.subplots(figsize=(8,8)) # Fixing a square scaling.
    # ax.plot(m_x1,m_x2, marker='.', color='w', linestyle='none') # Plotting the meshgrid.

    ## This double for-loop plots each arrow on the meshgrid.
    for p in x1_val:
        for q in x2_val:
            mag_color = line_color(t,p,q)[0]
            arrow_start_x1 = vector_line(t,p,q,m_radius_1,m_radius_2)[2][0]
            arrow_start_x2 = vector_line(t,p,q,m_radius_1,m_radius_2)[2][1]
            dx1 = vector_line(t,p,q,m_radius_1,m_radius_2)[3][0]
            dx2 = vector_line(t,p,q,m_radius_1,m_radius_2)[3][1]

            ax.plot(vector_line(t,p,q,m_radius_1,m_radius_2)[0],vector_line(t,p,q,m_radius_1,m_radius_2)[1],color=mag_color)
            #plt.arrow(arrow_start_x1,arrow_start_x2,dx1,dx2,shape='right',head_width=0.03,color=mag_color)
    ## Plots the solution curves
    for x0 in X_0:
        border_color = 'xkcd:black'#cm.get_cmap(color_choice)(0)
        inner_color = 'xkcd:beige'#cm.get_cmap(color_choice)(0.99)
        ax.plot(x_sol(t,x0)[:,0],x_sol(t,x0)[:,1],linewidth=2.5,color=border_color)# Border.
        ax.plot(x_sol(t,x0)[:,0],x_sol(t,x0)[:,1],color=inner_color) # Solution curve plot.

    if ex_choice == '3': # Plot eigenvector subspaces for linear systems.
        plot_e_vecs = input("Plot eigenvectors as well? Enter 'y' for yes ad anything else for no:\n ") 
        if plot_e_vecs == 'y':
            ax.plot(t*v1[0],t*v1[1],color='red')
            ax.plot(t*v2[0],t*v2[1],color='red')
    
    ax.set_xlabel(r'\LARGE $x_1$') # Label the axes!
    ax.set_ylabel(r'\LARGE $x_2$')

    ## Colorbar stuff!
    cmap = cm.winter
    norm = cm.colors.Normalize(vmin=mag_min,vmax=mag_max)
    plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, ticks=[mag_min,(3*mag_min+mag_max)/4,(mag_min+mag_max)/2,(mag_min+3*mag_max)/4,mag_max],label='Vector Field Magnitude')  

    return plt.show(fig)

## Show-off time I guess.
plotting_time()
