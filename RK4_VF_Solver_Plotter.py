## This code is for plotting numerical solutions to ODE systems in a 2D phase-plane as well as the corresponding vector field. Everything is adapted to the size of the solution curve.
## WARNING: This program currently uses the python eval() function. DO NOT run with untrusted input. This also means you SHOULD NOT include stupidly big numbers in the RHS of your ODE input. You will use up all your memory! 
## Don't forget to download the custom style sheet 'dark_custom1.mplstyle' found in the same folder as this program.
## Finally, the program itself will become more organized overtime, even with newer features. Currently it has a mild case of "spaghetti". 

import numexpr as ne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

## These are functions for example ODEs that are stored in a dictonary. Choices are called by user.

def RHS_ODE1(t,x):
    f1 = np.sin(x[1]) # RHS of first ODE in the system.
    f2 = -x[0]/8-np.cos(x[1])/2 # RHS of the second.
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

## User choice for number of solution curves from randomized initial data. 
def solution_number(d_ind):
    sol_number = input("\n\n How many solutions would you like to plot with the example ODE system? (REMARK: 10+ solutions may result in significantly longer run times): ")
    if sol_number == 1:
        X_0 = np.array([[random(),random()]])
    else:
        dimension = (int(sol_number),2)
        scaling_d = {'1': 3,'2': 3.2, '3': 0.5, '4': 1} # Choses the length for the range of random initial conditions. 
        X_0 = scaling_d[d_ind]*np.random.uniform(-1,1,dimension)
    return X_0,int(sol_number)
    
## Example parameters defined by user.
def example_parameters():
    d_ind = input("Select Example (enter the listed integer). Options are:\n\n 1: nonlinear Ex 1\n 2: nonlinear Ex 2\n 3: linear \n") # d_ind is the user's choice of index for the ODE dictionary. 
    if d_ind == '1':
        t_0 = 0 # Initial conditions
        T = 10*np.pi # End of desired solution interval 
        X_0,sol_number = solution_number(d_ind)
        A,v1,v2 = None,None,None
    elif d_ind == '2':
        t_0 = -14*np.pi # Initial conditions
        T = 14*np.pi # End of desired solution interval 
        X_0,sol_number = solution_number(d_ind)
        A,v1,v2 = None,None,None
    elif d_ind == '3':
        X_0,sol_number = solution_number(d_ind)
        A,t_0,T,v1,v2 = lin_coeff_matrix()
    else: 
        print("Incorrect input! Please enjoy nonlinear example 2 for your troubles.")
        d_ind = '2'
        t_0 = 0 # Initial conditions
        T = 14*np.pi # End of desired solution interval 
        X_0,sol_number = solution_number('2')
        A,v1,v2 = None,None,None
        print("Initial values randomly selected for your solution curve(s): ")
        print(X_0)
    return t_0,T,X_0,d_ind,sol_number,A,v1,v2
## This is where the eval() function is being used: to allow the user to input numpy code to define the ODE. 
def user_RHS():
    user_RHS_1 = input("Enter RHS of 1st Eq. with numpy notation: ")
    user_RHS_2 = input("Enter RHS of 2nd Eq. with numpy notation: ")
    user_IC_t0 = input("Enter the t_0 value: ")
    user_IC_X01 = input("Enter the 1st X_0 value: ")
    user_IC_X02 = input("Enter the 2nd X_0 value: ")
    user_T = input("Enter the upper bound for the t-interval: ")
    
    t_0 = float(user_IC_t0) # Initial conditions  
    X_0 = np.array([[float(user_IC_X01),float(user_IC_X02)]])
    T = float(user_T)
    return t_0,T,X_0
## User input.
def user_choices():
    print("\n\n Welcome! In the following prompts use x[0] and x[1] for the 1st and 2nd state variables respectively of your ODE system.\n")
    example = input("Use an example ODE? enter 'y' for yes or anything else for no: ")
    if example == 'y':
        t_0,T,X_0,d_ind,sol_number,A,v1,v2 = example_parameters()
    else:
        t_0,T,X_0 = user_RHS()
        d_ind = '4'
        sol_number = 1
        A,v1,v2 = None,None,None    
    return t_0,T,X_0,d_ind,sol_number,A,v1,v2
## Numerical Solution as a python function
h = 0.005 # Step-size
def x_sol(time, X_naught,d_ind):
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
 
## Now to make a meshgrid adapted to the size of the solution curves.
def mesh2D(t,sol):
    mesh_density = 32
    ## initializing lists to select largest and smallest coordinate values among all solution curves. 
    min_list_1 = []
    max_list_1 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(0,sol_number):
        sol_x1 = sol[i][:,0]
        sol_x2 = sol[i][:,1]
        min_list_1.append(np.min(sol_x1))
        max_list_1.append(np.max(sol_x1))
        min_list_2.append(np.min(sol_x2))
        max_list_2.append(np.max(sol_x2))

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
    return [x1_val,x2_val],[m_radius_1,m_radius_2],mesh_full

## Function that creates each little line on a mesh_grid according the vector field defined by the ODE system. 
def vector_line(t,p_x1,p_x2,r1,r2):
    point = np.array([p_x1,p_x2]) # Midpoint for the line
    RHS_f1 = RHS_ODE[d_ind](t,point)[0]
    RHS_f2 = RHS_ODE[d_ind](t,point)[1]

    angle = np.arctan2(r1*RHS_f2,r2*RHS_f1) # angle of v.f. with x1-axis. NOTE: arctan2 eats in two values whereas arctan takes in a a single argument (usually y/x). 
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

## Function to determine color of vectors/lines based on vector magnitude. Function returns rgb-alpha values in a tuple for use in the color option of a plot. 

def line_color(t,p_x1,p_x2,mesh):
    color_choice = 'winter' # Choosing the coloring for the vector field. Helpful later too.
    mag_mesh = RHS_ODE[d_ind](t,mesh)
    mag_max = np.max(np.sqrt(mag_mesh[0]**2+mag_mesh[1]**2)) # max magnitude of vf over whole mesh grid. 
    mag_min = np.min(np.sqrt(mag_mesh[0]**2+mag_mesh[1]**2)) # min magnitude of vf over whole mesh grid.
    vf_coloring = cm.get_cmap(color_choice) # Assign a colormap to float values between 0 and 1. 
    point = np.array([p_x1,p_x2]) # Point in the mesh.
    v_p = RHS_ODE[d_ind](t,point) # Vector at the point in the mesh.
    v_p_mag = np.sqrt(v_p[0]**2+v_p[1]**2) # Magnitude of said vector. 
    v_p_mag_scaled = (v_p_mag-mag_min)/(mag_max-mag_min) # Linear scaling of vf magnitude on mesh grid to the interval [0,1]. 
    return vf_coloring(v_p_mag_scaled),[mag_max,mag_min]

## Eigenvector plotting option.
def eigen_lines(v1,v2,x_val_mesh):
    plot_e_vecs = input("Plot eigenvectors as well? Enter 'y' for yes and anything else for no (Remark: there is still a minor issue of fitting if you selected only a single curve or if the initial conditions for multiple curves weren't sufficiently well distributed between all four quadrants):\n ") 
    if plot_e_vecs == 'y':
        x1_min = x_val_mesh[0][0]
        x1_max = x_val_mesh[0][31]
        print(x1_min,x1_max)
        x2_min = x_val_mesh[1][0]
        x2_max = x_val_mesh[1][31]
        print(x2_min,x2_max)
        diag_angle_1 = np.arctan2(x2_max,x1_max)        
        diag_angle_2 = np.arctan2(x2_max,x1_min)
        diag_angles = [diag_angle_1,diag_angle_2,diag_angle_1+np.pi,diag_angle_2+np.pi] 
        eigen_angle_1 = np.arctan2(v1[1],v1[0])
        eigen_angle_2 = np.arctan2(v2[1],v2[0])
        count_1 = 0 
        count_2 = 0
        for theta in reversed(diag_angles):
            if eigen_angle_1 >= theta:
                count_1 = count_1+1
            if eigen_angle_2 >= theta:
                count_2 = count_2+1  
        if count_1%4 == 0 or count_1%4 == 2:
            s1_i = x1_min/v1[0]
            s1_f = x1_max/v1[0]
        else:
            s1_i = x2_min/v1[1]
            s1_f = x2_max/v1[1]
        if count_2%4 == 0 or count_2%4 == 2:
            s2_i = x1_min/v2[0]
            s2_f = x1_max/v2[0]
        else:
            s2_i = x2_min/v2[1]
            s2_f = x2_max/v2[1]         
        s1 = np.linspace(s1_i, s1_f)
        s2 = np.linspace(s2_i, s2_f)
        print(s1)
        v1_line_1,v1_line_2 = s1*v1[0], s1*v1[1]
        v2_line_1,v2_line_2 = s2*v2[0], s2*v2[1]

    return [v1_line_1,v1_line_2],[v2_line_1,v2_line_2]

## A function to set up the plots.
def plotting_time(t,X_0,sol,x_val_mesh,m_radius,mesh):
        
    
    plt.style.use('dark_custom1')
    fig, ax = plt.subplots(figsize=(8,8)) # Fixing a square scaling.
    # ax.plot(m_x1,m_x2, marker='.', color='w', linestyle='none') # Plotting the meshgrid.

    ## This double for-loop plots each arrow on the meshgrid.
    for p in x_val_mesh[0]:
        for q in x_val_mesh[1]:
            mag_color,mag_extrema = line_color(t,p,q,mesh) # Gotta' streamline mag_extrema later. 
            arrow_start_x1 = vector_line(t,p,q,m_radius[0],m_radius[1])[2][0]
            arrow_start_x2 = vector_line(t,p,q,m_radius[0],m_radius[1])[2][1]
            dx1 = vector_line(t,p,q,m_radius[0],m_radius[1])[3][0]
            dx2 = vector_line(t,p,q,m_radius[0],m_radius[1])[3][1]

            ax.plot(vector_line(t,p,q,m_radius[0],m_radius[1])[0],vector_line(t,p,q,m_radius[0],m_radius[1])[1],color=mag_color)
            #plt.arrow(arrow_start_x1,arrow_start_x2,dx1,dx2,shape='right',head_width=0.03,color=mag_color)
    ## Plot eigenvector subspaces for linear systems.
    if d_ind == '3':
        v1_line,v2_line = eigen_lines(v1,v2,x_val_mesh)
        ax.plot(v1_line[0],v1_line[1],color='red')
        ax.plot(v2_line[0],v2_line[1],color='red')
        print(v1_line)    
    ## Plots the solution curves
    for i in range(0,sol_number):
        border_color = 'xkcd:black'#cm.get_cmap(color_choice)(0)
        inner_color = 'xkcd:beige'#cm.get_cmap(color_choice)(0.99)
        ax.plot(sol[i][:,0],sol[i][:,1],linewidth=2.5,color=border_color)# Border.
        ax.plot(sol[i][:,0],sol[i][:,1],color=inner_color) # Solution curve plot.

    ax.set_xlabel(r'\LARGE $x_1$') # Label the axes!
    ax.set_ylabel(r'\LARGE $x_2$')

    ## Colorbar stuff!
    mag_max = mag_extrema[0]
    mag_min = mag_extrema[1]
    cmap = cm.winter
    norm = cm.colors.Normalize(vmin=mag_min,vmax=mag_max)
    tick_list = [mag_min,(3*mag_min+mag_max)/4,(mag_min+mag_max)/2,(mag_min+3*mag_max)/4,mag_max]
    plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, ticks=tick_list,label='Vector Field Magnitude')

    return plt.show(fig)

## Execution of the program. Named after Zhu Li Moon from Avatar: Legend of Korra.
def Zhu_Li_DoTheThing():
    t = np.arange(t_0, T+h, h) # Solution interval.
    sol = [x_sol(t,x0,d_ind) for x0 in X_0] # List of solutions for each X_0. 
    x_val_mesh,m_radius,mesh = mesh2D(t,sol)
    plots = plotting_time(t,X_0,sol,x_val_mesh,m_radius,mesh)
    return plots
t_0,T,X_0,d_ind,sol_number,A,v1,v2 = user_choices()
Zhu_Li_DoTheThing()
