import numpy as np
from tqdm import tqdm
from CTRNN import CTRNN
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import matplotlib.markers as markers
import matplotlib.animation as animation

# Physical Constants
mu_Earth = 398.6*10**12     # [m^3/s^2] gravitational parameter Earth
R_Earth = 6371*1000         # [m] average radius Earth
G = 6.67430*10**(-11)       # [N*m^2/kg^2] gravitational constant

# Plotting Constants
color_lin = "C0"
color_ex = "C1"
color_orbit = "C2"
color_target = "red"
mSize = 50

# =========================================================================================================================================== AGENTS

# Random Agent --> just behaves randomly, as a way of testing core simulator functionalities
# (used for testing purposes while developing, not related to paper itself)
class RandomAgent:

    def __init__(self):
        # Empty lists to store movement history
        self.x, self.y, self.z = [],  [], []
        self.xdot, self.ydot, self.zdot = [],  [], []

        self.impulse = 0        # Total imparted impulse
        self.reward = -10**30   # Initial fitness should be 'infinitely' bad
        self.genome = 100*[-1]  # Nonsense genome
    
    # Returns random forces (Gaussian)
    def act(self, Y, t):

        sc = 100    # Scale

        # Normally distributed forces around 0
        Fx = sc*np.random.normal(loc = 0, scale = 1)
        Fy = sc*np.random.normal(loc = 0, scale = 1)
        Fz = sc*np.random.normal(loc = 0, scale = 1)

        return Fx, Fy, Fz

    # Evaluate fitness at the end of simulation
    def evaluate(self):
        
        # Final position
        endPos = np.sqrt(self.x[-1]**2 + self.y[-1]**2 + self.z[-1]**2)
        # Initial position
        beginPos = np.sqrt(self.x[0]**2 + self.y[0]**2 + self.z[0]**2)
        # Final position relative to initial one
        posProgression = endPos/beginPos*100
        # Final velocity
        endVel = np.sqrt(self.xdot[-1]**2 + self.ydot[-1]**2 + self.zdot[-1]**2)

        # *can be changed to any desired reward function*
        self.reward = -posProgression - endVel/10 - self.impulse/100    # Option 1: Minimize final position, final velocity, and total used impulse (= fuel)
        self.reward = -posProgression                                   # Option 2: Only minimize final position
        # Severly punish if S/C is moving very far away from target
        if posProgression > 1.5:
            self.reward -= 1000

        return self.reward

# Uses Continuous Time Recurrent Neural Network
# (recurrent because loops inside the connections ==> time-dynamic behaviour)
# (neurons contain activation state, which evolves over time ==> inertia)
class NeuralAgent(object):
    
    def __init__(self, size, genome = [], twoDim = False):

        # Empty lists to store movement history
        self.x, self.y, self.z = [],  [], []
        self.xdot, self.ydot, self.zdot = [],  [], []
        self.t = []

        self.size = size        # Number of nodes in CTRNN
        self.impulse = 0        # Total imparted impulse
        self.reward = -10**30   # Initial fitness should be 'infinitely' bad
        self.twoDim = twoDim    # Whether simulation is 2D or 3D

        # Create NN with certain size (= #neurons) and step size (i.e. Euler integration step size when calculating dynamic behaviour)
        # NN is fully connected! (i.e. every neuron combines output of all other neurons, including itself)
        # De facto, weights are initialized randomly, while gains, time-constants and biases are 1
        self.NN = CTRNN(self.size, step_size = 0.1)

        # Internal dynamics governed by:

        # tau*sdot = -s + I + sum(w_j * sigma(g_j*(s_j + theta_j)))

        # tau = time constant (of this neuron)
        # s = activation state (of this neuron)
        # I = external input (to this neuron)
        # w_j = weight (from neuron j to this neuron)
        # sigma = activation function
        # g_j = gain (of neuron j)
        # theta_j = bias (of neuron j)

        # Genome format: tau[size] - w[size*size] - g[size] - theta[size]
        if genome != []:
            self.genome = genome
            # Set time constants
            self.NN.taus = genome[:size]
            # Set weights
            weights = np.reshape(genome[size:(1 + size)*size], (size, size))    # Reshape array to matrix
            self.NN.weights = csr_matrix(weights)                               # Convert to sparse matrix
            # Set gains
            self.NN.gains = genome[(1 + size)*size:(2 + size)*size]
            # Set biases
            self.NN.biases = genome[(2 + size)*size:(3 + size)*size]
            # Fix initial states, because otherwise there is an aspect of randomness and non-repeatability (inside CTRNN package)
            self.NN.states = np.array(size*[0.5])
        else:
            raise Exception("No genome supplied in agent initialization")

    # Returns agent's decision (i.e. outputs: forces) based on its inputs and internal state
    def act(self, Y, t):

        numIn = len(Y)                      # Number of inputs
        numOut = 2 if self.twoDim else 3    # Number of outputs

        # Actual input list should be as big as CTRNN (most values will just be 0)
        inputs = np.array([0.0]*self.size)
        inputs[0:numIn] = np.array(Y)   # First numIn neurons receive external (non-zero) input

        # Provide inputs, and step internal states of CTRNN
        self.NN.euler_step(inputs)

        # Read states of CTRNN, and use last few as outputs (Last numOut neurons provide output. Random choice, but neurons are inherently similar)
        output = 2.0 * (self.NN.outputs[-numOut:] - 0.5)    # Output is in [0, 1] --> reshape to [-1, 1]
        output *= 10                                        # Rescale so forces are larger

        # This part was initially used to filter out outputs (i.e. create a threshold: only actually thrust if output is above a certain value)
        # In the end, I didn't use it, but it turns out that it acts as a perfect safeguard for overflow errors (filters out NaN outputs)
        # enabling the overflow exploit explained in the paper (agent causing internal overflow)
        output = [i if np.abs(i) > 0.0 else 0 for i in output]

        return output

    # Evaluate fitness at the end of simulation
    def evaluate(self):
        
        # Final position
        endPos = np.sqrt(self.x[-1]**2 + self.y[-1]**2) if self.twoDim else np.sqrt(self.x[-1]**2 + self.y[-1]**2 + self.z[-1]**2)
        # Final velocity
        endVel = np.sqrt(self.xdot[-1]**2 + self.ydot[-1]**2) if self.twoDim else np.sqrt(self.xdot[-1]**2 + self.ydot[-1]**2 + self.zdot[-1]**2)

        # *can be changed to any desired reward function*
        self.reward = -endPos                       # Run 1 (seed 20)
        self.reward = -endPos - endVel*1000         # Run 2 (seed 21)
        self.reward = -endPos - endVel*1000         # Run 3 (seed 25)
        self.reward = -endPos**2 - endVel*1000*100  # Run 4 (seed 26)

        return self.reward

    # Calculate Thrust Score ( = negative of total impulse)
    def scoreF(self, F, t):
        # Use absolute value of thrusts, and sum X-Y-Z components (assume S/C can only fire thrusters in these three directions)
        Ftot = np.sum(np.abs(F), axis = 0)
        Itot = 0
        for i in range(len(Ftot) - 1):
            dt = t[i + 1] - t[i]
            Itot += (Ftot[i] + Ftot[i + 1])*dt/2
        dts = t[1:] - t[:-1]
        Is = dts*np.sum(np.abs(F), axis = 0)[1:]
        Itot = np.sum(Is)
        return Itot

# =========================================================================================================================================== SIMULATION

# Spacecraft (main simulation class, performes RK4 integration on Newton's Laws + Clohessy-Wiltshire)
class Spacecraft:

    def __init__(self, H, m, agent, twoDim = False, subSamples = 1):
        self.R = H + R_Earth                    # Orbit radius
        self.n = np.sqrt(mu_Earth/(self.R**3))  # Orbit mean motion
        self.m = m                              # Spacecraft mass
        self.thrusting = False                  # Flag whether S/C is thrusting (for subdividing time steps)
        self.Fs_lin = np.array([[0, 0, 0]]).T   # Stores thrusts for linearized sims
        self.Fs_ex = np.array([[0, 0, 0]]).T    # Stores thrusts for exact sims
        self.linear = True                      # Flag whether to use linearized equations
        self.agent = agent                      # Agent (makes thrusting decisions) --> either random or CTRNN
        self.twoDim = twoDim                    # Flag whether sim is 2D or 3D
        self.counter = 0                        # Counts how many time steps have progressed
        self.subSamples = subSamples            # Number of times a timestep should be subdivided when S/C is thrusting
        self.Fprev = [0, 0, 0]                  # Stores previous thrusts
    
    # ------------------------------------------------------------------------------------------------------------------------- CORE INTERFACE

    # Simulate S/C behaviour during n_steps (SCENARIO WITH GRAVITY)
    def simulate(self, Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, linear = True, exact = True):

        self.dim = 2 if self.twoDim else 3

        # Discretize time
        a = 0
        h = dt
        self.dt = dt
        self.n_steps = n_steps
        b = a + h*n_steps

        # Empty arrays in case user doesn't simulate them
        t_lin = np.array((n_steps + 1)*[None])
        t_ex = np.array((n_steps + 1)*[None])
        Y_lin = np.array(2*self.dim*[(n_steps + 1)*[None]])
        Y_ex = np.array(2*self.dim*[(n_steps + 1)*[None]])

        # Reset F info
        self.thrusting = False
        self.Fs_lin = np.array([self.dim*[0]]).T
        self.Fs_ex = np.array([self.dim*[0]]).T

        if linear:
            self.linear = True

            # Simulate with RK4 and function dY1 (linearized CW)
            Y_lin, t_lin = self.RK4(self.dY1, Y0, a, b, h)

            # Extract position data
            self.agent.x = Y_lin[0, :]
            self.agent.y = Y_lin[1, :]
            if not self.twoDim:
                self.agent.z = Y_lin[2, :]
            # Extract velocity data
            self.agent.xdot = Y_lin[self.dim, :]
            self.agent.ydot = Y_lin[self.dim + 1, :]
            if not self.twoDim:
                self.agent.zdot = Y_lin[self.dim + 2, :]
            # Extract time data
            self.agent.t = t_lin
            # Calculate impulse
            self.agent.impulse = self.agent.scoreF(self.Fs_lin, t_lin)

        if exact:
            self.linear = False

            # Simulate with RK4 and function dY2 (three-body problem)
            Y_ex, t_ex = self.RK4(self.dY2, Y0, a, b, h)

            # Extract position data
            self.agent.x = Y_ex[0, :]
            self.agent.y = Y_ex[1, :]
            if not self.twoDim:
                self.agent.z = Y_ex[2, :]
            # Extract velocity data
            self.agent.xdot = Y_ex[self.dim, :]
            self.agent.ydot = Y_ex[self.dim + 1, :]
            if not self.twoDim:
                self.agent.zdot = Y_ex[self.dim + 2, :]
            # Extract time data
            self.agent.t = t_ex
            # Calculate impulse
            self.agent.impulse = self.agent.scoreF(self.Fs_ex, t_ex)

        # Plot
        if timePlot:
            self.timePlot(Y_lin, Y_ex, t_lin, t_ex, min = True) # F, pos and vel as function of time
        if orbitPlot:
            self.orbitPlot(Y_lin, Y_ex, t_lin, t_ex)            # S/C trajectory
        if orbitAnim:
            self.orbitAnimate(Y_lin, Y_ex, t_lin, t_ex)         # Animation of S/C trajectory

        return Y_lin, Y_ex, t_lin, t_ex
    
    # Simulate S/C behaviour during n_steps (SCENARIO WITHOUT GRAVITY)
    def simulateNoGrav(self, Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, forcePlot = True):

        self.dim = 2 if self.twoDim else 3
        
        # Discretize time
        a = 0
        h = dt
        self.dt = dt
        self.n_steps = n_steps
        b = a + h*n_steps

        # Empty linear arrays (empty lin are not used, but have to be included for plotting)
        t_lin = np.array((n_steps + 1)*[None])
        t_ex = np.array((n_steps + 1)*[None])
        Y_lin = np.array(2*self.dim*[(n_steps + 1)*[None]])
        Y_ex = np.array(2*self.dim*[(n_steps + 1)*[None]])

        # Reset F info
        self.thrusting = False
        self.Fs_lin = np.array([self.dim * [0]]).T
        self.Fs_ex = np.array([self.dim * [0]]).T

        self.linear = False # Deep-space is always exact, no linearized version

        # Simulate with RK4 and function dY3 (deep space)
        Y_ex, t_ex = self.RK4(self.dY3, Y0, a, b, h)
        
        # Extract position data
        self.agent.x = Y_ex[0, :]
        self.agent.y = Y_ex[1, :]
        if not self.twoDim:
            self.agent.z = Y_ex[2, :]
        # Extract velocity data
        self.agent.xdot = Y_ex[self.dim, :]
        self.agent.ydot = Y_ex[self.dim + 1, :]
        if not self.twoDim:
            self.agent.zdot = Y_ex[self.dim + 2, :]
        # Extract time data
        self.agent.t = t_ex
        # Calculate impulse
        self.agent.impulse = self.agent.scoreF(self.Fs_ex, t_ex)

        # Plot
        if timePlot:
            self.timePlot(Y_lin, Y_ex, t_lin, t_ex)     # F, pos and vel as function of time
        if orbitPlot:
            self.orbitPlot(Y_lin, Y_ex, t_lin, t_ex)    # S/C trajectory
        if orbitAnim:
            self.orbitAnimate(Y_lin, Y_ex, t_lin, t_ex) # Animation of S/C trajectory
        if forcePlot:
            self.forcePlot(Y_lin, Y_ex, t_lin, t_ex)    # F at different points along the trajectory

        return Y_lin, Y_ex, t_lin, t_ex
    
    # ------------------------------------------------------------------------------------------------------------------------- SIMULATION

    # Runge-Kutta numerical solver of the fourth order
    def RK4(self, f, Y0, a, b, h):
        
        # Divide time domain [a, b] into steps of size h
        t = [a]
        h /= self.subSamples
        
        # Make RK4 compatible with both scalar (1D) and vector (nD) functions
        if type(Y0) == float or type(Y0) == int:
            Y0 = [Y0]
        M = len(Y0)
        
        # Y will store solution
        Y = np.zeros((M, 1))
        Y[:, 0] = Y0

        # Make first step short (subsampled), to immediately capture behaviour
        dt = h/100

        # Loop --> numerical solution
        while t[-1] <= b:
            
            # RK4 scheme
            k1 = np.array(f(t[-1], Y[:, -1]))
            k2 = np.array(f(t[-1] + dt/2, Y[:, -1] + dt/2*k1))
            k3 = np.array(f(t[-1] + dt/2, Y[:, -1] + dt/2*k2))
            k4 = np.array(f(t[-1] + dt, Y[:, -1] + dt*k3))
            Yi = np.array([Y[:, -1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)]).T
            # Update positon and velocity log
            Y = np.append(Y, Yi, axis = 1)
            # Update time log
            t = np.append(t, t[-1] + dt)
            # Calculate forces (< agent action)
            self.determineForces(Y[:, -1], t[-1])

            # Next step size is determined by whether S/C is thrusting or not
            if self.thrusting:
                dt = h/10
            else:
                dt = h
            
        # Y is vector solution! --> Y[0] = all first components as a function of time
        return Y, t

    # Linearized Clohessy-Wiltshire equations
    def dY1(self, t, Y):

        if self.twoDim:
            x, _, x_dot, y_dot = Y
            Fx, Fy = self.Fs_lin[:, -1]    # Forces are determined inside RK4 (otherwise would happen multiple times per step)

        else:
            x, _, z, x_dot, y_dot, z_dot = Y
            Fx, Fy, Fz = self.Fs_lin[:, -1]    # Forces are determined inside RK4 (otherwise would happen multiple times per step)

        # CW equations
        x_dotdot = 2*self.n*y_dot + 3*self.n**2*x + Fx/self.m
        y_dotdot = -2*self.n*x_dot + Fy/self.m
        if not self.twoDim:
            z_dotdot = -self.n**2*z + Fz/self.m

            # Compute Derivatives
            derivatives = [x_dot, y_dot, z_dot, x_dotdot, y_dotdot, z_dotdot]
        else:
            # Compute Derivatives
            derivatives = [x_dot, y_dot, x_dotdot, y_dotdot]

        return derivatives

    # Exact CR3BP equations
    def dY2(self, t, Y):

        x, y, z, x_dot, y_dot, z_dot = Y

        r2 = np.array([[x, y, z]])
        r = np.array([[self.R, 0, 0]]) + r2
        r1 = r
        r_dot = np.array([x_dot, y_dot, z_dot])
        omega = np.array([[0, 0, self.n]])
        
        # Forces are determined inside RK4 (otherwise would happen multiple times per step)
        r_dotdot = -mu_Earth*r1/(np.linalg.norm(r1)**3) - 2*np.cross(omega, r_dot) -np.cross(omega, np.cross(omega, r)) + self.Fs_ex[:, -1]/self.m
        x_dotdot = r_dotdot[0, 0]
        y_dotdot = r_dotdot[0, 1]
        z_dotdot = r_dotdot[0, 2]

        # Compute Derivatives
        derivatives = [x_dot, y_dot, z_dot, x_dotdot, y_dotdot, z_dotdot]

        return derivatives
    
    # Deep space equations (no gravity, target is inertial frame --> simple a = F/m)
    def dY3(self, t, Y):
        
        if self.twoDim:
            _, _, x_dot, y_dot = Y
            Fx, Fy = self.Fs_ex[:, -1]     # Forces are determined inside RK4 (otherwise would happen multiple times per step)
        else:
            _, _, _, x_dot, y_dot, z_dot = Y
            Fx, Fy, Fz = self.Fs_ex[:, -1]     # Forces are determined inside RK4 (otherwise would happen multiple times per step)

        x_dotdot = Fx/self.m
        y_dotdot = Fy/self.m
        if not self.twoDim:
            z_dotdot = Fz/self.m

            # Compute Derivatives
            derivatives = [x_dot, y_dot, z_dot, x_dotdot, y_dotdot, z_dotdot]
        else:
            derivatives = [x_dot, y_dot, x_dotdot, y_dotdot]

        return derivatives

    # Determine required forces based on current conditions --> Agent decision
    def determineForces(self, Y, t):

        # Only update force commands at large timesteps (subsampled timesteps only used for better dynamics)
        if self.counter % self.subSamples == 0:
            # Forces are determined by agents act() function
            if self.twoDim:
                Fx, Fy, = self.agent.act(Y, t)
                Fz = 0 
            else:
                Fx, Fy, Fz = self.agent.act(Y, t)
        else:
            # If inside subsampled timesteps, just continu previous commands
            Fx, Fy, Fz = self.Fprev

        # Update counter and previous commands
        self.counter += 1
        self.Fprev = [Fx, Fy, Fz]

        # Update thrusting flag (determines subsampling of timesteps)
        if Fx == 0 and Fy == 0 and Fz == 0:
            self.thrusting = False
        else:
            self.thrusting = True
        
        # Transform forces into array and append to force logs
        forces = np.array([[Fx, Fy]]).T if self.twoDim else np.array([[Fx, Fy, Fz]]).T
        if self.linear:
            self.Fs_lin = np.append(self.Fs_lin, forces, axis = 1)
        else:
            self.Fs_ex = np.append(self.Fs_ex, forces, axis = 1)

    # ------------------------------------------------------------------------------------------------------------------------- POST PROCESSING

    # Plots force vectors along different points of the S/C trajectory
    def forcePlot(self, Y_lin, Y_ex, t_lin, t_ex):

        # Unpack simulated data
        if self.twoDim:
            x_lin, y_lin, x_ex, y_ex, y_comb, _ = self.unpackY(Y_lin, Y_ex)
        else:
            x_lin, y_lin, _, x_ex, y_ex, _, y_comb, _ = self.unpackY(Y_lin, Y_ex)

        ax = plt.axes()

        # Target + target orbit
        ax.plot(np.linspace(min(y_comb), max(y_comb), 2), [0, 0], '--', color = color_orbit, zorder = 0)
        ax.scatter([0], [0], color = color_target, marker = "x", s = mSize)

        # Which position along the trajectory to show
        numPoints = 20
        points = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19]

        # Plot linear case
        if t_lin[0] != None:
            for j in points:
                ind = int(j/numPoints*len(t_lin))
                Fvec = self.Fs_lin[:, ind]
                x = x_lin[ind]
                y = y_lin[ind]

                plt.arrow(y, x, Fy, Fx, fc="r", ec="r", head_width = 0.003, head_length = 0.002)

        # Plot exact case
        if t_ex[0] != None:
            for j in points:
                ind = int(j/numPoints*len(t_ex))
                Fvec = self.Fs_ex[:, ind]
                Fx = Fvec[0]/20
                Fy = Fvec[1]/20
                x = x_ex[ind]
                y = y_ex[ind]

                plt.arrow(y, x, Fy, Fx, fc="r", ec="r", head_width = 0.003, head_length = 0.002)
        
        # Plot settings
        ax.set_xlabel(r"$y$ (along-track) [$km$]")
        ax.set_ylabel(r"$x$ (radial) [$km$]")
        ax.set_aspect('equal')
        ax.set_ylim(-0.01, 0.01)
        plt.show()

    # Plots forces, positions and velocities as a funtion of time
    def timePlot(self, Y_lin, Y_ex, t_lin, t_ex, min = False):

        # Unpack simulated data
        if self.twoDim:
            x_lin, y_lin, x_ex, y_ex, _, _ = self.unpackY(Y_lin, Y_ex)
        else:
            x_lin, y_lin, z_lin, x_ex, y_ex, z_ex, _, _ = self.unpackY(Y_lin, Y_ex)

        _, ax = plt.subplots(3)

        # Linear positions
        if t_lin[0] != None:
            t_lin = t_lin.copy()
            t_lin /= 60
            ax[0].plot(t_lin, x_lin, label = r"$x$")
            ax[0].plot(t_lin, y_lin, label = r"$y$")
            if not self.twoDim:
                ax[0].plot(t_lin, z_lin, label = r"$z$")
        # Exact positions
        if t_ex[0] != None:
            t_ex = t_ex.copy()
            t_ex /= 60
            ax[0].plot(t_ex, x_ex, label = r"$x$")
            ax[0].plot(t_ex, y_ex, label = r"$y$")
            if not self.twoDim:
                ax[0].plot(t_ex, z_ex, label = r"$z$")
        # Position plot settings
        ax[0].legend()
        ax[0].set_ylabel(r"Position [$km$]")
        ax[0].get_xaxis().set_ticklabels([])

        # Linear velocities
        if t_lin[0] != None:
            ax[1].plot(t_lin, Y_lin[self.dim], label = r"$\dot{x}$")
            ax[1].plot(t_lin, Y_lin[self.dim + 1], label = r"$\dot{y}$")
            if not self.twoDim:
                ax[1].plot(t_lin, Y_lin[self.dim + 2], label = r"$\dot{z}$")
        # Exact velocities
        if t_ex[0] != None:
            ax[1].plot(t_ex, Y_ex[self.dim], label = r"$\dot{x}$")
            ax[1].plot(t_ex, Y_ex[self.dim + 1], label = r"$\dot{y}$")
            if not self.twoDim:
                ax[1].plot(t_ex, Y_ex[self.dim + 2], label = r"$\dot{z}$")
        # Velocity plot settings
        ax[1].legend()
        ax[1].set_ylabel(r"Velocity [$km/s$]")
        ax[1].get_xaxis().set_ticklabels([])

        # Linear forces
        if t_lin[0] != None:
            ax[2].plot(t_lin, self.Fs_lin[0], label = r"$F_{x}$")
            ax[2].plot(t_lin, self.Fs_lin[1], label = r"$F_{y}$")
            if not self.twoDim:
                ax[2].plot(t_lin, self.Fs_lin[2], label = r"$F_{z}$")
        # Exact forces
        if t_ex[0] != None:
            ax[2].plot(t_ex, self.Fs_ex[0], label = r"$F_{x}$")
            ax[2].plot(t_ex, self.Fs_ex[1], label = r"$F_{y}$")
            if not self.twoDim:
                ax[2].plot(t_ex, self.Fs_ex[2], label = r"$F_{z}$")
        # Force plot settings
        ax[2].legend()
        ax[2].set_ylabel(r"Thrust [$N$]")
        if min:
            ax[2].set_xlabel(r"Time [$min$]")
        else:
            ax[2].set_xlabel(r"Time [$s$]")
        plt.show()

    # Plots spacecraft trajectory (relative to target)
    def orbitPlot(self, Y_lin, Y_ex, t_lin, t_ex):

        # Unpack simulated data
        if self.twoDim:
            x_lin, y_lin, x_ex, y_ex, y_comb, _ = self.unpackY(Y_lin, Y_ex)
        else:
            x_lin, y_lin, _, x_ex, y_ex, _, y_comb, _ = self.unpackY(Y_lin, Y_ex)

        ax = plt.axes()

        # Target + target orbit
        ax.plot(np.linspace(min(y_comb), max(y_comb), 2), [0, 0], '--', color = color_orbit, zorder = 0)
        ax.scatter([0], [0], color = color_target, marker = "x", s = mSize)

        # S/C trajectory
        ax.plot(y_lin, x_lin, label = "Linearized")
        ax.plot(y_ex, x_ex, label = "Exact")

        # Plot settings
        ax.set_xlabel(r"$y$ (along-track) [$km$]")
        ax.set_ylabel(r"$x$ (radial) [$km$]")
        ax.set_ylim(-10, 20)
        ax.set_aspect('equal')
        plt.show()

    # Animate spacecraft trajectory (relative to target)
    def orbitAnimate(self, Y_lin, Y_ex, t_lin, t_ex):

        # Unpack simulated data
        y_comb, x_comb = self.unpackY(Y_lin, Y_ex)[-2:]
        
        # Function called by animator (updates frames)
        def update_plot(num, Y_lin, Y_ex, lines, points, text):
            time = num*self.dt

            if Y_lin[0, 0] != None:
                # Find closest index for this (animation) time step (=/= simulation time step)
                i = (np.abs(np.asarray(t_lin) - time)).argmin()

                # Update trajectory
                lines[0].set_data(np.flip(Y_lin[:2, :i])/1000)

                # Update marker
                points[0].remove()
                marker = self.determineMarker(i, linear = True)
                points[0] = ax.scatter((Y_lin[1, i])/1000, (Y_lin[0, i])/1000, color = color_lin, marker = marker, s = mSize, zorder = 1)

            if Y_ex[0, 0] != None:
                # Find closest index for this (animation) time step (=/= simulation time step)
                i = (np.abs(np.asarray(t_ex) - time)).argmin()
                
                # Update trajectory
                lines[1].set_data(np.flip(Y_ex[:2, :i])/1000)

                # Update marker
                points[1].remove()
                marker = self.determineMarker(i, linear = False)
                points[1] = ax.scatter((Y_ex[1, i])/1000, (Y_ex[0, i])/1000, color = color_ex, marker = marker, s = mSize, zorder = 2)

            # Print elapsed time in fancy format
            text.set_text(self.printTime(time))

            return lines, points

        fig = plt.figure()
        ax = fig.add_subplot()

        # Display agent's fitness
        self.agent.evaluate()
        ax.set_title("Reward = %.2f" %self.agent.reward)

        # Create lines and points initially without data
        lines = [ax.plot([], [], color = color_lin, label = "Linearized")[0], 
                 ax.plot([], [], color = color_ex, label = "Exact")[0]]
        points = [ax.scatter([], [], color = color_lin, marker = 's', s = mSize), 
                  ax.scatter([], [], color = color_ex, marker = 's', s = mSize)]

        # Plot settings
        rangeY = max(y_comb) - min(y_comb)
        rangeX = max(x_comb) - min(x_comb)
        rangeDiff = np.abs(rangeY - rangeX)
        marginY = 0.1*rangeY if rangeY > rangeX else rangeDiff/2 + 0.1*rangeX
        marginX = 0.1*rangeX if rangeX > rangeY else rangeDiff/2 + 0.1*rangeY
        ax.set(xlim = (min(y_comb) - marginY, max(y_comb) + marginY), xlabel = 'X')
        ax.set(ylim = (min(x_comb) - marginX, max(x_comb) + marginX), ylabel = 'Y')
        ax.set_xlabel(r"$y$ (along-track) [$km$]")
        ax.set_ylabel(r"$x$ (radial) [$km$]")
        ax.set_aspect('equal')

        # Target + target orbit
        ax.plot(np.linspace(min(y_comb), max(y_comb), 2), [0, 0], '--', color = color_target, zorder = 0)
        ax.scatter([0], [0], color = color_target, marker = "h", s = mSize)

        # Create timer text
        time = t_ex[0] if t_ex[0] != None else t_lin[0]
        text = ax.text(min(y_comb) - 0.7*marginY, min(x_comb) - 0.7*marginX, self.printTime(time))

        # Animation length --> use fixed time step (as opposed to varying time step in t)
        length = self.n_steps

        # Animate
        anim = animation.FuncAnimation(fig, update_plot, length, fargs = (Y_lin, Y_ex, lines, points, text), interval = 10)
        ax.legend()
        plt.show()
    
    # Print time in fancy format (for orbitAnimate)
    def printTime(self, t):
        if t == None:
            return ""

        # Split time [s] into years + days + hours + minutes + seconds
        years = t//(3600*24*365.25)
        t -= years*(3600*24*365.25)
        days = t//(3600*24)
        t -= days*(3600*24)
        hours = t//3600
        t -= hours*3600
        minutes = t//60
        t -= minutes*60
        seconds = round(t)

        # Print into a single string
        yrs = "" if years == 0 else "%dy " %(years)
        dys = "" if days + years == 0 else "%dd " %(days)
        hrs = "" if hours + days + years == 0 else "%dh " %(hours)
        mns = "" if minutes + hours + days + years == 0 else "%dm " %(minutes)
        scs = "%ds" %(seconds)

        return "t = " + yrs + dys + hrs + mns + scs

    # Define S/C marker based on dominant thrust direction (for orbitAnimate)
    def determineMarker(self, num, linear = True):

        # Find thrusts (at correct time step)
        Fs = self.Fs_lin if linear else self.Fs_ex
        Fx = Fs[0, num]
        Fy = Fs[1, num]
        if not self.twoDim:
            Fz = Fs[2, num]
        else:
            Fz = 0

        # No thrust case --> show square
        if Fx == 0 and Fy == 0 and Fz == 0:
            return "s"
        # Main Thrust in X direction --> arrow is upwards or downwards
        elif abs(Fx) > abs(Fy) and abs(Fx) > abs(Fz):
            return "^" if Fx > 0 else "v"
        # Main Thrust in Y direction --> arrow is left or right
        elif abs(Fy) > abs(Fx) and abs(Fy) > abs(Fz):
            return ">" if Fy > 0 else "<"
        # Main Thrust in Z direction --> arrow is towards or away from viewer
        elif abs(Fz) > abs(Fx) and abs(Fz) > abs(Fy):
            return "+" if Fz > 0 else "*"

    # Unpack Y (= RK4 Solution) into components (positions + velocities)
    def unpackY(self, Y_lin, Y_ex):

        # Unpack linear components and convert to km
        x_lin = Y_lin[0]/1000 if Y_lin[0, 0] != None else Y_lin[0]
        y_lin = Y_lin[1]/1000 if Y_lin[0, 0] != None else Y_lin[1]
        if not self.twoDim:
            z_lin = Y_lin[2]/1000 if Y_lin[0, 0] != None else Y_lin[2]
        # Unpack exact components and convert to km
        x_ex = Y_ex[0]/1000 if Y_ex[0, 0] != None else Y_ex[0]
        y_ex = Y_ex[1]/1000  if Y_ex[0, 0] != None else Y_ex[1]
        if not self.twoDim:
            z_ex = Y_ex[2]/1000  if Y_ex[0, 0] != None else Y_ex[2]

        # Combine linear and exact components (makes it easier to set combined plot settings)
        if Y_lin[0, 0] != None and Y_ex[0, 0] != None:
            y_comb = np.append(y_lin, y_ex)
            x_comb = np.append(x_lin, x_ex)
        elif Y_lin[0, 0] == None and Y_ex[0, 0] == None:
            y_comb = np.array([0])
            x_comb = np.array([0])
        elif Y_lin[0, 0] == None:
            y_comb = y_ex
            x_comb = x_ex
        else:
            y_comb = y_lin
            x_comb = x_lin
        
        if self.twoDim:
            return x_lin, y_lin, x_ex, y_ex, np.append(y_comb, [0]), np.append(x_comb, [0])
        else:
            return x_lin, y_lin, z_lin, x_ex, y_ex, z_ex, np.append(y_comb, [0]), np.append(x_comb, [0])

# =========================================================================================================================================== EVOLUTION

# Generation (= set of individual agents)
class Generation:

    def __init__(self, size, numIndiv, H, m, twoDim = False):
        self.size = size            # Size of agents
        self.numIndiv = numIndiv    # Number of agents per generation
        self.individuals = {}       # Dictionary to store all agents and their reward
        self.H = H                  # S/C orbit altitude
        self.m = m                  # S/C mass
        self.twoDim = twoDim        # Flag whether simulation is 2D or 3D
    
    # ------------------------------------------------------------------------------------------------------------------------- POPULATE
    
    # Populate a generation with random individuals (first gen)
    def populateRandom(self, seed, taus, ws, gs, thetas):
        for i in range(self.numIndiv):

            # Fill each agent's genome with random values
            np.random.seed(678514*seed + 124035*i)
            tau = np.random.uniform(taus[0], taus[1], self.size)
            w = np.random.uniform(ws[0], ws[1], self.size**2)
            g = np.random.uniform(gs[0], gs[1], self.size)
            theta = np.random.uniform(thetas[0], thetas[1], self.size)
            genome = np.concatenate((tau, w, g, theta))

            # Create agent
            agent = NeuralAgent(self.size, genome, twoDim = self.twoDim)
            # Create spacecraft
            spacecraft = Spacecraft(self.H, self.m, agent, twoDim = self.twoDim)

            # Initial fitness is 'infinitely' bad (before it has actually been calculated)
            self.individuals[spacecraft] = -10**10

    # Populate a generation based on the genomes of surviving individuals from the previous generation
    def populateGenomes(self, genomes, taus, ws, gs, thetas, numRandomParents):
        # Add some random parents to gene pool
        for _ in range(numRandomParents):
            tau = np.random.uniform(taus[0], taus[1], self.size)
            w = np.random.uniform(ws[0], ws[1], self.size**2)
            g = np.random.uniform(gs[0], gs[1], self.size)
            theta = np.random.uniform(thetas[0], thetas[1], self.size)
            genome = np.concatenate((tau, w, g, theta))
            genomes += [genome]

        # Keep surviving individuals themselves, and add them to the new generation (to prevent losing good solutions)
        for genome in genomes:
            agent = NeuralAgent(self.size, genome, twoDim = self.twoDim)
            spacecraft = Spacecraft(self.H, self.m, agent, twoDim = self.twoDim)
            self.individuals[spacecraft] = -10**10

        # For the other individuals of the new generation, determine genome based on two parents (either from previous gen, or random parent)
        for _ in range(self.numIndiv - len(genomes)):

            # Take two parents, and mix their genomes
            parents = np.random.randint(0, len(genomes), 2)
            # Also add some Gaussian noise to further enhance mutations
            genome = [np.random.normal(loc = 1, scale = 0.03)*genomes[np.random.choice(parents)][i] for i in range(len(genomes[0]))]

            # ALTERNATIVE METHODE (taken from the Brightspace example, but didn't give good results (not enough diversity))
            # [ Duplicate parents and slightly modify ]
            # genome = genomes[i%len(genomes)]    # Just make copies of parents
            # for j in range(len(genome)):
            #     if np.random.rand() <= 0.05:
            #         genome[j] *= np.random.uniform(-2, 2)    # Mutate some random parts of the genome
            #     else:
            #         genome[j] *= np.random.normal(loc = 1, scale = 0.06)    # Slightly vary genomes
            
            # Create agent and spacecraft based on this mixed genome
            agent = NeuralAgent(self.size, genome, twoDim = self.twoDim)
            spacecraft = Spacecraft(self.H, self.m, agent, twoDim = self.twoDim)
            self.individuals[spacecraft] = -10**10

    # Populate a generation based on a file (containing parents genome information)      
    def populateFromFile(self, fileName, taus, ws, gs, thetas, numRandomParents):

        f = open(fileName, "r")
        genomes = []

        while True:
            # Read genomes from file and add to gene pool
            genomeString = f.readline()[:-1]    # Skip \n character at the end
            if not genomeString:
                break   # EOF
            genomeList = genomeString.strip('][').split(', ')
            genomeList = [float(genomeList[i]) for i in range(len(genomeList))]
            genomes += [genomeList]

        # Use created gene pool to establish new generation
        self.populateGenomes(genomes, taus, ws, gs, thetas, numRandomParents)
    
    # ------------------------------------------------------------------------------------------------------------------------- SIMULATE
    
    # Simulate whole population (i.e. simulate each individual and determine fitness)
    def simulatePopulation(self, Y0, n_steps, dt, gravity = False, linear = False, randomizeStart = False):

        # Determine initial position
        if self.twoDim:
            Y0 = [Y0[0], Y0[1], Y0[3], Y0[4]]
            distance = np.sqrt(Y0[0]**2 + Y0[1]**2)
        else:
            distance = np.sqrt(Y0[0]**2 + Y0[1]**2 + Y0[2]**2)    

        for spacecraft in tqdm(self.individuals.keys()):

            # Randomize start position, so agent doesn't overfit towards a specific starting position
            if randomizeStart:
                scores = []

                # Average scores over 10 random initial positions
                for _ in range(10):

                    # Reset S/C internal states at the start of each run!
                    spacecraft.agent.NN.states = np.array(spacecraft.agent.size*[0.5])

                    # Deep space randomization --> did not give satisfactory results
                    if not gravity:
                        r0 = distance*np.random.normal(loc = 1, scale = 0.03)    # Gaussian distribution for starting distance --> spherical r coordinate
                        # Random spherical angles
                        theta0 = np.random.uniform(low = 0., high = np.pi)
                        phi0 = np.random.uniform(low = 0., high = 2*np.pi)
                        # Convert to cartesian
                        if self.twoDim:
                            x0 = r0*np.sin(phi0)
                            y0 = r0*np.cos(phi0)
                            Y0 = [x0, y0, 0, 0]
                        else:
                            x0 = r0*np.sin(theta0)*np.cos(phi0)
                            y0 = r0*np.sin(theta0)*np.sin(phi0)
                            z0 = r0*np.cos(theta0)
                            Y0 = [x0, y0, z0, 0, 0, 0]
                        
                        spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False)

                    # CW randomization --> less freedom; better results
                    else:
                        y0 = np.random.normal(loc = 100*1000, scale = 5*1000)
                        if self.twoDim:
                            Y0 = [0, y0, 0, 0]
                        else:
                            Y0 = [0, y0, 0, 0, 0, 0]
                        spacecraft.simulate(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, linear = linear, exact = not linear)
                        
                    scores += [spacecraft.agent.evaluate()]
                # Average scores of different runs
                self.individuals[spacecraft] = np.average(scores)

            # Just use fixed starting position
            else:
                # Simulate with or without gravity
                if not gravity:
                    spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False)
                else:
                    spacecraft.simulate(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, linear = linear, exact = not linear)
                # Determine fitness
                self.individuals[spacecraft] = spacecraft.agent.evaluate()

        # Calculate average fitness
        self.average = np.average(list(self.individuals.values()))
        # Calculate best fitness
        self.best = np.max(list(self.individuals.values()))
    
    # ------------------------------------------------------------------------------------------------------------------------- SELECT

    # Rank individuals from a generation, and pick out best ones (should be called after simulating the generation)
    def getBestIndividuals(self, num):

        # Sort all individuals based on their reward
        sortedList = sorted(self.individuals, key = lambda x:self.individuals[x], reverse = True)

        # Get surviving individuals and return their genomes
        best = sortedList[:num]
        bestGenomes = [spacecraft.agent.genome for spacecraft in best]
        bestRewards = [spacecraft.agent.reward for spacecraft in best]

        return bestGenomes, bestRewards
        
        



