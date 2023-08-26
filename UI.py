import csv
import numpy as np
import Simulator as sim
import matplotlib.cm as cmx
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

# =========================================================================================================================================== TESTING

# Simulate a randomly created CTRNN agent (testing purposes)
def simulateRandomCTRNN(i, gravity = False):

    # Genome format: tau[size] - w[size*size] - g[size] - theta[size]
    np.random.seed(i)
    tau = np.random.uniform(10, 100, 20)
    w = np.random.uniform(-10, 10, 20*20)
    g = np.random.uniform(-1, 1, 20)
    theta = np.random.uniform(-1, 1, 20)
    genome = np.concatenate((tau, w, g, theta))
    # Create agent and spacecraft
    agent = sim.NeuralAgent(20, genome)
    spacecraft = sim.Spacecraft(H, m, agent)

    # Orbital scenario --> CW or CR3BP
    if gravity:
        spacecraft.simulate(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = True, linear = True, exact = True)

    # Deep space scenario --> no gravity
    else:
        spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = True, orbitPlot = False, orbitAnim = True)

# Create 50 random CTRNN agents, and find best performing one (testing purposes)
def searchRandomCTRNN():

    reward_max = -10**10
    i_max = -1

    for i in range(50):

        print("%d/50" %(i + 1))

        # Create random CTRNN agent
        np.random.seed(i)
        tau = np.random.uniform(10, 100, 20)
        w = np.random.uniform(-10, 10, 20*20)
        g = np.random.uniform(-1, 1, 20)
        theta = np.random.uniform(-1, 1, 20)
        genome = np.concatenate((tau, w, g, theta))
        # Add agent to spacecraft
        agent = sim.NeuralAgent(20, genome)
        spacecraft = sim.Spacecraft(H, m, agent)
        # Simulate
        Y_ex, _ = spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False)

        # Determine reward, and whether it is best up until now or not
        x, y, z = Y_ex[:3, -1]
        reward = -np.sqrt(x**2 + y**2 + z**2)
        if reward > reward_max:
            i_max = i
            reward_max = reward
    
    # Print which seed (i) gives hight reward --> i can then be used as input for simulateRandomCTRNN
    print(i_max)
    print(reward_max)

# Simulates two generation (one random + one evolutionary) (testing purposes)
def twoGenerations(i):
    size = 20
    numIndiv = 10

    # First generation (random)
    generation = sim.Generation(size, numIndiv, H, m)
    taus = [10, 100]
    ws = [-10, 10]
    gs = [-1, 1]
    thetas = [-1, 1]
    generation.populateRandom(i, taus, ws, gs, thetas)
    generation.simulatePopulation(Y0, n_steps, dt)
    # Determine best fitness scores
    bestGenomes, bestRewards = generation.getBestIndividuals(10)
    print(bestRewards)
    
    # Second generation (based on best individuals of first gen)
    generation2 = sim.Generation(size, numIndiv, H, m)
    generation2.populateGenomes(bestGenomes)
    generation2.simulatePopulation(Y0, n_steps, dt)
    # Determine best fitness scores
    bestGenomes, bestRewards = generation2.getBestIndividuals(10)
    print(bestRewards)

# =========================================================================================================================================== MAIN FUNCTION

# Simulates a given number of generations --> MAIN FUNCTION
def generations(seed, size, numIndiv, numBest, numGen, numRandomParents,
                continued = False, gravity = True, linear = True, random = False):

    # CTRNN settings
    taus = [10, 100]
    ws = [-10, 10]
    gs = [-1, 1]
    thetas = [-1, 1]

    # Flags for file names (for use in sensivity analyses)
    strBest = "" if numBest == 20 else "_Best" + str(numBest)
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)

    # Loop over generations
    for i in range(numGen):

        generation = sim.Generation(size, numIndiv, H, m, twoDim = True)

        # First generation is random
        if i == 0 and not continued:
            generation.populateRandom(seed, taus, ws, gs, thetas)
        # Other option: first generation is based on a generation from a file
        elif i == 0 and continued:
            filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + "_Gen" + str(20) + strBest + strRandom + ".txt"
            generation.populateFromFile(filename, taus, ws, gs, thetas, numRandomParents)
        # Other generations: based on previous generation survivors
        else: 
            generation.populateGenomes(bestGenomes, taus, ws, gs, thetas, numRandomParents)

        # Simulate generation
        generation.simulatePopulation(Y0, n_steps, dt, gravity = gravity, linear = linear, randomizeStart = random)
        # Determine best individuals and print stats
        bestGenomes, _ = generation.getBestIndividuals(numBest)
        print("Generation %d: average = %.2e, best = %.2e" %(i + 1,  generation.average, generation.best))

        # Write average and best scores to file (for later plotting)
        filename = "Data/scores_Seed" + str(seed) + strBest + strRandom + ".csv"
        f = open(filename, "a") if i > 0 else open(filename, "w")
        f.write(str(generation.best) + ',' + str(generation.average) + '\n')

        # Write genomes of best individuals to file (for later plotting)
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + "_Gen" + str(i + 1) + strBest + strRandom + ".txt"
        f = open(filename, "w")
        for genome in bestGenomes:
            f.write(str(list(genome)) + "\n")
        f.close()

# =========================================================================================================================================== PLOTTING

# Determine color from colormap (for plotting)
def getColor(i, maxNum, type = 0):
    values = list(range(0, maxNum + 2))
    colorMap = plt.get_cmap('RdYlGn') if type == 0 else plt.get_cmap('coolwarm')
    cNorm = clr.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colorMap)
    return scalarMap.to_rgba(values[i + 1])

# Show behaviour of best-performing individual of a given generation
def showBest(seed, size, Y0, numBest, numRandomParents,
             twoDim = False, gravity = True, linear = True, generation = -1):
    
    # Flags for file names (for use in sensivity analyses)
    strBest = "" if numBest == 20 else "_Best" + str(numBest)
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)

    # If no generation supplied, use old file format (de facto, only last gen stored)
    if generation == -1:
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + strBest + strRandom + ".txt"
    # Otherwise, pick specific generation file
    else:
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + "_Gen" + str(generation) + strBest + strRandom + ".txt"
    
    # Read best genome (= first on file)
    f = open(filename, "r")
    bestGenomeString = f.readline()[:-1]    # Skip \n character at the end
    bestGenomeList = bestGenomeString.strip('][').split(', ')
    bestGenomeList = [float(bestGenomeList[i]) for i in range(len(bestGenomeList))]

    # Create agent with this genome
    agent = sim.NeuralAgent(size, bestGenomeList, twoDim = twoDim)
    spacecraft = sim.Spacecraft(H, m, agent, twoDim = twoDim)

    # Determine initial position
    if twoDim:
        Y0 = [Y0[0], Y0[1], Y0[3], Y0[4]]
    
    # Simulate behaviour of this agent, and plot desired graphs
    if not gravity:
        spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = True, orbitPlot = True, orbitAnim = True, forcePlot = True)
    else:
        spacecraft.simulate(Y0, n_steps, dt, timePlot = True, orbitPlot = True, orbitAnim = True,
                             linear = linear, exact = not linear)

# Show trajectory for best-performing individual in a number of generations
def showAll(seed, size, Y0, numBest, numRandomParents,
            twoDim = False, gravity = True, linear = True, step = 1):

    # Flags for file names (for use in sensivity analyses)
    strBest = "" if numBest == 20 else "_Best" + str(numBest)
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)
    
    end = False
    if twoDim:
        Y0 = [Y0[0], Y0[1], Y0[3], Y0[4]]
    ax = plt.axes()

    # Target + target orbit
    if gravity:
        ax.plot(np.linspace(-20, 110, 2), [0, 0], '--', color = sim.color_orbit, zorder = 0, label = "Target Orbit")
    ax.scatter([0], [0], color = sim.color_target, marker = "x", s = 2*sim.mSize, label = "Target")

    # Generation counter
    i = 0
    i0 = 0

    print("0    5    10   15   20   25")
    print("|----|----|----|----|----|")
    print("#", end = "")

    # Generations which should actually be displayed
    generations = [0, 1, 2, 3, 4, 19]

    while not end:
        
        if gravity:
            i = generations[i0]

        # Open file for this generation
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + "_Gen" + str(i + 1) + strBest + strRandom + ".txt"
        try:
            f = open(filename, "r")
        except:
            end = True  # No such file --> end of the generations reached
            break

        # Get genome of best-performing individual of this generation
        bestGenomeString = f.readline()[:-1]    # Skip \n character at the end
        bestGenomeList = bestGenomeString.strip('][').split(', ')
        bestGenomeList = [float(bestGenomeList[i]) for i in range(len(bestGenomeList))]

        # Create agent based on this genome
        agent = sim.NeuralAgent(size, bestGenomeList, twoDim = twoDim)
        spacecraft = sim.Spacecraft(H, m, agent, twoDim = twoDim, subSamples = 100)
        
        # Simulate agent's behaviour
        if not gravity:
            Y_lin, Y_ex, _, _ = spacecraft.simulateNoGrav(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False)
        else:
            Y_lin, Y_ex, _, _ = spacecraft.simulate(Y0, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, linear = linear, exact = not linear)

        # Unpack simulated data
        if twoDim:
            x_lin, y_lin, x_ex, y_ex, _, _ = spacecraft.unpackY(Y_lin, Y_ex)
        else:
            x_lin, y_lin, _, x_ex, y_ex, _, _, _ = spacecraft.unpackY(Y_lin, Y_ex)

        # Plot trajectory
        if not linear or not gravity:
            scale = 1000 if not gravity else 1
            ax.plot(y_ex*scale, x_ex*scale, label = "Gen " + str(i + 1), color = getColor(i//step, 20//step))
        else:
            ax.plot(y_lin, x_lin, label = "Gen " + str(i + 1), color = getColor(i0, len(generations)))
        
        print(step*"#", end = "")

        if gravity and i0 == len(generations) - 1:
            break
        i += step
        i0 += 1
    
    # Plot settings
    if gravity: 
        ax.set_xlabel(r"$y$ (along-track) [$km$]")
        ax.set_ylabel(r"$x$ (radial) [$km$]")
    else:
        ax.set_xlabel(r"$y$ (along-track) [$m$]")
        ax.set_ylabel(r"$x$ (radial) [$m$]")
    ax.legend()
    if gravity:
        ax.set_ylim(-5, 15)
    plt.show()

# Plot behaviour of best-perfomring individual in a given generation, for Gaussian starting positions (+- 2*sigma, +- sigma, and nominal)
def showTwoSigma(seed, size, Y0, sigma, numBest, numRandomParents,
                 twoDim = False, gravity = True, linear = True, generation = -1):

    # Flags for file names (for use in sensivity analyses)
    strBest = "" if numBest == 20 else "_Best" + str(numBest)
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)
    
    # Nominal starting position
    if twoDim:
        Y0 = [Y0[0], Y0[1], Y0[3], Y0[4]]

    ax = plt.axes()
    # Target + target orbit
    if gravity:
        ax.plot(np.linspace(-20, 110, 2), [0, 0], '--', color = sim.color_orbit, zorder = 0, label = "Target Orbit")
    ax.scatter([0], [0], color = sim.color_target, marker = "x", s = 2*sim.mSize, label = "Target")

    # If no generation supplied, use old file format (de facto, only last gen stored)
    if generation == -1:
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + strBest + strRandom + ".txt"
    # Otherwise, pick specific generation file
    else:
        filename = "Data/bestIndividualsLastGeneration_Seed" + str(seed) + "_Gen" + str(generation) + strBest + strRandom + ".txt"
    
    # Read best genome (= first on file)
    f = open(filename, "r")
    bestGenomeString = f.readline()[:-1]    # Skip \n character at the end
    bestGenomeList = bestGenomeString.strip('][').split(', ')
    bestGenomeList = [float(bestGenomeList[i]) for i in range(len(bestGenomeList))]

    # Different starting positions (+- 2*sigma, +- sigma, and nominal)
    dYs = [-2*sigma, -sigma, 0, sigma, 2*sigma]
    labels = [r"$-2\sigma$", r"$-\sigma$", "Nominal", r"$\sigma$", r"$2\sigma$"]

    # Loop over starting positions
    for i in range(len(dYs)):
        
        # Recreate agent every time, so that it fully resets
        agent = sim.NeuralAgent(size, bestGenomeList, twoDim = twoDim)
        spacecraft = sim.Spacecraft(H, m, agent, twoDim = twoDim, subSamples = 100)

        # Set starting position
        Y1 = Y0.copy()
        for j in range(len(Y0)):
            if Y0[j] != 0:
                Y1[j] += dYs[i]
        
        # Simulate agent's behaviour
        if not gravity:
            Y_lin, Y_ex, _, _ = spacecraft.simulateNoGrav(Y1, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False)
        else:
            Y_lin, Y_ex, _, _ = spacecraft.simulate(Y1, n_steps, dt, timePlot = False, orbitPlot = False, orbitAnim = False, linear = linear, exact = not linear)

        # Unpack simulated data
        if twoDim:
            x_lin, y_lin, x_ex, y_ex, _, _ = spacecraft.unpackY(Y_lin, Y_ex)
        else:
            x_lin, y_lin, _, x_ex, y_ex, _, _, _ = spacecraft.unpackY(Y_lin, Y_ex)

        # Plot trajectory
        if not linear or not gravity:
            scale = 1000 if not gravity else 1
            ax.plot(y_ex*scale, x_ex*scale, label = labels[i], color = getColor(i, len(dYs)))
        else:
            ax.plot(y_lin, x_lin, label = labels[i], color = getColor(i, len(dYs)))
    
    # Plot settings
    if gravity: 
        ax.set_xlabel(r"$y$ (along-track) [$km$]")
        ax.set_ylabel(r"$x$ (radial) [$km$]")
    else:
        ax.set_xlabel(r"$y$ (along-track) [$m$]")
        ax.set_ylabel(r"$x$ (radial) [$m$]")
    ax.legend()
    if gravity:
        ax.set_ylim(-5, 15)
    plt.show()

# Plot evolutionary progress of average and best fitness scores
def plotProgress(seed, gravity, numBest, numRandomParents):

    # Flags for file names (for use in sensivity analyses)
    strBest = "" if numBest == 20 else "_Best" + str(numBest)
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)

    # Read scores file
    filename = "Data/scores_Seed" + str(seed) + strBest + strRandom + ".csv"
    f = open(filename, "r")
    reader = csv.reader(f, delimiter = ",")
    out = []
    for row in reader:
        out += [[float(i) for i in row]]
    out = np.array(out).T
    generations = [i + 1 for i in range(len(out[0]))]

    # Plot on same figure
    ax = plt.axes()
    ax2 = ax.twinx()

    # Plot evolution of scores
    lns1 = ax.plot(generations, out[0], label = "Best Fitness", color = "C1")
    lns2 = ax2.plot(generations, out[1], label = "Average Fitness", color = "C2")
    lns3 = ax.plot(generations, len(generations)*[0], "--", label = "Fitness Limit", color = "C0")

    # Plot settings
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.set_xlabel("Generation [-]")
    ax.set_ylabel("Best Fitness Score [-]")
    ax2.set_ylabel("Average Fitness Score [-]")
    ax.legend(lns, labs)
    ax.set_xticks(generations)
    ax.grid(axis = "x", linestyle = ":")

    # Align axes zeros
    y_lims = np.array([ax.get_ylim() for ax in [ax, ax2]])
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)

    # Set scientific format
    if gravity:
        ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
        ax2.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))

    plt.subplots_adjust(right = 0.85)
    plt.show()

# Plot evolutionary progress of average and best fitness scores, for different values of numBest
def plotSensitivityBest(seed, gravity, numBests, numRandomParents):

    # Separate figures for best and average scores (otherwise too cluttered)
    ax = plt.figure(0).gca()
    ax2 = plt.figure(1).gca()
    i = 0

    # Flag for correct file name
    strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)
    
    # Loop over all values of numBest
    for numBest in numBests:

        # Flag for correct file name
        strBest = "" if numBest == 20 else "_Best" + str(numBest)

        # Open correct score file and read data
        filename = "Data/scores_Seed" + str(seed) + strBest + strRandom + ".csv"
        f = open(filename, "r")
        reader = csv.reader(f, delimiter = ",")
        out = []
        for row in reader:
            out += [[float(i) for i in row]]
        out = np.array(out).T
        generations = [i + 1 for i in range(len(out[0]))]

        # Plot evolution of scores
        ax.plot(generations, out[0], label = "numBest = " + str(numBest), color = getColor(i, len(numBests), type = 1))
        ax2.plot(generations, out[1], label = "numBest = " + str(numBest), color = getColor(i, len(numBests), type = 1))
        i += 1

    # Plot settings
    ax.set_xlabel("Generation [-]")
    ax2.set_xlabel("Generation [-]")
    ax.set_ylabel("Best Fitness Score [-]")
    ax2.set_ylabel("Average Fitness Score [-]")
    ax.legend()
    ax2.legend()
    ax.set_xticks(generations)
    ax2.set_xticks(generations)
    ax.grid(axis = "x", linestyle = ":")
    ax2.grid(axis = "x", linestyle = ":")

    # Set logarithmic scales
    ax.set_yscale('symlog')
    ax2.set_yscale('symlog')
    ax.set_yticks([-2*10**3, -4*10**3, -6*10**3, -8*10**3, -10**4, -2*10**4])
    ax2.set_yticks([-1*10**4, -2*10**4, -4*10**4, -6*10**4, -8*10**4, -10**5, -2*10**5])
    ax.get_yaxis().set_major_formatter(tkr.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(tkr.ScalarFormatter())
    if numRandomParents != 2:
        ax2.set_ylim(top = -0.7*10**4)

    # Set scientific format
    if gravity:
        ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
        ax2.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))

    plt.show()

# Plot evolutionary progress of average and best fitness scores, for different values of numRandom
def plotSensitivityRandom(seed, gravity, numRandomParentss):

    # Separate figures for best and average scores (otherwise too cluttered)
    ax = plt.figure(0).gca()
    ax2 = plt.figure(1).gca()
    i = 0
    
    # Loop over all values of numRandom
    for numRandomParents in numRandomParentss:

        # Flag for correct file name
        strRandom = "" if numRandomParents == 2 else "_Random" + str(numRandomParents)

        # Open correct score file and read data
        filename = "Data/scores_Seed" + str(seed) + strRandom + ".csv"
        f = open(filename, "r")
        reader = csv.reader(f, delimiter = ",")
        out = []
        for row in reader:
            out += [[float(i) for i in row]]
        out = np.array(out).T
        generations = [i + 1 for i in range(len(out[0]))]

        # Plot evolution of scores
        ax.plot(generations, out[0], label = "numRandom = " + str(numRandomParents), color = getColor(i, len(numRandomParentss), type = 1))
        ax2.plot(generations, out[1], label = "numRandom = " + str(numRandomParents), color = getColor(i, len(numRandomParentss), type = 1))
        i += 1

    # Plot settings
    ax.set_xlabel("Generation [-]")
    ax2.set_xlabel("Generation [-]")
    ax.set_ylabel("Best Fitness Score [-]")
    ax2.set_ylabel("Average Fitness Score [-]")
    ax.legend()
    ax2.legend()
    ax.set_xticks(generations)
    ax2.set_xticks(generations)
    ax.grid(axis = "x", linestyle = ":")
    ax2.grid(axis = "x", linestyle = ":")

    # Set logarithmic scales
    ax.set_yscale('symlog')
    ax2.set_yscale('symlog')
    ax.set_yticks([-2*10**3, -4*10**3, -6*10**3, -8*10**3, -10**4, -2*10**4])
    ax2.set_yticks([-1*10**4, -2*10**4, -4*10**4, -6*10**4, -8*10**4, -10**5, -2*10**5])
    ax.get_yaxis().set_major_formatter(tkr.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(tkr.ScalarFormatter())
    ax.set_ylim(top = -1.2*10**3)
    ax2.set_ylim(top = -0.8*10**4)

    # Set scientific format
    if gravity:
        ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
        ax2.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))

    plt.show()


# =========================================================================================================================================== INPUTS

# Target Orbit Altitude
H = 700*1000    # [m]
# Spacecraft Mass
m = 500         # [kg]

# Initial condition: no relative velocity, only relative position
#                                                   X --> relative radial position
#                                                   Y --> relative along-track position
#                                                   Z --> relative cross-track position
Y0 = [0, 100*1000, 0, 0, 0, 0]  # [m]
T = 200*60                      # [s] total sim time
n_steps = 50                    # Number of rough scale timesteps --> are divided by 100 (or less) when the S/C is thrusting!
dt = T/n_steps

numGen = 20             # Number of generations
numIndiv = 200          # Number of individuals per generation
numBest = 20            # Number of survivors (= best-performing individuals) at the end of each generation
numRandomParents = 2    # Number of random parents at the start of each generation
size = 20               # Size of each individual (= number of nodes in the CTRNN)
seed = 25               # Seed used in RNG

gravity = True          # Flag whether to use orbital (True) or deep-space (False) scenario
linear = True           # Flag whether to use linearized (True) or exact (False) equations --> not really used in the paper
random = False          # Flag whether intial positions should be randomized (True) or kept fixed (False)

sigma = 5000            # [m] Standard deviation used for Gaussian initial positions


# =========================================================================================================================================== INTERFACE


# generations(seed, size, numIndiv, numBest, numGen, numRandomParents, continued = False, gravity = gravity, linear = linear, random = random)

showBest(seed, size, Y0, numBest, numRandomParents,
         twoDim = True, gravity = gravity, linear = linear, generation = 20)
# showAll(seed, size, Y0, numBest, numRandomParents,
#         twoDim = True, gravity = gravity, linear = linear, step = 3)
# showTwoSigma(seed, size, Y0, sigma, numBest, numRandomParents,
#              twoDim = True, gravity = gravity, linear = linear, generation = 20)

# plotProgress(seed, gravity, numBest, numRandomParents)
# plotSensitivityBest(seed, gravity, numBests = [2, 5, 20, 50, 100], numRandomParents = 0)
# plotSensitivityRandom(seed, gravity, numRandomParentss = [0, 2, 5, 10, 20])



# =========================================================================================================================================== NOTES

# seed 6 (size 20, 100 generations of 100 individuals)
# seed 9 (size 10, 10 generations of 50 individuals)
# seed 10 (size 10, 20 generations of 200 individuals)
# seed 12 (size 50, 20 generations of 100 individuals --> relPos**3*i**2 reward) 
#                   ==> decent result!
# seed 13 (size 50, 20 generations of 100 individuals --> relPos**3*i**2 + Vel**2*i**4 after 75% reward)
#                   ==> decent result!
# seed 17 (size 50, 20 generations of 200 individuals (20 best, 0 random) - 10-100tau -10-10w - same reward - 10x randomized Y0 - mixing + keeping parents - output *= 1)
#                   ==> reacts a bit to its environment

# ----------------------------------------------------------------------------------------------------------------------------------- SEEDS USED IN REPORT |
#                                                                                                                                                          V
# seed 20: (size 50, 20 gen of 200; 20 best, 0 random) - 10-100tau -10-10w - mixing + keeping parents - output *= 1
#                                                        2D, reward = -endPos, no randomized Y0, T = 1500
#                   ==> converges nicely, ending up at target (with high velocity)

#           if randomizing Y0 (with same settings as above), doesn't really work (even if #individuals bumped up to 2000; or with copying parents;
#                                                                                or with numBest = 5; or with 2 random; or with size = 200 or 10...)
#                   ==> skip Y0 randomization for now.

# seed 21: (size 10, 20 gen of 200; 20 best, 2 random) - 10-100tau -10-10w - mixing + keeping parents - output *= 1)
#                                                        2D, reward = -endPos - endVel*1000, no randomized Y0, T = 1500
#                   ==> perfect behaviour: ending up at target with no velocity

# seed 25: (size 20, 20 gen of 200; 20 best, 2 random) - 10-100tau -10-10w - mixing + keeping parents - output *= 10) --> Y0 at 0,100k
#                                                        2D CW, reward = -endPos - endVel*1000, no randomized Y0, T = 200*60
#                   ==> exactly what it should be (roughly 2 orbits sim, so gets there via double loop)

#           Sensitivity analysis: numBest = 2 - 5 - 20 - 50 - 100
#                                 numRandom = 0 - 2 - 5 - 10 - 20

# seed 26: (size 10, 20 gen of 200; 20 best, 2 random) - 10-100tau -10-10w - mixing + keeping parents - output *= 10) --> Y0 at 0,100k ; 200min sim
#                                                        2D CW, reward = -endPos**2 - endVel*100k, slightly randomized Y0 (normal around 100k with sigma = 5k)
#                   ==> S/C adapts to starting location (if within 2*sigma range)


