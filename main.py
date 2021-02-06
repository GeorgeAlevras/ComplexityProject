import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import os


def threshold_prob(p=0.5, n=1):
    """ Returns random threshold value by determining the probability for each """
    elements = [1, 2]  # The values the discrete random variable can take: {1, 2}
    probs = [p, 1-p]  # Probabilities for each value in the same order
    return np.random.choice(elements, n, p=probs)  # n denotes the size of the array


def initialise(size=4, p=0.5):
    heights = np.zeros(size)
    slopes = np.zeros(size)
    thresholds = threshold_prob(p, n=size)
    return heights, slopes, thresholds


def update_slopes(heights):
    return abs(np.diff(heights, append=[0]))


def drive_and_relax(heights, slopes, thresholds, grains=16, p=0.5):
    avalanches = []  # List holding the avalanche sizes
    h_1 = []  # List holding the time-series heights at site s=1
    steady_state = False  # Model starts at transient state
    steady_state_time = 0  # Time of steady-state
    height_configs = [heights]  # Configurations of heights

    for g in range(grains):
        heights_sum = np.sum(heights)
        avalanche_size = 0  # Initialise avalanche
        heights[0] += 1  # Drive system by adding 1 grain at site i=1
        slopes = update_slopes(heights)
        if ((slopes - thresholds) <= 0).all():  # If all slopes are valid, drive again
            avalanches.append(avalanche_size)
        else:
            while not ((slopes - thresholds) <= 0).all():  # Keeps relaxing supercritical sites until all slopes are valid
                for i in range(len(heights)):
                    if slopes[i] > thresholds[i]:
                        heights[i] -= 1
                        if i != len(heights) - 1:
                            heights[i+1] += 1
                        slopes = update_slopes(heights)
                        thresholds[i] = threshold_prob(p=p, n=1)[0]
                        avalanche_size += 1
                        if i == len(heights) - 1:
                            steady_state = True  # System is in steady-state when last site is relaxed for first time
                    else:
                        pass
            avalanches.append(avalanche_size)
        
        if steady_state and steady_state_time == 0:
            # steady_state_time = g - 1
            steady_state_time = heights_sum
        
        h_1.append(heights[0])
        height_configs.append(heights.tolist())

    height_configs = np.array(height_configs)
    configs_counted = Counter(map(tuple, height_configs))  # make a map counting occurence of each configuration of heights
    no_reccurant = len([k for k in configs_counted.keys() if configs_counted[k]>1])
    return heights, slopes, thresholds, h_1, steady_state_time, no_reccurant, avalanches


def task_1(compute=True, plot=False):
    if compute:
        """ Plotting the heights themselves """
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=10000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Pile Visualised: L=64, p=0.5'), np.array(heights))  # Save data in a .npy file
        print('Height at i=1:', heights[0], 'sum of slopes:', np.sum(slopes))  # Ensuring height at i=1 equivalent to sum of slopes

        """ Changing the probability for the same size of the lattice"""
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights_0_5, slopes_0_5, thresholds_0_5, h_1_0_5, sst_0_5, reccur, avalanches_0_5 = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=0.5'), np.array(h_1_0_5))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=16, p=0)
        heights_0, slopes_0, thresholds_0, h_1_0, sst_0, reccur, avalanches_0 = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=0'), np.array(h_1_0))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=16, p=1)
        heights_1, slopes_1, thresholds_1, h_1_1, sst_1, reccur, avalanches_1 = drive_and_relax(heights, slopes, thresholds, grains=4000, p=1)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=1'), np.array(h_1_1))  # Save data in a .npy file
    
        """ Demonstrating the bounding of the slopes by the thresholds """
        heights, slopes, thresholds = initialise(size=32)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=1000)
        np.save(os.path.join('Numpy Files', 'Slopes Bounded: L=32, p=0.5 Slopes'), np.array(slopes))  # Save data in a .npy file
        np.save(os.path.join('Numpy Files', 'Slopes Bounded: L=32, p=0.5 Thresholds'), np.array(thresholds))  # Save data in a .npy file
    
        """ Checking the height at the first site fir L = 16, 32 """
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0.5)
        np.save(os.path.join('Numpy Files', 'First Site Height: L=16, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=32, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0.5)
        np.save(os.path.join('Numpy Files', 'First Site Height: L=32, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file

    if plot:
        """ Plotting the heights themselves """
        heights = np.load('Pile Visualised: L=64, p=0.5.npy', allow_pickle=True)
        plt.plot(heights, label='Heights at each lattice')
        plt.xlabel('Site Location', fontname='Times New Roman', fontsize=12)
        plt.ylabel('Height at site i=1', fontname='Times New Roman', fontsize=12)
        plt.legend()
        plt.savefig('Plots/Task1/pile_visualised_L=64,p=0.5.png')
        plt.show()

        """ Changing the probability for the same size of the lattice"""
        h_1_0 = np.load('Changing Probabilities: L=16, p=0.npy', allow_pickle=True)
        h_1_0_5 = np.load('Changing Probabilities: L=16, p=0.5.npy', allow_pickle=True)
        h_1_1 = np.load('Changing Probabilities: L=16, p=1.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_0, label=r'$p=0, \langle{steady-state}\rangle=$' + str(np.average(h_1_0[300:])))
        plt.plot(h_1_0_5, label=r'$p=0.5, \langle{steady-state}\rangle=$' + str(round(np.average(h_1_0_5[300:]), 5)))
        plt.plot(h_1_1, label=r'$p=1, \langle{steady-state}\rangle=$' + str(np.average(h_1_1[300:])))
        plt.xlabel('$\it{t}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{h(t; L)}$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.ylim(0, 34)
        plt.xlim(-200, 4100)
        plt.savefig('Plots/Task1/chaning_probability.png')
        plt.show()

        """ Demonstrating the bounding of the slopes by the thresholds """
        sites = np.linspace(1, 32, 32)
        slopes = np.load('Slopes Bounded: L=32, p=0.5 Slopes.npy', allow_pickle=True)
        thresholds = np.load('Slopes Bounded: L=32, p=0.5 Thresholds.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(sites, slopes, 'o', label=r'$Actual Slopes$')
        plt.plot(sites, thresholds, 'x', label=r'$Threshold Slopes$')
        plt.xlabel('$\it{i}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel('$\it{z_i}$', fontname='Times New Roman', fontsize=20)
        plt.xlim([0, 33])
        plt.ylim([0, 2.2])
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.savefig('Plots/Task1/slopes_bounded.png')
        plt.show()

        """ Checking the height at the first site fir L = 16, 32 """
        h_1_16, sst_16 = np.load('First Site Height: L=16, p=0.5.npy', allow_pickle=True)
        h_1_32, sst_32 = np.load('First Site Height: L=32, p=0.5.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_16, label=r'$L=16, \langle{steady-state}\rangle = $' + str(round(np.average(h_1_16[sst_16:]), 5)))
        plt.plot(h_1_32, label=r'$L=32, \langle{steady-state}\rangle = $' + str(round(np.average(h_1_32[sst_32:]), 5)))
        plt.xlabel('$\it{t}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{h(t; L)}$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.ylim(0, 62)
        plt.xlim(-200, 4100)
        plt.savefig('Plots/Task1/heights_at_site_1.png')
        plt.show()


def check_recurrent_configs(size=4, grains=100000):
    heights, slopes, thresholds = initialise(size=size, p=0.5)
    heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=grains, p=0.5)
    return reccur


def task_2_a(compute=True, plot=False):
    if compute:
        heights, slopes, thresholds = initialise(size=4, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=4, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=8, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=8, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=16, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=32, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=32, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=64, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=128, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=128, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=256, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Task 2a: L=256, p=0.5'), np.array([h_1, sst]))  # Save data in a .npy file

    if plot:
        h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
        h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
        h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
        h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
        h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
        h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)
        h_1_256, sst_256 = np.load('Task 2a: L=256, p=0.5.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_4, label=r'$L=4, \langle{t_c(L=4)}\rangle  = $' + str(sst_4))
        plt.plot(h_1_8, label=r'$L=8, \langle{t_c(L=8)}\rangle  = $' + str(sst_8))
        plt.plot(h_1_16, label=r'$L=16, \langle{t_c(L=16)}\rangle  = $' + str(sst_16))
        plt.plot(h_1_32, label=r'$L=32, \langle{t_c(L=32)}\rangle  = $' + str(sst_32))
        plt.plot(h_1_64, label=r'$L=64, \langle{t_c(L=64)}\rangle  = $' + str(sst_64))
        plt.plot(h_1_128, label=r'$L=128, \langle{t_c(L=128)}\rangle  = $' + str(sst_128))
        plt.plot(h_1_256, label=r'$L=256, \langle{t_c(L=258)}\rangle  = $' + str(sst_256))
        plt.xlabel('$\it{t}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{h(t; L)}$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-2000, 64000)
        plt.ylim(0, 460)
        plt.savefig('Plots/Task2/task2a.png')
        plt.show()


def task_2_b(compute=True, plot=False):
    l = [4, 8, 16, 32, 64, 128, 256]
    avg_cross_over_times = []
    t = 0
    
    if compute:
        for i in range(5):
            heights, slopes, thresholds = initialise(size=4, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=500, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 4')
        t = 0
        for i in range(5):
            heights, slopes, thresholds = initialise(size=8, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=500, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 8')
        t = 0
        for i in range(5):
            heights, slopes, thresholds = initialise(size=16, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=1000, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 16')
        t = 0
        for i in range(5):
            heights, slopes, thresholds = initialise(size=32, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=2000, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 32')
        t = 0
        for i in range(5):       
            heights, slopes, thresholds = initialise(size=64, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 64')
        t = 0
        for i in range(5):
            heights, slopes, thresholds = initialise(size=128, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=15000, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 128')
        t = 0
        for i in range(5):
            heights, slopes, thresholds = initialise(size=256, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=58000, p=0.5)
            t += sst
        avg_cross_over_times.append(t/5)
        print('Done with 256')
        np.save(os.path.join('Numpy Files', 'Task 2b'), np.array(avg_cross_over_times))  # Save data in a .npy file

    if plot:
        avg_cross_over_times = np.load('Task 2b.npy', allow_pickle=True)
        print(avg_cross_over_times.tolist())
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(l, avg_cross_over_times, 'o')
        plt.xlabel(r'$\it{L}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\langle{t_c}\rangle$', fontname='Times New Roman', fontsize=18)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.ylim(-2000, 58000)
        plt.xlim(0, 260)
        plt.savefig('Plots/Task2/task2b.png')
        plt.show()


def task_2_d(compute=True, plot=False):
    if compute:
        heights_4 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=4, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_4 += h_1
        heights_4 = 0.2*heights_4
        np.save(os.path.join('Numpy Files', 'Task 2d: L=4, p=0.5'), heights_4)  # Save data in a .npy file
        print('Done with 4')
        heights_8 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=8, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_8 += h_1
        heights_8 = 0.2*heights_8
        np.save(os.path.join('Numpy Files', 'Task 2d: L=8, p=0.5'), heights_8)  # Save data in a .npy file
        print('Done with 8')
        heights_16 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=16, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_16 += h_1
        heights_16 = 0.2*heights_16
        np.save(os.path.join('Numpy Files', 'Task 2d: L=16, p=0.5'), heights_16)  # Save data in a .npy file
        print('Done with 16')
        heights_32 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=32, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_32 += h_1
        heights_32 = 0.2*heights_32
        np.save(os.path.join('Numpy Files', 'Task 2d: L=32, p=0.5'), heights_32)  # Save data in a .npy file
        print('Done with 32')
        heights_64 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=64, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_64 += h_1
        heights_64 = 0.2*heights_64
        np.save(os.path.join('Numpy Files', 'Task 2d: L=64, p=0.5'), heights_64)  # Save data in a .npy file
        print('Done with 64')
        heights_128 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=128, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_128 += h_1
        heights_128 = 0.2*heights_128
        np.save(os.path.join('Numpy Files', 'Task 2d: L=128, p=0.5'), heights_128)  # Save data in a .npy file
        print('Done with 128')
        heights_256 = np.zeros(64000)
        for i in range(5):
            heights, slopes, thresholds = initialise(size=256, p=0.5)
            heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
            heights_256 += h_1
        heights_256 = 0.2*heights_256
        np.save(os.path.join('Numpy Files', 'Task 2d: L=256, p=0.5'), heights_256)  # Save data in a .npy file
        print('Done with 256')
    
    if plot:
        t = np.linspace(1, 64000, 64000)
        t_4 = t/(4**2)
        t_8 = t/(8**2)
        t_16 = t/(16**2)
        t_32 = t/(32**2)
        t_64 = t/(64**2)
        t_128 = t/(128**2)
        t_256 = t/(256**2)
        h_1_4 = np.load('Task 2d: L=4, p=0.5.npy', allow_pickle=True)
        h_1_4 /= 4
        h_1_8 = np.load('Task 2d: L=8, p=0.5.npy', allow_pickle=True)
        h_1_8 /= 8
        h_1_16 = np.load('Task 2d: L=16, p=0.5.npy', allow_pickle=True)
        h_1_16 /= 16
        h_1_32 = np.load('Task 2d: L=32, p=0.5.npy', allow_pickle=True)
        h_1_32 /= 32
        h_1_64 = np.load('Task 2d: L=64, p=0.5.npy', allow_pickle=True)
        h_1_64 /= 64
        h_1_128 = np.load('Task 2d: L=128, p=0.5.npy', allow_pickle=True)
        h_1_128 /= 128
        h_1_256 = np.load('Task 2d: L=256, p=0.5.npy', allow_pickle=True)
        h_1_256 /= 256
        t_s = np.linspace(0, 0.88, 881)
        x_sq = 1.75*np.sqrt(t_s)
        fig, ax = plt.subplots()
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(t_4, h_1_4, label=r'$L=4$')
        plt.plot(t_8, h_1_8, label=r'$L=8$')
        plt.plot(t_16, h_1_16, label=r'$L=16$')
        plt.plot(t_32, h_1_32, label=r'$L=32$')
        plt.plot(t_64, h_1_64, label=r'$L=64$')
        plt.plot(t_128, h_1_128, label=r'$L=128$')
        plt.plot(t_256, h_1_256, label=r'$L=256$')
        plt.plot(t_s, x_sq, '--', label=r'$y= 1.72\sqrt{t}$', alpha=0.9, linewidth=2, color='k')
        plt.xlabel('$\it{t}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{\tilde h(t; L)}$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-0.1, 1.25)
        plt.ylim(0, 2)
        plt.savefig('Plots/Task2/task2d.png')
        plt.show()


def task_2_e():
    h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
    h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
    h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
    h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
    h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
    h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)
    h_1_256, sst_256 = np.load('Task 2a: L=256, p=0.5.npy', allow_pickle=True)

    h_4_avg = np.average(h_1_4[sst_4:])
    h_8_avg = np.average(h_1_8[sst_8:])
    h_16_avg = np.average(h_1_16[sst_16:])
    h_32_avg = np.average(h_1_32[sst_32:])
    h_64_avg = np.average(h_1_64[sst_64:])
    h_128_avg = np.average(h_1_128[sst_128:])
    h_256_avg = np.average(h_1_256[sst_256:])
    h_4_std = np.std(h_1_4[sst_4:])
    h_8_std = np.std(h_1_8[sst_8:])
    h_16_std = np.std(h_1_16[sst_16:])
    h_32_std = np.std(h_1_32[sst_32:])
    h_64_std = np.std(h_1_64[sst_64:])
    h_128_std = np.std(h_1_128[sst_128:])
    h_256_std = np.std(h_1_256[sst_256:])

    sites = [4, 8, 16, 32, 64, 128, 256]
    fig, ax = plt.subplots()
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    plt.plot(sites[0], h_4_avg, 'o', label=r'$L=4,$' + ' ' + r'$std=$' + str(round(h_4_std, 4)))
    plt.plot(sites[1], h_8_avg, 'o', label=r'$L=8,$' + ' ' +  r'$std=$' + str(round(h_8_std, 4)))
    plt.plot(sites[2], h_16_avg, 'o', label=r'$L=16,$' + ' ' +  r'$std=$' + str(round(h_16_std, 4)))
    plt.plot(sites[3], h_32_avg, 'o', label=r'$L=32,$' + ' ' +  r'$std=$' + str(round(h_32_std, 4)))
    plt.plot(sites[4], h_64_avg, 'o', label=r'$L=64,$' + ' ' +  r'$std=$' + str(round(h_64_std, 4)))
    plt.plot(sites[5], h_128_avg, 'o', label=r'$L=128,$' + ' ' +  r'$std=$' + str(round(h_128_std, 4)))
    plt.plot(sites[6], h_256_avg, 'o', label=r'$L=256,$' + ' ' +  r'$std=$' + str(round(h_256_std, 4)))
    plt.xlabel('$\it{L}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$\it{\langle{steady-state}\rangle}$', fontname='Times New Roman', fontsize=17)
    plt.legend()
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(0, 270)
    plt.ylim(0, 460)
    plt.savefig('Plots/Task2/task2e.png')
    plt.show()


if __name__ == '__main__':
    # task_1(compute=False, plot=True)
    # print('No. of reccurant configs:', check_recurrent_configs(size=2, grains=int(2e4)))
    # task_2_a(compute=False, plot=True)
    # task_2_b(compute=False, plot=True)
    # task_2_d(compute=False, plot=True)
    task_2_e()
