import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import os
from logbin_2020 import logbin
from scipy.stats import skew, kurtosis
import scipy.optimize as op
import argparse


def threshold_prob(p=0.5, n=1):
    """ Returns random threshold value by determining the probability for each """
    elements = [1, 2]  # The values the discrete random variable can take: {1, 2}
    probs = [p, 1-p]  # Probabilities for each value in the same order
    return np.random.choice(elements, n, p=probs)  # n denotes the size of the array


def initialise(size=4, p=0.5):
    """ Initialises the sites of a system of given size L and threshold probability p """
    heights = np.zeros(size)
    slopes = np.zeros(size)
    thresholds = threshold_prob(p, n=size)
    return heights, slopes, thresholds


def update_slopes(heights):
    """ Updates all the threshold values of the system """
    return abs(np.diff(heights, append=[0]))


def drive_and_relax(heights, slopes, thresholds, grains=16, p=0.5):
    """ 
        Keeps driving and relaxing a system achieving a final stable configuration until 
        a given number of grains have been added to the system, with threshold probability p
    """

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
                    if slopes[i] > thresholds[i]:  # Condition for relaxing a supercritical site
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
            steady_state_time = heights_sum
        
        h_1.append(heights[0])
        height_configs.append(heights.tolist())
    
    height_configs = height_configs[int(steady_state_time):]  # For Task 1 (counting configurations) delete this line
    height_configs = np.array(height_configs)
    configs_counted = Counter(map(tuple, height_configs))  # make a map counting occurence of each configuration of heights
    no_reccurant = len([k for k in configs_counted.keys() if configs_counted[k]>1])
    return heights, slopes, thresholds, h_1, steady_state_time, configs_counted, avalanches  # replace [4] with no_recurrant for Task1


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

        """ Average avalanche size with system size """
        avalanche_sizes = []
        heights, slopes, thresholds = initialise(size=4, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=4000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        heights, slopes, thresholds = initialise(size=8, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=8000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=16000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        heights, slopes, thresholds = initialise(size=32, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=32000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        heights, slopes, thresholds = initialise(size=128, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=90000, p=0.5)
        avalanche_sizes.append(np.average(avalanches))
        np.save(os.path.join('Numpy Files', '1: avg_avalanche_size'), np.array(avalanche_sizes))  # Save data in a .npy file

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
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_0, label=r'$p=0, \langle{h(t>t_c; \: L)}\rangle=$' + r'${}$'.format(str(np.average(h_1_0[300:]))))
        plt.plot(h_1_0_5, label=r'$p=0.5, \langle{h(t>t_c; \: L)}\rangle=$' + r'${}$'.format(str(round(np.average(h_1_0_5[300:]), 3))))
        plt.plot(h_1_1, label=r'$p=1, \langle{h(t>t_c; \: L)}\rangle=$' + r'${}$'.format(str(np.average(h_1_1[300:]))))
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
        plt.ylim(0, 35)
        plt.xlim(-100, 4000)
        plt.savefig('Plots/Task1/chaning_probability.png')
        plt.show()

        """ Demonstrating the bounding of the slopes by the thresholds """
        sites = np.linspace(1, 32, 32)
        slopes = np.load('Slopes Bounded: L=32, p=0.5 Slopes.npy', allow_pickle=True)
        thresholds = np.load('Slopes Bounded: L=32, p=0.5 Thresholds.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(sites, slopes, 'o', label=r'$Actual\: Slopes$')
        plt.plot(sites, thresholds, 'x', label=r'$Threshold\: Slopes$')
        plt.xlabel('$\it{i}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel('$\it{z_i}$', fontname='Times New Roman', fontsize=20)
        plt.xlim([0, 35])
        plt.ylim([-0.1, 2.5])
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

        """ Checking the height at the first site for L = 16, 32 """
        h_1_16, sst_16 = np.load('First Site Height: L=16, p=0.5.npy', allow_pickle=True)
        h_1_32, sst_32 = np.load('First Site Height: L=32, p=0.5.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_16, label=r'$L=16, \langle{h(t>t_c; \: L)}\rangle = $' + r'${}$'.format(str(round(np.average(h_1_16[sst_16:]), 3))))
        plt.plot(h_1_32, label=r'$L=32, \langle{h(t>t_c; \: L)}\rangle = $' + r'${}$'.format(str(round(np.average(h_1_32[sst_32:]), 3))))
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
        plt.ylim(0, 60)
        plt.xlim(-100, 4000)
        plt.savefig('Plots/Task1/heights_at_site_1.png')
        plt.show()

        """ Average avalanche size with system size """
        avalanche_avg = np.load('1: avg_avalanche_size.npy', allow_pickle=True)
        l = [4, 8, 16, 32, 64, 128]
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        log_l = np.log(l)
        log_avalanches_avg = np.log(avalanche_avg)
        fit_phase, cov_phase = np.polyfit(log_l, log_avalanches_avg, 1, cov=True)
        p_phase = np.poly1d(fit_phase)
        l_fit = np.linspace(min(log_l), max(log_l), 1000)
        plt.plot(log_l, log_avalanches_avg, 'o', label=r'$data$')
        plt.plot(l_fit, p_phase(l_fit), label=r'$fit: $' + r'$gradient=$' + r'${}$'.format(str(round(fit_phase[0], 3))))
        plt.xlabel('$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{log({\langle{s}\rangle})}$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.ylim(1, 5)
        plt.xlim(1, 5)
        plt.savefig('Plots/Task1/avg_avalanche_scaling.png')
        plt.show()


def check_recurrent_configs(size=4, grains=100000):
    """ Checks the total number of recurrent configurations for a given system """
    heights, slopes, thresholds = initialise(size=size, p=0.5)
    heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=grains, p=0.5)
    return reccur


def task_2_a(compute=True, plot=False):
    if compute:
        """ Obtains the pile height for a given number of grains to be added """
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
        """ Plots the pile height for a given number of grains to be added """
        h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
        h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
        h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
        h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
        h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
        h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)
        h_1_256, sst_256 = np.load('Task 2a: L=256, p=0.5.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h_1_4, label=r'$L=4$')
        plt.plot(h_1_8, label=r'$L=8$')
        plt.plot(h_1_16, label=r'$L=16$')
        plt.plot(h_1_32, label=r'$L=32$')
        plt.plot(h_1_64, label=r'$L=64$')
        plt.plot(h_1_128, label=r'$L=128$')
        plt.plot(h_1_256, label=r'$L=256$')
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
        plt.ylim(0, 500)
        plt.savefig('Plots/Task2/task2a.png')
        plt.show()


def task_2_b(compute=True, plot=False):
    l = [4, 8, 16, 32, 64, 128, 256]
    avg_cross_over_times = []
    t = 0
    
    if compute:
        """ Computes the mean cross-over times for different sized systems """
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
        """ Plots the mean cross-over times for different sized systems """
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
        plt.ylabel(r'$data$', fontname='Times New Roman', fontsize=18)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.ylim(-2000, 60000)
        plt.xlim(0, 260)
        plt.savefig('Plots/Task2/task2b.png')
        plt.show()

        """ Plots the mean cross-over times for different sized systems on a double-log plot """
        l_log = np.log(l)
        avg_cross_over_times_log = np.log(avg_cross_over_times)
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        fit_phase, cov_phase = np.polyfit(l_log, avg_cross_over_times_log, 1, cov=True)
        p_phase = np.poly1d(fit_phase)
        l_fit = np.linspace(min(l_log), max(l_log), 1000)
        plt.plot(l_log, avg_cross_over_times_log, 'o', label=r'$data$')
        plt.plot(l_fit, p_phase(l_fit), label=r'$fit: $' + r'$gradient=$' + ' ' + r'${}$'.format(str(round(fit_phase[0], 4))))
        plt.xlabel(r'$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$log({\langle{t_c}\rangle})$', fontname='Times New Roman', fontsize=18)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1, 6)
        plt.ylim(2, 12)
        plt.savefig('Plots/Task2/task2b_log_plot.png')
        plt.show()


def task_2_d(compute=True, plot=False):
    if compute:
        """ Produces a data collapse of the pile height for different sized systems """
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
        """ Plots a data collapse of the pile height for different sized systems """
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
        x_sq = 1.8355*np.sqrt(t_s)
        
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
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
        plt.plot(t_s, x_sq, '--', label=r'$y= 1.8355t^{1/2}$', alpha=0.9, linewidth=2, color='k')
        plt.xlabel('$\it{t}$' + ' ' + r'$/L^{2}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{\tilde h(t; L)}$' + ' / ' + r'$L$', fontname='Times New Roman', fontsize=17)
        plt.legend()
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-0.05, 1.2)
        plt.ylim(0, 2)
        plt.savefig('Plots/Task2/task2d.png')
        plt.show()


def task_2_e():
    """ Obtain and plot a_0, a_1 and w_1 (2nd order error-correction terms for the average steady-state height """
    h_1_2, sst_2 = np.load('Task 2a: L=2, p=0.5.npy', allow_pickle=True)
    h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
    h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
    h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
    h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
    h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
    h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)
    h_1_256, sst_256 = np.load('Task 2a: L=256, p=0.5.npy', allow_pickle=True)

    h_2_avg = np.average(h_1_2[int(sst_2):])
    h_4_avg = np.average(h_1_4[sst_4:])
    h_8_avg = np.average(h_1_8[sst_8:])
    h_16_avg = np.average(h_1_16[sst_16:])
    h_32_avg = np.average(h_1_32[sst_32:])
    h_64_avg = np.average(h_1_64[sst_64:])
    h_128_avg = np.average(h_1_128[sst_128:])
    h_256_avg = np.average(h_1_256[sst_256:])
    h_avg = [h_2_avg, h_4_avg, h_8_avg, h_16_avg, h_32_avg, h_64_avg, h_128_avg, h_256_avg]
    
    sites = [2, 4, 8, 16, 32, 64, 128, 256]

    a_0 = h_256_avg / 256
    print(a_0)

    def corr_terms(l, a_0, a_1, w_1):  # Function in script for 2nd order correction terms to steady-state height
         return a_0 - a_1*l**-w_1
    
    # Use scipy's curve-fitting function to obtain the parameters of the relation
    a_0_n, a_1_n, w_1_n = op.curve_fit(corr_terms, sites, np.array(h_avg)/np.array(sites), bounds=([0,0,-10], [10,10,10]))[0]
    print('a_0={}, w_1={}'.format(str(a_0_n), str(w_1_n)))
    
    y = [l - h/a_0 for l, h in zip(sites[:4], h_avg[:4])]
    sites_log = np.log(sites[:4])
    y_log = np.log(y)
    fig, ax = plt.subplots()
    params = {'legend.fontsize': 12}
    plt.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    fit_phase, cov_phase = np.polyfit(sites_log, y_log, 1, cov=True)
    p_phase = np.poly1d(fit_phase)
    l_fit = np.linspace(min(sites_log), max(sites_log), 500)
    plt.plot(sites_log, y_log, 'o')
    plt.plot(l_fit, p_phase(l_fit), label=r'$fit: $' + r'$\omega_1=$' + ' ' + r'${}$'.format(str(round(1 - fit_phase[0], 5))))
    plt.xlabel(r'$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$log({L-\langle{h(t; L)}\rangle_t\: /\: a_0})$', fontname='Times New Roman', fontsize=18)
    plt.legend()
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(0.5, 3)
    plt.ylim(-1.6, -0.4)
    plt.savefig('Plots/Task2/task2e_i.png')
    plt.show()
    
    a_1 = (a_0*8 - h_8_avg) / (a_0 * 8**(1-0.5065))
    print(a_1)
    actual_data = [h/l for h, l in zip(h_avg, sites)]
    curve_fit_data = [a_0_n - a_0_n*a_1_n*l**-w_1_n for l in sites]
    method_data = [a_0 - a_0*a_1*l**-0.5065 for l in sites]
    fig, ax = plt.subplots()
    params = {'legend.fontsize': 12}
    plt.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    plt.plot(sites, actual_data, 'o', label=r'$Data$')
    plt.plot(sites, curve_fit_data, label=r'$Scipy.Optimize.Curvefit \: Fit$')
    plt.plot(sites, method_data, label=r'$Devised \: Procedure \: Fit$')
    plt.xlabel(r'$\it{L}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$\it{\langle{h(t; L)}\rangle_t \: / \: L}$', fontname='Times New Roman', fontsize=18)
    plt.legend()
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(-2, 70)
    plt.ylim(1.35, 1.75)
    plt.savefig('Plots/Task2/task2e_ii.png')
    plt.show()


def task_2_f():
    """ Obtain and plot the scaling of the standard deviation of the steady-state pile height """
    h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
    h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
    h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
    h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
    h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
    h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)

    h_4_std = np.std(h_1_4[sst_4:][::4])
    h_8_std = np.std(h_1_8[sst_8:][::8])
    h_16_std = np.std(h_1_16[sst_16:][::16])
    h_32_std = np.std(h_1_32[sst_32:][::32])
    h_64_std = np.std(h_1_64[sst_64:][::64])
    h_128_std = np.std(h_1_128[sst_128:][::128])
    h_std = [h_4_std, h_8_std, h_16_std, h_32_std, h_64_std, h_128_std]
    
    sites = [4, 8, 16, 32, 64, 128]
    sites_log = np.log(sites)
    std_log = np.log(h_std)
    
    fig, ax = plt.subplots()
    params = {'legend.fontsize': 12}
    plt.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    fit_phase, cov_phase = np.polyfit(sites_log, std_log, 1, cov=True)
    p_phase = np.poly1d(fit_phase)
    l_fit = np.linspace(min(sites_log), max(sites_log), 1000)
    plt.plot(sites_log, std_log, 'o')
    plt.plot(l_fit, p_phase(l_fit), label=r'$fit: $' + r'$gradient=$' + ' ' + r'${}$'.format(str(round(fit_phase[0], 5))))
    plt.xlabel(r'$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$log({\sigma_h(L)})$', fontname='Times New Roman', fontsize=18)
    plt.legend()
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(1, 5)
    plt.ylim(-0.4, 0.8)
    plt.savefig('Plots/Task2/task2f.png')
    plt.show()


def task_2_g(compute=False, plot=True):
    if compute:
        """ Obtain the heigth probability distributions and their data collapse for different system sizes """
        probabilities = []
        
        heights, slopes, thresholds = initialise(size=4, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        h_1 = h_1[int(sst):]
        map_4 = Counter(map(int, h_1))
        probabilities_4 = []
        for i in range(0, 513):
            probabilities_4.append(map_4[i]/sum(map_4.values()) if i in map_4.keys() else 0)
        print('Done with 4')
        heights, slopes, thresholds = initialise(size=8, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        h_1 = h_1[int(sst):]
        map_8 = Counter(map(int, h_1))
        probabilities_8 = []
        for i in range(0, 513):
            probabilities_8.append(map_8[i]/sum(map_8.values()) if i in map_8.keys() else 0)
        print('Done with 8')
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        h_1 = h_1[int(sst):]
        map_16 = Counter(map(int, h_1))
        probabilities_16 = []
        for i in range(0, 513):
            probabilities_16.append(map_16[i]/sum(map_16.values()) if i in map_16.keys() else 0)
        print('Done with 16')
        heights, slopes, thresholds = initialise(size=32, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=64000, p=0.5)
        h_1 = h_1[int(sst):]
        map_32 = Counter(map(int, h_1))
        probabilities_32 = []
        for i in range(0, 513):
            probabilities_32.append(map_32[i]/sum(map_32.values()) if i in map_32.keys() else 0)
        print('Done with 32')
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=100000, p=0.5)
        h_1 = h_1[int(sst):]
        map_64 = Counter(map(int, h_1))
        probabilities_64 = []
        for i in range(0, 513):
            probabilities_64.append(map_64[i]/sum(map_64.values()) if i in map_64.keys() else 0)
        print('Done with 64')
        heights, slopes, thresholds = initialise(size=128, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=100000, p=0.5)
        h_1 = h_1[int(sst):]
        map_128 = Counter(map(int, h_1))
        probabilities_128 = []
        for i in range(0, 513):
            probabilities_128.append(map_128[i]/sum(map_128.values()) if i in map_128.keys() else 0)
        print('Done with 128')
        heights, slopes, thresholds = initialise(size=256, p=0.5)
        heights, slopes, thresholds, h_1, sst, configs_counted, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        h_1 = h_1[int(sst):]
        map_256 = Counter(map(int, h_1))
        probabilities_256 = []
        for i in range(0, 513):
            probabilities_256.append(map_256[i]/sum(map_256.values()) if i in map_256.keys() else 0)
        print('Done with 256')
        probabilities = [probabilities_4, probabilities_8, probabilities_16, probabilities_32, probabilities_64, probabilities_128, probabilities_256]
        np.save(os.path.join('Numpy Files', 'Task 2g'), np.array(probabilities))  # Save data in a .npy file

        
    if plot:
        """ Plot the heigth probability distributions for different system sizes """
        h_1_4, sst_4 = np.load('Task 2a: L=4, p=0.5.npy', allow_pickle=True)
        h_1_8, sst_8 = np.load('Task 2a: L=8, p=0.5.npy', allow_pickle=True)
        h_1_16, sst_16 = np.load('Task 2a: L=16, p=0.5.npy', allow_pickle=True)
        h_1_32, sst_32 = np.load('Task 2a: L=32, p=0.5.npy', allow_pickle=True)
        h_1_64, sst_64 = np.load('Task 2a: L=64, p=0.5.npy', allow_pickle=True)
        h_1_128, sst_128 = np.load('Task 2a: L=128, p=0.5.npy', allow_pickle=True)
        h_1_256, sst_256 = np.load('Task 2a: L=256, p=0.5.npy', allow_pickle=True)
        h_4_std = np.std(h_1_4[sst_4:][::4])
        h_8_std = np.std(h_1_8[sst_8:][::8])
        h_16_std = np.std(h_1_16[sst_16:][::16])
        h_32_std = np.std(h_1_32[sst_32:][::32])
        h_64_std = np.std(h_1_64[sst_64:][::64])
        h_128_std = np.std(h_1_128[sst_128:][::128])
        h_256_std = np.std(h_1_256[sst_256:][::16])

        h = np.linspace(0, 512, 513)
        probabilities = np.load('Task 2g.npy', allow_pickle=True)
        probabilities_4 = probabilities[0]
        probabilities_8 = probabilities[1]
        probabilities_16 = probabilities[2]
        probabilities_32 = probabilities[3]
        probabilities_64 = probabilities[4]
        probabilities_128 = probabilities[5]
        probabilities_256 = probabilities[6]
        avg_4 = sum([p*x for p, x in zip(probabilities_4, h)])
        avg_8 = sum([p*x for p, x in zip(probabilities_8, h)])
        avg_16 = sum([p*x for p, x in zip(probabilities_16, h)])
        avg_32 = sum([p*x for p, x in zip(probabilities_32, h)])
        avg_64 = sum([p*x for p, x in zip(probabilities_64, h)])
        avg_128 = sum([p*x for p, x in zip(probabilities_128, h)])
        avg_256 = sum([p*x for p, x in zip(probabilities_256, h)])

        """ Obtain the data collapse of the heigth probability distributions for different system sizes """
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params) 
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(h, probabilities_4, label=r'$L=4, \: \langle{h_{4}}\rangle = $' + r'${}$'.format(str(round(avg_4, 2))))
        plt.plot(h, probabilities_8, label=r'$L=8, \: \langle{h_{8}}\rangle = $' + r'${}$'.format(str(round(avg_8, 2))))
        plt.plot(h, probabilities_16, label=r'$L=16, \: \langle{h_{16}}\rangle = $' + r'${}$'.format(str(round(avg_16, 2))))
        plt.plot(h, probabilities_32, label=r'$L=32, \: \langle{h_{32}}\rangle = $' + r'${}$'.format(str(round(avg_32, 2))))
        plt.plot(h, probabilities_64, label=r'$L=64, \: \langle{h_{64}}\rangle = $' + r'${}$'.format(str(round(avg_64, 2))))
        plt.plot(h, probabilities_128, label=r'$L=128, \: \langle{h_{128}}\rangle = $' + r'${}$'.format(str(round(avg_128, 2))))
        plt.plot(h, probabilities_256, label=r'$L=256, \: \langle{h_{256}}\rangle = $' + r'${}$'.format(str(round(avg_256, 2))))
        plt.legend()
        plt.xlabel('$\it{h}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{P(h; L)}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(0, 500)
        plt.ylim(0, 0.5)
        plt.savefig('Plots/Task2/task2g_b_i.png')
        plt.show()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        p_4_non_zero = [x for x in probabilities_4*h_4_std if x != 0]
        p_8_non_zero = [x for x in probabilities_8*h_8_std if x != 0]
        p_16_non_zero = [x for x in probabilities_16*h_16_std if x != 0]
        p_32_non_zero = [x for x in probabilities_32*h_32_std if x != 0]
        p_64_non_zero = [x for x in probabilities_64*h_64_std if x != 0]
        p_128_non_zero = [x for x in probabilities_128*h_128_std if x != 0]
        p_256_non_zero = [x for x in probabilities_256*h_256_std if x != 0]

        ind_4_start = next((i for i, x in enumerate(probabilities_4*h_4_std) if x!=0), None)
        ind_4_end = ind_4_start + len(p_4_non_zero)
        x_4_non_zero = ((h-avg_4)/h_4_std)[ind_4_start:ind_4_end]
        ind_8_start = next((i for i, x in enumerate(probabilities_8*h_8_std) if x!=0), None)
        ind_8_end = ind_8_start + len(p_8_non_zero)
        x_8_non_zero = ((h-avg_8)/h_8_std)[ind_8_start:ind_8_end]
        ind_16_start = next((i for i, x in enumerate(probabilities_16*h_16_std) if x!=0), None)
        ind_16_end = ind_16_start + len(p_16_non_zero)
        x_16_non_zero = ((h-avg_16)/h_16_std)[ind_16_start:ind_16_end]
        ind_32_start = next((i for i, x in enumerate(probabilities_32*h_32_std) if x!=0), None)
        ind_32_end = ind_32_start + len(p_32_non_zero)
        x_32_non_zero = ((h-avg_32)/h_32_std)[ind_32_start:ind_32_end]
        ind_64_start = next((i for i, x in enumerate(probabilities_64*h_64_std) if x!=0), None)
        ind_64_end = ind_64_start + len(p_64_non_zero)
        x_64_non_zero = ((h-avg_64)/h_64_std)[ind_64_start:ind_64_end]
        ind_128_start = next((i for i, x in enumerate(probabilities_128*h_128_std) if x!=0), None)
        ind_128_end = ind_128_start + len(p_128_non_zero)
        x_128_non_zero = ((h-avg_128)/h_128_std)[ind_128_start:ind_128_end]
        ind_256_start = next((i for i, x in enumerate(probabilities_256*h_256_std) if x!=0), None)
        ind_256_end = ind_256_start + len(p_256_non_zero)
        x_256_non_zero = ((h-avg_256)/h_256_std)[ind_256_start:ind_256_end]

        x = np.linspace(-4, 8, 513)
        f = lambda x: 1/(np.sqrt(2*np.pi)) * np.exp(-0.5*x**2)  # Normal distribution
        y = f(x)

        dif_sq_4 = sum([(p-f(x))**2/f(x) for p, x in zip(p_4_non_zero, x_4_non_zero)])
        dif_sq_8 = sum([(p-f(x))**2/f(x) for p, x in zip(p_8_non_zero, x_8_non_zero)])
        dif_sq_16 = sum([(p-f(x))**2/f(x) for p, x in zip(p_16_non_zero, x_16_non_zero)])
        dif_sq_32 = sum([(p-f(x))**2/f(x) for p, x in zip(p_32_non_zero, x_32_non_zero)])
        dif_sq_64 = sum([(p-f(x))**2/f(x) for p, x in zip(p_64_non_zero, x_64_non_zero)])
        dif_sq_128 = sum([(p-f(x))**2/f(x) for p, x in zip(p_128_non_zero, x_128_non_zero)])
        dif_sq_256 = sum([(p-f(x))**2/f(x) for p, x in zip(p_256_non_zero, x_256_non_zero)])
        print(r'$\chi^2 values:$', dif_sq_4, dif_sq_8, dif_sq_16, dif_sq_32, dif_sq_64, dif_sq_128, dif_sq_256)

        skew_4, kur_4 = skew(h_1_4[sst_4:]), kurtosis(h_1_4[sst_4:])
        skew_8, kur_8 = skew(h_1_8[sst_8:]), kurtosis(h_1_8[sst_8:])
        skew_16, kur_16 = skew(h_1_16[sst_16:]), kurtosis(h_1_16[sst_16:])
        skew_32, kur_32 = skew(h_1_32[sst_32:]), kurtosis(h_1_32[sst_32:])
        skew_64, kur_64 = skew(h_1_64[sst_64:]), kurtosis(h_1_64[sst_64:])
        skew_128, kur_128 = skew(h_1_128[sst_128:]), kurtosis(h_1_128[sst_128:])
        skew_256, kur_256 = skew(h_1_256[sst_256:]), kurtosis(h_1_256[sst_256:])
        
        plt.plot((h-avg_4)/h_4_std, probabilities_4*h_4_std, 'x', label=r'$L=4:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_4, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_4, 2))))
        plt.plot((h-avg_8)/h_8_std, probabilities_8*h_8_std, 'x', label=r'$L=8:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_8, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_8, 2))))
        plt.plot((h-avg_16)/h_16_std, probabilities_16*h_16_std, 'x', label=r'$L=16:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_16, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_16, 2))))
        plt.plot((h-avg_32)/h_32_std, probabilities_32*h_32_std, 'x', label=r'$L=32:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_32, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_32, 2))))
        plt.plot((h-avg_64)/h_64_std, probabilities_64*h_64_std, 'x', label=r'$L=64:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_64, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_64, 2))))
        plt.plot((h-avg_128)/h_128_std, probabilities_128*h_128_std, 'x', label=r'$L=128:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_128, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_128, 2))))
        plt.plot((h-avg_256)/h_256_std, probabilities_256*h_256_std, 'x', label=r'$L=256:$' + ' ' + r'$s=$' + r'${}$'.format(str(round(skew_256, 2))) + ', ' + r'$k=$' + r'${}$'.format(str(round(kur_256, 2))))
        plt.plot(x, y, '--', label=r'$Normal: \: \mu=0, \: \sigma=1$' +'\n\t    ' + r'$s=0, \: k=0$', linewidth=2, alpha=0.7, color='k')
        plt.legend()
        plt.xlabel(r'$\it{(h - \langle{h}\rangle) \: / \: σ_h}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{σ_h \: P(h; L)}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-4, 8)
        plt.ylim(0, 0.45)
        plt.savefig('Plots/Task2/task2g_b_ii.png')
        plt.show()

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot((h-avg_4)/h_4_std, np.log(probabilities_4*h_4_std), 'x', label=r'$L=4$')
        plt.plot((h-avg_8)/h_8_std, np.log(probabilities_8*h_8_std), 'x', label=r'$L=8$')
        plt.plot((h-avg_16)/h_16_std, np.log(probabilities_16*h_16_std), 'x', label=r'$L=16$')
        plt.plot((h-avg_32)/h_32_std, np.log(probabilities_32*h_32_std), 'x', label=r'$L=32$')
        plt.plot((h-avg_64)/h_64_std, np.log(probabilities_64*h_64_std), 'x', label=r'$L=64$')
        plt.plot((h-avg_128)/h_128_std, np.log(probabilities_128*h_128_std), 'x', label=r'$L=128$')
        plt.plot((h-avg_256)/h_256_std, np.log(probabilities_256*h_256_std), 'x', label=r'$L=256$')
        x = np.linspace(-5, 8, 513)
        y = f(x)
        plt.plot(x, np.log(y), '--', label=r'$Normal$')
        plt.legend()
        plt.xlabel(r'$\it{(h - \langle{h}\rangle) \: / \: σ_h}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{log(σ_h \: P(h; L))}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-5, 8)
        plt.ylim(-12, 0)
        plt.savefig('Plots/Task2/task2g_b_iii.png')
        plt.show()


def task_3_a(compute=True, plot=False):
    if compute:
        """ Obtain the avalanche size probabilities for different system sizes """
        heights, slopes, thresholds = initialise(size=4, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_4 = avalanches[int(sst):]
        print('Done with 4')
        heights, slopes, thresholds = initialise(size=8, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_8 = avalanches[int(sst):]
        print('Done with 8')
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_16 = avalanches[int(sst):]
        print('Done with 16')
        heights, slopes, thresholds = initialise(size=32, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_32 = avalanches[int(sst):]
        print('Done with 32')
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_64 = avalanches[int(sst):]
        print('Done with 64')
        heights, slopes, thresholds = initialise(size=128, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_128 = avalanches[int(sst):]
        print('Done with 128')
        heights, slopes, thresholds = initialise(size=256, p=0.5)
        heights, slopes, thresholds, h_1, sst, reccur, avalanches = drive_and_relax(heights, slopes, thresholds, grains=128000, p=0.5)
        avalanches_256 = avalanches[int(sst):]
        print('Done with 256')
        
        avalanches_total = [avalanches_4, avalanches_8, avalanches_16, avalanches_32, avalanches_64, avalanches_128, avalanches_256]
        np.save(os.path.join('Numpy Files', 'Task 3a'), np.array(avalanches_total))  # Save data in a .npy file
        
    if plot:
        """ Plot the avalanche size probabilities and their data collapse """
        avalanches = np.load('Task 3a.npy', allow_pickle=True)
        x_4, y_4 = logbin(avalanches[0], scale=1.5)
        x_8, y_8 = logbin(avalanches[1], scale=1.5)
        x_16, y_16 = logbin(avalanches[2], scale=1.5)
        x_32, y_32 = logbin(avalanches[3], scale=1.5)
        x_64, y_64 = logbin(avalanches[4], scale=1.5)
        x_128, y_128 = logbin(avalanches[5], scale=1.5)
        x_256, y_256 = logbin(avalanches[6], scale=1.5)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(x_4, y_4, label=r'$L=4 \:$')
        plt.plot(x_8, y_8, label=r'$L=8 \:$')
        plt.plot(x_16, y_16, label=r'$L=16 \:$')
        plt.plot(x_32, y_32, label=r'$L=32 \:$')
        plt.plot(x_64, y_64, label=r'$L=64 \:$')
        plt.plot(x_128, y_128, label=r'$L=128 \:$')
        plt.plot(x_256, y_256, label=r'$L=256 \:$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend()
        plt.xlabel(r'$\it{s}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{\tilde P_N(s)}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1, 1e6)
        plt.ylim(10e-10, 10)
        plt.savefig('Plots/Task3/task3a_a_i.png')
        plt.show()

        l = [4, 8, 16, 32, 64, 128, 256]
        s_max_4 = max(avalanches[0])
        s_max_8 = max(avalanches[1])
        s_max_16 = max(avalanches[2])
        s_max_32 = max(avalanches[3])
        s_max_64 = max(avalanches[4])
        s_max_128 = max(avalanches[5])
        s_max_256 = max(avalanches[6])

        s_max = [s_max_4, s_max_8, s_max_16, s_max_32, s_max_64, s_max_128, s_max_256]
               
        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        s_max_log = np.log(s_max)
        l_log = np.log(l)
        fit_phase_1, cov_phase = np.polyfit(l_log, s_max_log, 1, cov=True)
        p_phase = np.poly1d(fit_phase_1)
        l_space_log = np.linspace(min(l_log), max(l_log), 500)
        plt.plot(l_log, s_max_log, 'o', label=r'$data$')
        plt.plot(l_space_log, p_phase(l_space_log), label=r'$fit: \: gradient={}$'.format(str(round(fit_phase_1[0], 3))))
        plt.legend()
        plt.xlabel(r'$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{log({s_{c}})}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(1, 6)
        plt.ylim(2, 14)
        plt.savefig('Plots/Task3/task3a_b_i.png')
        plt.show()

        x_256_1, y_256_1 = logbin(avalanches[6], scale=1)
        s_256_log = np.log(x_256_1[:170])
        y_256_log = np.log(y_256_1[:170])

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        fit_phase_2, cov_phase = np.polyfit(s_256_log, y_256_log, 1, cov=True)
        p_phase = np.poly1d(fit_phase_2)
        s_space_log = np.linspace(min(s_256_log), max(s_256_log), 500)
        plt.plot(s_256_log, y_256_log, 'o', label=r'$data$')
        plt.plot(s_space_log, p_phase(s_space_log), label=r'$fit: \: gradient\equiv-\tau={}$'.format(str(round(fit_phase_2[0], 3))))
        plt.legend()
        plt.xlabel(r'$\it{log({s})}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{log({P(s; L)})}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(-1, 6)
        plt.ylim(-10, -1)
        plt.savefig('Plots/Task3/task3a_b_ii.png')
        plt.show()

        tau = -fit_phase_2[0]
        d = fit_phase_1[0]

        x_4_log = x_4 / (4**d)
        x_8_log = x_8 / (8**d)
        x_16_log = x_16 / (16**d)
        x_32_log = x_32 / (32**d)
        x_64_log = x_64 / (64**d)
        x_128_log = x_128 / (128**d)
        x_256_log = x_256 / (256**d)
        y_4_log = [(x**tau) * y for x, y in zip(x_4, y_4)]
        y_8_log = [(x**tau) * y for x, y in zip(x_8, y_8)]
        y_16_log = [(x**tau) * y for x, y in zip(x_16, y_16)]
        y_32_log = [(x**tau) * y for x, y in zip(x_32, y_32)]
        y_64_log = [(x**tau) * y for x, y in zip(x_64, y_64)]
        y_128_log = [(x**tau) * y for x, y in zip(x_128, y_128)]
        y_256_log = [(x**tau) * y for x, y in zip(x_256, y_256)]

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 12}
        plt.rcParams.update(params)
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.plot(x_4_log, y_4_log, label=r'$L=4 \:$')
        plt.plot(x_8_log, y_8_log, label=r'$L=8 \:$')
        plt.plot(x_16_log, y_16_log, label=r'$L=16 \:$')
        plt.plot(x_32_log, y_32_log, label=r'$L=32 \:$')
        plt.plot(x_64_log, y_64_log, label=r'$L=64 \:$')
        plt.plot(x_128_log, y_128_log, label=r'$L=128 \:$')
        plt.plot(x_256_log, y_256_log, label=r'$L=256 \:$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend()
        plt.xlabel(r'$\it{s \: / \: L^{D}}$', fontname='Times New Roman', fontsize=17)
        plt.ylabel(r'$\it{ s^{\tau_s} \tilde P_N(s)}$', fontname='Times New Roman', fontsize=17)
        plt.minorticks_on()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(10e-5, 10)
        plt.ylim(10e-4, 1)
        plt.savefig('Plots/Task3/task3a_b_iii.png')
        plt.show()


def task_3_b():
    """ Perform k-th moment analysis to obtain D and \tau """
    
    avalanches = np.load('Task 3a.npy', allow_pickle=True)
    
    k_mom_4 = [np.average(np.power(avalanches[0], i)) for i in range(1, 5)]
    k_mom_8 = [np.average(np.power(avalanches[1], i)) for i in range(1, 5)]
    k_mom_16 = [np.average(np.power(avalanches[2], i)) for i in range(1, 5)]
    k_mom_32 = [np.average(np.power(avalanches[3], i)) for i in range(1, 5)]
    k_mom_64 = [np.average(np.power(avalanches[4], i)) for i in range(1, 5)]
    k_mom_128 = [np.average(np.power(avalanches[5], i)) for i in range(1, 5)]
    k_mom_256 = [np.average(np.power(avalanches[6], i)) for i in range(1, 5)]

    k_1 = [k_mom_4[0], k_mom_8[0], k_mom_16[0], k_mom_32[0], k_mom_64[0], k_mom_128[0], k_mom_256[0]]
    k_2 = [k_mom_4[1], k_mom_8[1], k_mom_16[1], k_mom_32[1], k_mom_64[1], k_mom_128[1], k_mom_256[1]]
    k_3 = [k_mom_4[2], k_mom_8[2], k_mom_16[2], k_mom_32[2], k_mom_64[2], k_mom_128[2], k_mom_256[2]]
    k_4 = [k_mom_4[3], k_mom_8[3], k_mom_16[3], k_mom_32[3], k_mom_64[3], k_mom_128[3], k_mom_256[3]]
    
    y_1 = np.log(k_1)
    y_2 = np.log(k_2)
    y_3 = np.log(k_3)
    y_4 = np.log(k_4)

    l = [4, 8, 16, 32, 64, 128, 256]
    l_log = np.log(l)

    fit_phase_1, cov_phase_1 = np.polyfit(l_log, y_1, 1, cov=True)
    p_phase_1 = np.poly1d(fit_phase_1)
    fit_phase_2, cov_phase_2 = np.polyfit(l_log, y_2, 1, cov=True)
    p_phase_2 = np.poly1d(fit_phase_2)
    fit_phase_3, cov_phase_3 = np.polyfit(l_log, y_3, 1, cov=True)
    p_phase_3 = np.poly1d(fit_phase_3)
    fit_phase_4, cov_phase_4 = np.polyfit(l_log[:-1], y_4[:-1], 1, cov=True)
    p_phase_4 = np.poly1d(fit_phase_4)

    gradient_1 = fit_phase_1[0]
    gradient_2 = fit_phase_2[0]
    gradient_3 = fit_phase_3[0]
    gradient_4 = fit_phase_4[0]

    fig, ax = plt.subplots()
    params = {'legend.fontsize': 12}
    plt.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    l_log_space = np.linspace(min(l_log), max(l_log), 500)
    plt.plot(l_log, y_1, 'o')
    plt.plot(l_log, y_2, 'o')
    plt.plot(l_log, y_3, 'o')
    plt.plot(l_log[:-1], y_4[:-1], 'o')
    plt.plot(l_log_space, p_phase_1(l_log_space), label=r'$fit: k=1,$' + ' ' + r'$gradient=$' + r'${}$'.format(round(gradient_1, 3)))
    plt.plot(l_log_space, p_phase_2(l_log_space), label=r'$fit: k=2,$' + ' ' + r'$gradient=$' + r'${}$'.format(round(gradient_2, 3)))
    plt.plot(l_log_space, p_phase_3(l_log_space), label=r'$fit: k=3,$' + ' ' + r'$gradient=$' + r'${}$'.format(round(gradient_3, 3)))
    plt.plot(l_log_space[:-80], p_phase_4(l_log_space[:-80]), label=r'$fit: k=4,$' + ' ' + r'$gradient=$' + r'${}$'.format(round(gradient_4, 3)))
    plt.legend()
    plt.xlabel(r'$\it{log({L})}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$\it{log({\langle{s^k}\rangle})}$', fontname='Times New Roman', fontsize=17)
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(1, 6)
    plt.ylim(0, 35)
    plt.savefig('Plots/Task3/task3b_i.png')
    plt.show()

    gradients = [gradient_1, gradient_2, gradient_3, gradient_4]
    k_s = [1, 2, 3, 4]
    fit_phase_k, cov_phase_k = np.polyfit(k_s, gradients, 1, cov=True)
    p_phase_k = np.poly1d(fit_phase_k)
    
    fig, ax = plt.subplots()
    params = {'legend.fontsize': 12}
    plt.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    plt.plot(k_s, gradients, 'o')
    k_space = np.linspace(0, 4, 500)
    plt.plot(k_space, p_phase_k(k_space), label=r'$fit:$' + r'$gradient\equiv D = $' + r'${}$'.format(round(fit_phase_k[0], 3)) \
                                                + '\n' + r'$fit: \tau_s = $' + r'${}$'.format(round(1 - fit_phase_k[1]/fit_phase_k[0], 4)))
    plt.legend()
    plt.xlabel(r'$\it{k}$', fontname='Times New Roman', fontsize=17)
    plt.ylabel(r'$\it{D(1 + k - \tau_s)}$', fontname='Times New Roman', fontsize=17)
    plt.minorticks_on()
    ax.tick_params(direction='in')
    ax.tick_params(which='minor', direction='in')
    plt.grid(b=True, which='major', color='#8e8e8e', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xlim(0.5, 4.5)
    plt.ylim(0, 8)
    plt.savefig('Plots/Task3/task3b_ii.png')
    plt.show()


def use_args(args):
    """
    :param args: arguments provided by user at terminal / cmd
    """

    tasks = ['1', '2a', '2b', '2d', '2e', '2f', '2g', '3a', '3b']  # Valid task numbers
    if args.task not in tasks:  # If task No. provided is not valid raise value error
        raise ValueError('Argument not valid. Please enter one of the following tasks\n\t{}'.format(tasks))
    else:
        if args.task == '1':
            if args.execute:
                task_1(compute=True, plot=False)
                # print('No. of reccurant configs:', check_recurrent_configs(size=2, grains=int(2e4)))  # remember to change vars
            else:
                task_1(compute=False, plot=True)
                # print('No. of reccurant configs:', check_recurrent_configs(size=2, grains=int(2e4)))  # remember to change vars
        elif args.task == '2a':
            if args.execute:            
                task_2_a(compute=True, plot=False)
            else:
                task_2_a(compute=False, plot=True)
        elif args.task == '2b':
            if args.execute:
                task_2_b(compute=True, plot=False)
            else:
                task_2_b(compute=False, plot=True)
        elif args.task == '2d':
            if args.execute:
                task_2_d(compute=True, plot=False)
            else:
                task_2_d(compute=False, plot=True)
        elif args.task == '2e':
            task_2_e()
        elif args.task == '2f':
            task_2_f()
        elif args.task == '2g':
            if args.execute:
                task_2_g(compute=True, plot=False)
            else:
                task_2_g(compute=False, plot=True)
        elif args.task == '3a':
            if args.execute:
                task_3_a(compute=True, plot=False)
            else:
                task_3_a(compute=False, plot=True)
        elif args.task == '3b':
            task_3_b()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Complexity & Networks: Complexity Project - Script Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-t', '--task', type=str, help='Task number to be executed')
    parser.add_argument('-e', '--execute', action='store_true', help='Flag: if present will execute rather than plot task')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
