import numpy as np
import matplotlib.pyplot as plt
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

    for g in range(grains):
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
        if steady_state:
            h_1.append(heights[0])
    
    return heights, slopes, thresholds, h_1, avalanches


def task_1(compute=True, plot=False):
    if compute:
        """ Plotting the heights themselves """
        heights, slopes, thresholds = initialise(size=64, p=0.5)
        heights, slopes, thresholds, h_1, avalanches = drive_and_relax(heights, slopes, thresholds, grains=10000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Pile Visualised: L=64, p=0.5'), np.array(heights))  # Save data in a .npy file
    
        """ Changing the probability for the same size of the lattice"""
        heights, slopes, thresholds = initialise(size=16, p=0.5)
        heights_0_5, slopes_0_5, thresholds_0_5, h_1_0_5, avalanches_0_5 = drive_and_relax(heights, slopes, thresholds, grains=2000, p=0.5)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=0.5'), np.array(h_1_0_5))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=16, p=0)
        heights_0, slopes_0, thresholds_0, h_1_0, avalanches_0 = drive_and_relax(heights, slopes, thresholds, grains=2000, p=0)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=0'), np.array(h_1_0))  # Save data in a .npy file
        heights, slopes, thresholds = initialise(size=16, p=1)
        heights_1, slopes_1, thresholds_1, h_1_1, avalanches_1 = drive_and_relax(heights, slopes, thresholds, grains=2000, p=1)
        np.save(os.path.join('Numpy Files', 'Changing Probabilities: L=16, p=1'), np.array(h_1_1))  # Save data in a .npy file
    
        """ Demonstrating the bounding of the slopes by the thresholds """
        heights, slopes, thresholds = initialise(size=32)
        heights, slopes, thresholds, h_1, avalanches = drive_and_relax(heights, slopes, thresholds, grains=1000)
        np.save(os.path.join('Numpy Files', 'Slopes Bounded: L=32, p=0.5 Slopes'), np.array(slopes))  # Save data in a .npy file
        np.save(os.path.join('Numpy Files', 'Slopes Bounded: L=32, p=0.5 Thresholds'), np.array(thresholds))  # Save data in a .npy file
    
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
        plt.plot(h_1_0, label='p=0, Average of steady-state:' + str(np.average(h_1_0)))
        plt.plot(h_1_0_5, label='p=0.5, Average of steady-state:' + str(round(np.average(h_1_0_5), 5)))
        plt.plot(h_1_1, label='p=1, Average of steady-state:' + str(np.average(h_1_1)))
        plt.xlabel('Time t - measured as the No. of additions of grains', fontname='Times New Roman', fontsize=12)
        plt.ylabel('Height at site i=1', fontname='Times New Roman', fontsize=12)
        plt.legend()
        plt.savefig('Plots/Task1/chaning_probability.png')
        plt.show()

        """ Demonstrating the bounding of the slopes by the thresholds """
        slopes = np.load('Slopes Bounded: L=32, p=0.5 Slopes.npy', allow_pickle=True)
        thresholds = np.load('Slopes Bounded: L=32, p=0.5 Thresholds.npy', allow_pickle=True)
        plt.plot(slopes, 'o', label='Actual Slopes')
        plt.plot(thresholds, 'x', label='Threshold Slopes')
        plt.xlabel('Site Location', fontname='Times New Roman', fontsize=12)
        plt.ylabel('Slope', fontname='Times New Roman', fontsize=12)
        plt.ylim([-0.25, 2.25])
        plt.legend()
        plt.savefig('Plots/Task1/slopes_bounded.png')
        plt.show()


if __name__ == '__main__':
    task_1(compute=False, plot=True)
