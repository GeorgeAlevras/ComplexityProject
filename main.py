import numpy as np
import matplotlib.pyplot as plt
import os


def initialise(size=4):
    heights = np.zeros(size)
    slopes = np.zeros(size)
    thresholds = np.random.randint(1, 3, size=size)  # {1, 2} with equal probability of 0.5
    return heights, slopes, thresholds


def update_slopes(heights):
    return abs(np.diff(heights, append=[0]))


def drive_and_relax(heights, slopes, thresholds, grains=16):
    avalanches = []
    s_h = []

    for g in range(grains):
        avalanche_size = 0
        heights[0] += 1
        slopes = update_slopes(heights)
        if ((slopes - thresholds) <= 0).all():
            avalanches.append(avalanche_size)
        else:
            while not ((slopes - thresholds) <= 0).all():
                for i in range(len(heights)):
                    if slopes[i] > thresholds[i]:
                        heights[i] -= 1
                        if i != len(heights) - 1:
                            heights[i+1] += 1
                        slopes = update_slopes(heights)
                        thresholds[i] = np.random.randint(1, 3)
                        avalanche_size += 1
                    else:
                        pass
            avalanches.append(avalanche_size)

        s_h.append(heights[0])
    return heights, slopes, thresholds, s_h, avalanches


def task_1_plots(heights, slopes, thresholds, s_h, avalanches):
    """ Load all the data saved in .npy files """
    p_0 = np.load('Probability of 0.npy', allow_pickle=True)
    p_0_5 = np.load('Probability of 0.5.npy', allow_pickle=True)
    p_1 = np.load('Probability of 1.npy', allow_pickle=True)
    p_0_5_16 = np.load('Probability of 0.5 16.npy', allow_pickle=True)
    p_0_5_32 = np.load('Probability of 0.5 32.npy', allow_pickle=True)
    p_0_5_64 = np.load('Probability of 0.5 64.npy', allow_pickle=True)
    p_0_5_128 = np.load('Probability of 0.5 128.npy', allow_pickle=True)

    """ Changing the probability for the same size of the lattice"""
    # plt.plot(p_0, label='p=0, Average of steady-state:' + str(np.average(p_0[300:])))
    # plt.plot(p_0_5, label='p=0.5, Average of steady-state:' + str(round(np.average(p_0_5[300:]), 5)))
    # plt.plot(p_1, label='p=1, Average of steady-state:' + str(np.average(p_1[300:])))
    # plt.xlabel('Time t - measured as the No. of additions of grains', fontname='Times New Roman', fontsize=12)
    # plt.ylabel('Height at site i=1', fontname='Times New Roman', fontsize=12)
    # plt.legend()
    # plt.savefig('Plots/Task1/test_1_all_ps.png')
    # plt.show()

    """ Changing the size of the lattice for the same probability"""
    # plt.plot(p_0_5_16, label='L=16, p=0.5, Average of steady-state:' + str(round(np.average(p_0_5_16[300:]), 5)))
    # plt.plot(p_0_5_32, label='L=32, p=0.5, Average of steady-state:' + str(round(np.average(p_0_5_32[1000:]), 5)))
    # plt.plot(p_0_5_64, label='L=64, p=0.5, Average of steady-state:' + str(round(np.average(p_0_5_64[4200:]), 5)))    
    # plt.plot(p_0_5_128, label='L=128, p=0.5, Average of steady-state:' + str(round(np.average(p_0_5_128[15000:]), 5)))
    # plt.xlabel('Time t - measured as the No. of additions of grains', fontname='Times New Roman', fontsize=12)
    # plt.ylabel('Height at site i=1', fontname='Times New Roman', fontsize=12)
    # plt.legend()
    # plt.savefig('Plots/Task1/test_1_16v32v64v128.png')
    # plt.show()

    """ Demonstrating the bounding of the slopes by the thresholds """
    # plt.plot(slopes, 'o', label='Actual Slopes')
    # plt.plot(thresholds, 'x', label='Threshold Slopes')
    # plt.xlabel('Site Locations', fontname='Times New Roman', fontsize=12)
    # plt.ylabel('Slope', fontname='Times New Roman', fontsize=12)
    # plt.ylim([-0.25, 2.25])
    # plt.legend()
    # plt.savefig('Plots/Task1/test_1_slopes_bounding.png')
    # plt.show()

    """ Avalanche sizes for L=128 """
    plt.plot(avalanches, label='Size - No. of Relaxations needed for 1 addition')
    plt.xlabel('Time t - measured as the No. of additions of grains', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Avalanche size', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('Plots/Task1/test_1_avalanches.png')
    plt.show()


if __name__ == '__main__':
    heights, slopes, thresholds = initialise(size=128)
    heights, slopes, thresholds, s_h, avalanches = drive_and_relax(heights, slopes, thresholds, grains=30000)
    # np.save(os.path.join('Numpy Files', 'Probability of 0.5 128'), np.array(s_h))  # Save data in a .npy file
    task_1_plots(heights, slopes, thresholds, s_h, avalanches)
