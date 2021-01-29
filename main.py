import numpy as np


def initialise(size=4):
    heights = np.zeros(size)
    slopes = np.zeros(size)
    thresholds = np.random.randint(1, 3, size=size)
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
            print('Heights: ', heights, 'Slopes: ', slopes, 'Thresholds: ', thresholds)
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
            print('Heights: ', heights, 'Slopes: ', slopes, 'Thresholds: ', thresholds)

        if g > 800:
            s_h.append(heights[0])
    return s_h, avalanches


if __name__ == '__main__':
    heights, slopes, thresholds = initialise(size=32)
    s_h, avalanches = drive_and_relax(heights, slopes, thresholds, grains=20000)
    print(np.average(s_h))
