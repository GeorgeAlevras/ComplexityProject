import numpy as np


class OsloModel:
    """
        Object to represent the Oslo Model with a d=1 lattice of l sites.
    """

    def __init__(self, l):
        self.size = l
        self.sites = np.zeros((l, 2))  # Array holds height of grains and threshold slope at each site
        self.sites[:,1] = np.random.randint(1, 3, size=self.size)  # Initialise random thresholds {1, 2}

    def __repr__(self):
        return '\nOslo Model: \n\nSite size = ' + str(np.shape(self.sites)[0]) + '\nSite Heights: \n\t' + str(self.sites[:,0]) + \
            '\nSite Threshold Slopes: \n\t' + str(self.sites[:,1])

    def get_slope(self, i):
        i -= 1
        if i >= self.size or i < 0:
            raise ValueError ("Invalid location value, must be 1 <= i <= l")
        elif i == self.size - 1:  # When at edge of lattice (l)
            return self.sites[i][0]
        else:
            return self.sites[i][0] - self.sites[i+1][0]
        

def model_algorithm(model):
    pass


if __name__ == '__main__':
    model = OsloModel(l=20)
    print(model)
