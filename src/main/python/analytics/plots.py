import numpy as np
import matplotlib.pyplot as plt


class Plots(object):

    def timeserie(self):
        plt.plot(np.arange(0, 8760), current.reshape((8760,)))
        plt.show()
        plt.savefig('series')