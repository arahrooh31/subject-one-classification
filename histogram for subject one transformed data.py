# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:03:18 2019

@author: Al Rahrooh
"""

#Histogram for transformed S1_SP
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = X1_transformed
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()


#Histogram for transformed S1_50
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = X2_transformed
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()


#Histogram for transformed S1_75
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = X3_transformed
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()


#Histogram for transformed S1_100
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = X4_transformed
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()


#Histogram for transformed S1_125
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = X5_transformed
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()



