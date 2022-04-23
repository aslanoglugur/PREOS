import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
"""Benzene-Toluene x y data at 1 atm"""

alpha = 2.34

x = np.arange(0, 1.01, 0.01)
y = (x*alpha)/(1 + (alpha-1)*x)

plt.plot(x, y)
plt.plot(x, x)
plt.plot([0.99],[0.99], 'ro')
plt.plot([0.01],[0.01], 'ro')
plt.plot([0.5],[0.5], 'ro')
plt.plot([0],[0.33], 'ro')
plt.ylabel('y')
plt.xlabel('x')
plt.grid(which='both')

plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.2)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()