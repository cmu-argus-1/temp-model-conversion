import numpy as np
import matplotlib.pyplot as plt

# Read csv file confidences.csv and plot 
confidences = np.loadtxt("confidences.csv", delimiter=",")
plt.plot(confidences)
plt.show()