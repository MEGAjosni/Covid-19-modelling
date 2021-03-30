import numpy as np
import matplotlib.pyplot as plt

list_of_lists = [[1,2,3,4,1,2,3,4,1,2,3,4],[2,3,5,9,2,3,5,9,2,3,5,9],[5,9,8,1,5,9,8,1,5,9,8,1],[1,2,3,4,1,2,3,4,1,2,3,4],[2,3,5,9,2,3,5,9,2,3,5,9],[5,9,8,1,5,9,8,1,5,9,8,1]]

data = np.array(list_of_lists)
length = data.shape[0]
width = data.shape[1]
x, y = np.meshgrid(np.arange(length), np.arange(width))

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(x, y, data.T)
plt.show()