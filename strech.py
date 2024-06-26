import matplotlib.pyplot as plt
import numpy as np


# Create some data
data = np.random.rand(100, 100)

# Create a LogStretch instance
stretch = LogStretch()

# Create an ImageNormalize instance
norm = ImageNormalize(vmin=0., vmax=1., stretch=stretch)

# Display the image
plt.imshow(data, norm=norm)
plt.show()