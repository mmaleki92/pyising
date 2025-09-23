import os
import numpy as np
import matplotlib.pyplot as plt


temps = os.listdir("examples/simultion/L_64")

n = 0
for temp in temps:
    for i in range(100):
        config = np.load(f"examples/simultion/L_64/{temp}/all_configs.npy")

    

        plt.imshow(config[-i])
        plt.savefig(f"{temp}_{n}.png")
        plt.cla()
        n += 1