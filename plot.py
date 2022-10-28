import json
import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3)

with open('generation_results.json', 'r') as file:
    generation_results = json.load(file)
file.close()

number_of_generations = len(generation_results['best_fitnesses'])
x = [i for i in range(number_of_generations)]

axs[0].plot(x,generation_results['best_fitnesses'])
axs[0].set(title='Best fitnesses')

axs[1].plot(x,generation_results['worst_fitnesses'])
axs[1].set(title='worst_fitnesses')

axs[2].plot(x,generation_results['average_fitnesses'])
axs[2].set(title='average_fitnesses')

fig.tight_layout(pad=3)

plt.show()