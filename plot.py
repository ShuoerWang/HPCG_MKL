import os
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.getcwd()

results = {}

for item in os.listdir(current_dir):
    if os.path.isdir(item):
        results[item] = None
        for file in os.listdir(item):
            if file.startswith("HPCG-Benchmark"):
                with open(os.path.join(item, file)) as f:
                    contents = f.read()
                for line in contents.splitlines():
                    if "Benchmark Time Summary::SpMV=" in line:
                        number = float(line.split("Benchmark Time Summary::SpMV=")[1])
                        results[item] = number

grouped_results = {'clang++': [], 'g++': [], 'icpx': []}
for key, value in results.items():
    if key.startswith('clang++'):
        grouped_results['clang++'].append(value)
    elif key.startswith('g++'):
        grouped_results['g++'].append(value)
    elif key.startswith('icpx'):
        grouped_results['icpx'].append(value)

optimization = ['O0', 'O1', 'O2', 'O3']
x = np.arange(len(optimization))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in grouped_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SpMV benchmark')
ax.set_xlabel('Compiler Optimization')
ax.set_xticks(x + width, list(optimization))
ax.legend(loc='upper right', ncols=3)
ax.set_title('serial')
ax.set_ylim(0, 0.02)

plt.show()

plt.tight_layout()
plt.savefig('hpcg_benchmark_results.png')

print(results)