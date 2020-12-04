import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='run training loop')
parser.add_argument('csv')
args = parser.parse_args()


solution_length = pd.read_csv(args.csv)
solution_length_filtered = solution_length[solution_length['DeepQube'] != 9999]
success = pd.read_csv(args.csv)
success['DeepQube'] = success['DeepQube'].apply(
    lambda x: 'Solved' if x != 9999 else 'Fail')

gods_number_moves = np.arange(4, 15)
gods_number_range = np.linspace(4, 15, 100)
gods_number_frequency = 0.01*np.array([0.015, 0.061, 0.24, 0.90, 3.11, 9.81, 25.33, 36.77, 21.3, 2.46, 0.0075])
z = np.polyfit(gods_number_moves, 5000*gods_number_frequency, 2)
p = np.poly1d(z)

plt.figure(1, figsize=(16, 9))
plt.plot(gods_number_range, p(gods_number_range), 'r--')
plot1 = sns.histplot(data=solution_length_filtered, binwidth=1, legend=False)
plt.legend(('Probabilistic distribution \n of optimal solutions', 'Baseline', 'DeepQube'))
plot1.set(xlabel='Solution move count (quarter turn metric)')
plt.ylim(bottom=0)

plt.savefig('histogram.png', dpi=300)

plt.figure(2)
plot2 = sns.countplot(x='DeepQube', data=success, edgecolor=(
    0, 0, 0), palette=['limegreen', 'orangered'])
plot2.set(xlabel='', ylabel='Count')
i = 0
for p in plot2.patches:
    h = p.get_height()

    count = success['DeepQube'].value_counts()[i]

    plot2.text(p.get_x() + p.get_width()/2.0, h + 50,
               f'{count} ({count/5000 * 100}%)', ha='center')
    i += 1

plt.savefig('count_plot.png', dpi=300)
