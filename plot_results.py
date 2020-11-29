import seaborn as sns
import pandas as pd
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

plt.figure(1, figsize=(16, 9))
plot1 = sns.histplot(data=solution_length_filtered)
plot1.set(xlabel='Solution move count (quarter turn metric)')

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
               f"{count} ({count/5000 * 100}%)", ha="center")
    i += 1

plt.savefig('count_plot.png', dpi=300)
