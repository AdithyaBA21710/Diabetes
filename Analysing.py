from sklearn import datasets
from matplotlib import pyplot as plt


diabetes = datasets.load_diabetes(as_frame=True)
features = diabetes['feature_names']
features.remove('sex')

fig, axs = plt.subplots(3, 3)
fig.suptitle('Diabetes Dataset')
for i in range(3):
    for j in range(3):
        n = j + i * 3
        feature = features[n]
        axs[i, j].scatter(diabetes['data'][feature], diabetes['target'], s=1)
        axs[i, j].set_xlabel(feature)
        axs[i, j].set_ylabel('target')
plt.tight_layout()
plt.show()
