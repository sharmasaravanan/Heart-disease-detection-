from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps

classifier = linear_model.LogisticRegression()

df = pd.read_csv("./devinormalized.csv")
features_name=['1','2','3','4','5','6','7','8','9','10','11','12','13']
X=np.array(df[features_name])
y=np.array(df['class'])

def f_per_particle(m, alpha):
    total_features = 13
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 13 # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
optimizer.reset()

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=2)

X_selected_features = X[:,pos==1]

df1 = pd.DataFrame(X_selected_features)
df1['class'] = pd.Series(y)

sns.pairplot(df, hue='class')
sns.pairplot(df1, hue='class')

plt.show()

df1.to_csv("./devifeatures1.csv")
