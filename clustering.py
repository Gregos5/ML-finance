import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import random as rand
from sklearn.cluster import KMeans


def random_blobs(amount, max, clusters=1):
    Mat= [[0.0 for i in range(2)] for j in range(amount)]
    for k in range(clusters):
        for l in range(round(amount*k/clusters), round(amount*(k+1)/clusters)):
            Mat[l][0] = rand.randint(max*k/clusters ,max*(k+1)/clusters) + rand.randint(-10,10)*0.1
            Mat[l][1] = rand.randint(max*k/clusters ,max*(k+1)/clusters) + rand.randint(-10,10)*0.1  
    return Mat

n = 100
X = random_blobs(n,100, 4)

x = [0]*n
y = x
for i in range(n):
    x[i] = X[i][0]
    y[i] = X[i][1]
'''
plt.scatter(x,y)
plt.show()
'''

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
#print(labels)

colors = ["g.","r.", "y.", "c."]

for i in range(n):
    #print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 1, zorder = 100)

plt.show()
