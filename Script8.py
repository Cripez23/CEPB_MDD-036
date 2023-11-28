import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import os

def k_means(points: List[np.array], k: int, name: str):
    dim = len(points[0])
    N = len(points)
    num_cluster = k
    iterations = 5
    x = np.array(points)
    y = np.random.randint(0, num_cluster, N)
    mean = np.zeros((num_cluster, dim))
    for t in range(iterations):
        for k in range(num_cluster):
            cluster_points = x[y == k]
            if len(cluster_points) > 0:
                mean[k] = np.mean(cluster_points, axis=0)
            else:
                mean[k] = np.nan 
        for i in range(N):
            dist = np.sum((mean - x[i]) ** 2, axis=1)
            pred = np.argmin(dist)
            y[i] = pred
    for kl in range(num_cluster):
        xp = x[y == kl, 0]
        yp = x[y == kl, 1]
        plt.scatter(xp, yp)
    if not os.path.exists("P8"):
        os.makedirs("P8")
    plt.savefig(f"P8/{name}.png")
    plt.close()

    return mean

df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df = df.dropna()
df_mean = df.groupby(['CODIGO_SEXO', 'INSTITUCION'])['RINON_IZQUIERDO'].mean().reset_index()
df_mean = df_mean.drop('INSTITUCION', axis=1)
list_t = [
     (np.array(tuples[0:2]), tuples[1])
     for tuples in df_mean.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
kn = k_means(
     points,
     4,
     'kmeans'
)
print(kn)

# ------------------------------------------------------------------------------------------
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'PULMON_IZQUIERDO', 'PULMON_DERECHO', 'CORAZON', 'HIGADO', 'PANCREAS']
df1 = df.drop(eliminar, axis=1)
list_t = [
    (np.array(tuples[0:2]), tuples[1])
    for tuples in df1.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
kn = k_means(
    points,
    4,
    'Kmeans2'
)
print(kn)

# ------------------------------------------------------------------------------------------
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'RINON_IZQUIERDO', 'RINON_DERECHO', 'CORAZON', 'HIGADO', 'PANCREAS']
df1 = df.drop(eliminar, axis=1)
list_t = [
    (np.array(tuples[0:2]), tuples[1])
    for tuples in df1.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
kn = k_means(
    points,
    4,
    'Kmeans3'
)
print(kn)
