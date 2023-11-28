import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

def scatter_gruopby(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.savefig(file_path)
    plt.close()

def scatter_gruop_by1(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    filter_df = df[[x_column, y_column, label_column]].dropna(subset=[x_column, y_column]).dropna(subset=[label_column])
    cmap = plt.cm.get_cmap("viridis", len(pd.unique(filter_df[label_column])))
    label_colors = {label: cmap(i) for i, label in enumerate(pd.unique(filter_df[label_column]))}
    ax.scatter(
        filter_df[x_column],
        filter_df[y_column],
        c=filter_df[label_column].map(label_colors),
        label=label_column,
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'Scatter Group by {label_column}')
    ax.legend()
    plt.savefig(file_path)
    plt.close()
def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    predicted_labels = [
        np.argmax(np.bincount([label_to_number[labels[index]] for index in point_nearest]))
        for point_nearest in points_k_nearest
    ]
    return predicted_labels

# ------------------------------------------------------------------------------------------
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['CODIGO_SEXO', 'INSTITUCION'])['PANCREAS'].mean().reset_index()
label_to_number = {label: i for i, label in enumerate(df_mean['INSTITUCION'].unique())}
number_to_label = {i: label for i, label in enumerate(df_mean['INSTITUCION'].unique())}
scatter_gruopby("P7/GruposP.png", df_mean, "PANCREAS", "CODIGO_SEXO", "INSTITUCION")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df_mean.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df_mean[['PANCREAS','CODIGO_SEXO']].to_numpy(),
    df_mean['INSTITUCION'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)

# ------------------------------------------------------------------------------------------
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['CODIGO_SEXO', 'INSTITUCION'])['HIGADO'].mean().reset_index()

label_to_number = {label: i for i, label in enumerate(df_mean['INSTITUCION'].unique())}
number_to_label = {i: label for i, label in enumerate(df_mean['INSTITUCION'].unique())}

scatter_gruopby("P7/GruposH.png", df_mean, "HIGADO", "CODIGO_SEXO", "INSTITUCION")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df_mean.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df_mean[['HIGADO','CODIGO_SEXO']].to_numpy(),
    df_mean['INSTITUCION'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)
# ------------------------------------------------------------------------------------------
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['CODIGO_SEXO', 'INSTITUCION'])['CORAZON'].mean().reset_index()

label_to_number = {label: i for i, label in enumerate(df_mean['INSTITUCION'].unique())}
number_to_label = {i: label for i, label in enumerate(df_mean['INSTITUCION'].unique())}
scatter_gruopby("P7/GruposC.png", df_mean, "CORAZON", "CODIGO_SEXO", "INSTITUCION")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df_mean.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df_mean[['CORAZON','CODIGO_SEXO']].to_numpy(),
    df_mean['INSTITUCION'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)

# ------------------------------------------------------------------------------------------
label_to_number = {label: i for i, label in enumerate(df['INSTITUCION'].unique())}
number_to_label = {i: label for i, label in enumerate(df['INSTITUCION'].unique())}
scatter_gruopby("P7/gruposRID.png", df, "RINON_IZQUIERDO", "RINON_DERECHO", "INSTITUCION")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df[['RINON_IZQUIERDO','RINON_DERECHO']].to_numpy(),
    df['INSTITUCION'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)

# ------------------------------------------------------------------------------------------
label_to_number = {label: i for i, label in enumerate(df['INSTITUCION'].unique())}
number_to_label = {i: label for i, label in enumerate(df['INSTITUCION'].unique())}
scatter_gruopby("P7/gruposPID.png", df, "PULMON_IZQUIERDO", "PULMON_DERECHO", "INSTITUCION")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df[['PULMON_IZQUIERDO','PULMON_DERECHO']].to_numpy(),
    df['INSTITUCION'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)
df = df.sort_values(by='CODIGO_SEXO', ascending=True)


# ------------------------------------------------------------------------------------------
label_to_number = {label: i for i, label in enumerate(df['CODIGO_SEXO'].unique())}
number_to_label = {i: label for i, label in enumerate(df['CODIGO_SEXO'].unique())}

scatter_gruop_by1("P7/gruposCSRID.png", df, "RINON_IZQUIERDO", "RINON_DERECHO", "CODIGO_SEXO")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df[['RINON_IZQUIERDO', 'RINON_DERECHO']].to_numpy(),
    df['CODIGO_SEXO'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)

# ------------------------------------------------------------------------------------------
scatter_gruop_by1("P7/gruposCSPID.png", df, "PULMON_IZQUIERDO", "PULMON_DERECHO", "CODIGO_SEXO")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
kn = k_nearest_neightbors(
    df[['PULMON_IZQUIERDO', 'PULMON_DERECHO']].to_numpy(),
    df['CODIGO_SEXO'].to_numpy(),
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5
)
print(kn)

