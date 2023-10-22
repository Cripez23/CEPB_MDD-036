import pandas as pd
from tabulate import tabulate
from typing import Tuple, List

#Lectura del CSV para guardarlo en un dataframe
csvFile = "P2/DonacionOrganos1.1.csv"
df = pd.read_csv(csvFile)

df_total = df.groupby(["CODIGO_SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'RINON_IZQUIERDO': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalRiñonIzquierdo.csv"
df_total.to_csv(newCSVFile, index=False)

df_total = df.groupby(["CODIGO_SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'RINON_DERECHO': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalRiñonDerecho.csv"
df_total.to_csv(newCSVFile, index=False)

df_total = df.groupby(["SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'PULMON_IZQUIERDO': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalPulmonIzquierdo.csv"
df_total.to_csv(newCSVFile, index=False)

df_total = df.groupby(["CODIGO_SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'PULMON_DERECHO': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalPulmonDerecho.csv"
df_total.to_csv(newCSVFile, index=False)

df_total = df.groupby(["CODIGO_SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'CORAZON': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalCorazón.csv"
df_total.to_csv(newCSVFile, index=False)

df_total = df.groupby(["CODIGO_SEXO", "CODIGO_ENTIDAD_FEDERATIVA"]).agg({'HIGADO': ['sum', 'count', 'mean', 'min', 'max']})
df_total = df_total.reset_index()
newCSVFile = "P3/TotalHigado.csv"
df_total.to_csv(newCSVFile, index=False)


