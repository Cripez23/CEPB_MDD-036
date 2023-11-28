import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd

# ------------------------------------------------------------------------------------------
# Regresion por entidad - corazón 
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['CORAZON'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['CORAZON']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['CORAZON'], label='Promedios por Corazón')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Corazón')
plt.title('Regresion lineal: Promedio de traspaso de corazón por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-Corazon.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - pancreas
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['PANCREAS'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['PANCREAS']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['PANCREAS'], label='Promedios por Pancreas')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Pancreas')
plt.title('Regresión lineal: Promedio de traspaso de pancreas por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-Pancreas.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - higado
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['HIGADO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['HIGADO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['HIGADO'], label='Promedios por Higado')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Higado')
plt.title('Regresión lineal: Promedio de traspaso de higado por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-Higado.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - Riñon Izquierdo
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['RINON_IZQUIERDO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['RINON_IZQUIERDO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['RINON_IZQUIERDO'], label='Promedios por Riñon Izquierdo')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Riñon Izquierdo')
plt.title('Regresión lineal: Promedio de traspaso de Riñon Izquierdo por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-RI.png")
plt.tight_layout()
plt.close()
print(df_mean)  
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - Riñon Derecho
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['RINON_DERECHO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['RINON_DERECHO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['RINON_DERECHO'], label='Promedios por Riñon Derecho')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Riñon Derecho')
plt.title('Regresión lineal: Promedio de traspaso de Riñon Derecho por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-RD.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - Pulmon Derecho
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['PULMON_DERECHO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['PULMON_DERECHO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['PULMON_DERECHO'], label='Promedios por Pulmon Derecho')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Pulmon Derecho')
plt.title('Regresión lineal: Promedio de traspaso de Pulmon Derecho por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-PD.png")
plt.tight_layout()
plt.close()
print(df_mean)
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresion por entidad - Riñon Derecho
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby('ENTIDAD_FEDERATIVA')['PULMON_IZQUIERDO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['PULMON_IZQUIERDO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['PULMON_IZQUIERDO'], label='Promedios por Pulmon Izquierdo')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de Pulmon Izquierdo')
plt.title('Regresión lineal: Promedio de traspaso de Pulmon Derecho por entidad federativa')
plt.legend()
plt.savefig("P6/RegresionPEF-PI.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)


# ------------------------------------------------------------------------------------------
# Regresión lineal por entidad-sexo para Corazón 
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['ENTIDAD_FEDERATIVA','CODIGO_SEXO'])['CORAZON'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,
                    11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,27,28,29,30,
                    31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,
                    51,52,53,54,55,56,57,58,59,60,
                    61,62,63,64]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['CORAZON']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['CORAZON'], label='Promedios por Sexo')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de traspasos de corazón')
plt.title('Regresion lineal: Promedio de traspasos de corazón por entidades para ambos sexos')

plt.legend()
plt.savefig("P6/RegresionPEyS-Corazon.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresión lineal por entidad-sexo para Pancreas 
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['ENTIDAD_FEDERATIVA','CODIGO_SEXO'])['PANCREAS'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,
                    11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,27,28,29,30,
                    31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,
                    51,52,53,54,55,56,57,58,59,60,
                    61,62,63,64]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['PANCREAS']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['PANCREAS'], label='Promedios por Sexo')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de traspasos de corazón')
plt.title('Regresion lineal: Promedio de traspasos de pancreas por entidades para ambos sexos')

plt.legend()
plt.savefig("P6/RegresionPEyS-Pancreas.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
# Regresión lineal por entidad-sexo para Pancreas 
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_mean = df.groupby(['ENTIDAD_FEDERATIVA','CODIGO_SEXO'])['HIGADO'].mean().reset_index()
df_mean['CEF'] = [1,2,3,4,5,6,7,8,9,10,
                    11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,27,28,29,30,
                    31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,
                    51,52,53,54,55,56,57,58,59,60,
                    61,62,63,64]

X = sm.add_constant(df_mean['CEF'])
Y = df_mean['HIGADO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df_mean['CEF'], df_mean['HIGADO'], label='Promedios por Sexo')
plt.plot(df_mean['CEF'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Entidades por codigo')
plt.ylabel('Promedio de traspasos de corazón')
plt.title('Regresion lineal: Promedio de traspasos de pancreas por entidades para ambos sexos')

plt.legend()
plt.savefig("P6/RegresionPEyS-Pancreas.png")
plt.tight_layout()
plt.close()
print(df_mean) 
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
X = sm.add_constant(df['RINON_IZQUIERDO'])
Y = df['RINON_DERECHO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df['RINON_IZQUIERDO'], df['RINON_DERECHO'], label='Riñon izquierdo')
plt.plot(df['RINON_IZQUIERDO'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Riñon izquierdo')
plt.ylabel('Riñon derecho')
plt.title('Regresion lineal: Riñon izquierdo a Riñon derecho')
plt.legend()
plt.savefig("P6/RegresionRiñones.png")
plt.tight_layout()
plt.close()
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)

# ------------------------------------------------------------------------------------------
X = sm.add_constant(df['PULMON_IZQUIERDO'])
Y = df['PULMON_DERECHO']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df['PULMON_IZQUIERDO'], df['PULMON_DERECHO'], label='Pulmon Izquierdo')
plt.plot(df['PULMON_IZQUIERDO'], modelo.predict(X), color='red', label='Pendiente de Regresión')
plt.xlabel('Pulmon Izquierdo')
plt.ylabel('Pulmon Derecho')
plt.title('Regresion lineal: Pulmon Izquierdo a Pulmon Derecho ')
plt.legend()
plt.savefig("P6/RegresionPulmones.png")
plt.tight_layout()
plt.close()
predicciones = modelo.predict(sm.add_constant(X))
spearman, p_valor = spearmanr(predicciones, Y)
print("Coeficiente de correlación de Spearman:", spearman)
print("Valor p:", p_valor)
coeficiente_pearson, p_value = pearsonr(predicciones, Y)
print("Coeficiente de correlación de Pearson:", coeficiente_pearson)
print("Valor p:", p_value)