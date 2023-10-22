import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#1
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_IZQUIERDO"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpRiñonIzquierdo.png")
plt.close()

#2
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_DERECHO"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpRiñonDerecho.png")
plt.close()

#3
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_IZQUIERDO"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpPulmonIzquierdo.png")
plt.close()

#4
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_DERECHO"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpPulmonDerecho.png")
plt.close()

#5
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["CORAZON"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpCorazon.png")
plt.close()

#6
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimTotal = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["HIGADO"]].mean()
df_BimTotal.boxplot(by = 'INSTITUCION', figsize=(27,18))
plt.xticks(rotation=90)
plt.savefig("P4/BP/bpHigado.png")
plt.close()

# -------------------------------------------------------------------------------------

#1
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_IZQUIERDO"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "RINON_IZQUIERDO")
    plt.savefig(f"P4/PT/ptRiñonIzquierdo{bim}.png")
    plt.close()

#2
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_DERECHO"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "RINON_DERECHO")
    plt.savefig(f"P4/PT/ptRiñonDerecho{bim}.png")
    plt.close()

#3
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_IZQUIERDO"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "PULMON_IZQUIERDO")
    plt.savefig(f"P4/PT/ptPulmonIzquierdo{bim}.png")
    plt.close()

#4
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_DERECHO"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "PULMON_DERECHO")
    plt.savefig(f"P4/PT/ptPulmonDerecho{bim}.png")
    plt.close()

#5
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["CORAZON"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "CORAZON")
    plt.savefig(f"P4/PT/ptCorazon{bim}.png")
    plt.close()

#6
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_BimSum = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["HIGADO"]].sum()
df_BimSum.reset_index(inplace=True)
df_BimSum.set_index("INSTITUCION", inplace=True)
for bim in set(df_BimSum['CODIGO_SEXO']):
    df_BimSum[df_BimSum['CODIGO_SEXO'] == bim].plot(y = "HIGADO")
    plt.savefig(f"P4/PT/ptHigado{bim}.png")
    plt.close()
