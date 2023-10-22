import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


#1
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICRZ = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_IZQUIERDO"]].sum()
df_ICRZ.reset_index(inplace=True)
df_ICRZ.set_index("INSTITUCION", inplace=True)
df_ICRZ.reset_index(inplace=True)
df_Anova = df_ICRZ.rename(columns={"RINON_IZQUIERDO" : "RiñonIzq"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("RiñonIzq ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

#2
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICRD = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_DERECHO"]].sum()
df_ICRD.reset_index(inplace=True)
df_ICRD.set_index("INSTITUCION", inplace=True)
df_ICRD.reset_index(inplace=True)
df_Anova = df_ICRD.rename(columns={"RINON_DERECHO" : "RiñonDer"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("RiñonDer ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

#3
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICPI = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_IZQUIERDO"]].sum()
df_ICPI.reset_index(inplace=True)
df_ICPI.set_index("INSTITUCION", inplace=True)
df_ICPI.reset_index(inplace=True)
df_Anova = df_ICPI.rename(columns={"PULMON_IZQUIERDO" : "PulomIzq"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("PulomIzq ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

#4
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICPD = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_DERECHO"]].sum()
df_ICPD.reset_index(inplace=True)
df_ICPD.set_index("INSTITUCION", inplace=True)
df_ICPD.reset_index(inplace=True)
df_Anova = df_ICPD.rename(columns={"PULMON_DERECHO" : "PulomDer"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("PulomDer ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

#5
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICC = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["CORAZON"]].sum()
df_ICC.reset_index(inplace=True)
df_ICC.set_index("INSTITUCION", inplace=True)
df_ICC.reset_index(inplace=True)
df_Anova = df_ICC.rename(columns={"CORAZON" : "Corazón"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("Corazón ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

#6
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_ICH = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["HIGADO"]].sum()
df_ICH.reset_index(inplace=True)
df_ICH.set_index("INSTITUCION", inplace=True)
df_ICH.reset_index(inplace=True)
df_Anova = df_ICH.rename(columns={"HIGADO" : "Higado"}).drop(['CODIGO_SEXO'], axis=1)
print(df_Anova.head())

model = ols("Higado ~ INSTITUCION", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por institución.")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por institución")

# ----------------------------------------------------------------------------------------------

#1
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIRZ = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_IZQUIERDO"]].sum()
df_CIRZ.reset_index(inplace=True)
df_CIRZ.set_index("CODIGO_SEXO", inplace=True)
df_CIRZ.reset_index(inplace=True)
df_Anova = df_CIRZ.rename(columns={"RINON_IZQUIERDO" : "RiñonIzq"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("RiñonIzq ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")

#2
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIRD = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["RINON_DERECHO"]].sum()
df_CIRD.reset_index(inplace=True)
df_CIRD.set_index("CODIGO_SEXO", inplace=True)
df_CIRD.reset_index(inplace=True)
df_Anova = df_CIRD.rename(columns={"RINON_DERECHO" : "RiñonDer"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("RiñonDer ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")

#3
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIPI = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_IZQUIERDO"]].sum()
df_CIPI.reset_index(inplace=True)
df_CIPI.set_index("CODIGO_SEXO", inplace=True)
df_CIPI.reset_index(inplace=True)
df_Anova = df_CIPI.rename(columns={"PULMON_IZQUIERDO" : "PulomIzq"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("PulomIzq ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")

#4
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIPD = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["PULMON_DERECHO"]].sum()
df_CIPD.reset_index(inplace=True)
df_CIPD.set_index("CODIGO_SEXO", inplace=True)
df_CIPD.reset_index(inplace=True)
df_Anova = df_CIPD.rename(columns={"PULMON_DERECHO" : "PulomDer"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("PulomDer ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")

#5
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIC = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["CORAZON"]].sum()
df_CIC.reset_index(inplace=True)
df_CIC.set_index("CODIGO_SEXO", inplace=True)
df_CIC.reset_index(inplace=True)
df_Anova = df_CIC.rename(columns={"CORAZON" : "Corazón"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("Corazón ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")

#6
print("\n\n")
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
df_CIH = df.groupby(["INSTITUCION", "CODIGO_SEXO"])[["HIGADO"]].sum()
df_CIH.reset_index(inplace=True)
df_CIH.set_index("CODIGO_SEXO", inplace=True)
df_CIH.reset_index(inplace=True)
df_Anova = df_CIH.rename(columns={"HIGADO" : "Higado"}).drop(['INSTITUCION'], axis=1)
print(df_Anova.head())

model = ols("Higado ~ CODIGO_SEXO", data=df_Anova).fit()
anovaDF = sm.stats.anova_lm(model, typ=1)
if anovaDF["PR(>F)"].iloc[0] < 0.005:
    print("Hay diferencias en los donativos por codigo de sexo")
    print(anovaDF)
else:
    print("No hay diferencias en los donativos por codigo de sexo")