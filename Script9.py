import numbers
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

def transform_variable(df: pd.DataFrame, x:str) -> pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x]
    else:
        return pd.Series([i for i in range(0, len(df[x]))])
def modif_col(df_mean: pd.DataFrame, x:str, val_inf:int, val_sup:int):
    if val_inf != 0 and val_sup != 0:
        df = df_mean.loc[(df_mean[x] >= val_inf) & (df_mean[x]<=val_sup)].reset_index()
        print(df)
        return df
    else:
        return df_mean
def linear_reg(df: pd.DataFrame, x:str, y: str) -> Dict[str, float]:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())
    coef = model.params
    m = coef.values[1]
    b = coef.values[0]
    r_2_t_df = pd.DataFrame(model.summary().tables[0])
    r2 = r_2_t_df.values[0][3]
    r2_adj = r_2_t_df.values[1][3]
    band_table = model.summary().tables[1][1]
    lb = str(band_table[5])
    lb = float(lb)
    hb = str(band_table[6])
    hb = float(hb)
    return {'m': m, 'b': b, 'r2': r2, 'r2_adj': r2_adj, 'low_band': lb, 'hi_band': hb}

def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, low_band: float, hi_band: float):
    fixed_x = transform_variable(df, x)
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[m * x + b for _, x in fixed_x.items()], color='green')
    plt.fill_between(df[x], [m * x + low_band for _, x in fixed_x.items()], [m * x + hi_band for _, x in fixed_x.items()], alpha=0.5, color='red')

df = pd.read_csv("P2/DonacionOrganos1.1.csv")
# ------------------------------------------------------------------------------------------
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'RINON_IZQUIERDO', 'RINON_DERECHO', 'CORAZON', 'HIGADO', 'PANCREAS']
df1 = df.drop(eliminar, axis=1)
df1.reset_index(inplace=True)
df1 = df1.dropna()
df1 = df1.loc[df1['PULMON_IZQUIERDO'] <= 10]
lin = linear_reg(df1, "PULMON_DERECHO", "PULMON_IZQUIERDO")
plt_lr(df=df1, x="PULMON_DERECHO", y="PULMON_IZQUIERDO", **lin)
plt.xticks(rotation=90)
plt.savefig('P9/relConPID.png')
plt.close()

# ------------------------------------------------------------------------------------------
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'PULMON_IZQUIERDO', 'PULMON_DERECHO', 'CORAZON', 'HIGADO', 'PANCREAS']
df1 = df.drop(eliminar, axis=1)
df1.reset_index(inplace=True)
df1 = df1.dropna()
df1 = df1.loc[df1['RINON_IZQUIERDO'] <= 10]
lin = linear_reg(df1, "RINON_DERECHO", "RINON_IZQUIERDO")
plt_lr(df=df1, x="RINON_DERECHO", y="RINON_IZQUIERDO", **lin)
plt.xticks(rotation=90)
plt.savefig('P9/relConRID.png')
plt.close()

# ------------------------------------------------------------------------------------------
# Se hará la predicción para el CORAZON CON HIGADO
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'RINON_IZQUIERDO', 'RINON_DERECHO', 'PULMON_IZQUIERDO', 'PULMON_DERECHO', 'PANCREAS']
df1 = df.drop(eliminar, axis=1)
df1.reset_index(inplace=True)
df1 = df1.dropna()
df1 = df1.loc[df1['CORAZON'] <= 10]
lin = linear_reg(df1, "HIGADO", "CORAZON")
plt_lr(df=df1, x="HIGADO", y="CORAZON", **lin)
plt.xticks(rotation=90)
plt.savefig('P9/relConCH.png')
plt.close()

# ------------------------------------------------------------------------------------------
# Se hará la predicción para el CORAZON CON PANCREAS
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'RINON_IZQUIERDO', 'RINON_DERECHO', 'PULMON_IZQUIERDO', 'PULMON_DERECHO', 'HIGADO']
df1 = df.drop(eliminar, axis=1)
df1.reset_index(inplace=True)
df1 = df1.dropna()
df1 = df1.loc[df1['CORAZON'] <= 10]
lin = linear_reg(df1, "PANCREAS", "CORAZON")
plt_lr(df=df1, x="PANCREAS", y="CORAZON", **lin)
plt.xticks(rotation=90)
plt.savefig('P9/relConCP.png')
plt.close()

# ------------------------------------------------------------------------------------------
# Se hará la predicción para el PANCREAS CON HIGADO
eliminar = ['SEXO','CODIGO_SEXO', 'TIPO_DONANTE','MUERTE','ENTIDAD_FEDERATIVA','CODIGO_ENTIDAD_FEDERATIVA','ESTABLECIMIENTO','INSTITUCION','FECHA_PROCURACION',
            'RINON_IZQUIERDO', 'RINON_DERECHO', 'PULMON_IZQUIERDO', 'PULMON_DERECHO', 'CORAZON']
df1 = df.drop(eliminar, axis=1)
df1.reset_index(inplace=True)
df1 = df1.dropna()
df1 = df1.loc[df1['PANCREAS'] <= 10]
lin = linear_reg(df1, "HIGADO", "PANCREAS")
plt_lr(df=df1, x="HIGADO", y="PANCREAS", **lin)
plt.xticks(rotation=90)
plt.savefig('P9/relConPH.png')
plt.close()
