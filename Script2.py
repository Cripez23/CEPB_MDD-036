import pandas as pd

csvFile = "P1/DonacionOrganos.csv"
df = pd.read_csv(csvFile)

deleteColumns = ["EDAD_ANIOS", "RINON_BLOCK", "INTESTINO", "CORNEA_IZQUIERDA", "CORNEA_DERECHA", "PIEL", "HUESOS", "CORAZON_TEJIDOS"]
df = df.drop(columns=deleteColumns)

newCSVFile = "P2/DonacionOrganos1.1.csv"
df.to_csv(newCSVFile, index=False)