import pandas as pd

# Lectura de un archivo con un dataframe
csvFile = "SismosMagnitud.csv"
df = pd.read_csv(csvFile)

# Eliminación de las columnas que no se quieren en el CSV
deleteColumns = ["Depth Error", "Depth Seismic Stations", "Magnitude Type", "Magnitude Error", "Magnitude Seismic Stations", "Azimuthal Gap",	"Horizontal Distance", "Horizontal Error", "Root Mean Square"]
df = df.drop(columns=deleteColumns)

# Guardado del nuevo archivo CSV con las columnas eliminadas
newCSVFile = "SismosMagnitud1.1.csv"
df.to_csv(newCSVFile, index=False)