from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def open_file(path: str) -> str:
    content = ""
    with open(path, "r") as f:
        content = f.readlines()
    return " ".join(content)
def open_dataframe_column(df: pd.DataFrame, column_name: str) -> str:
    if column_name not in df.columns:
        raise ValueError(f"La columna '{column_name}' no existe en el DataFrame.")
    content = " ".join(df[column_name].astype(str))

    return content

all_words = ""
frase = open_file("texto.txt")
palabras = frase.rstrip().split(" ")
for arg in palabras:
    tokens = arg.split()
    all_words += " ".join(tokens) + " "
wordcloud = WordCloud(
    background_color="white", min_font_size=5
).generate(all_words)
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("P10/PA.png")
plt.close()

# ------------------------------------------------------------------------------------------
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
palabras = open_dataframe_column(df, 'INSTITUCION')
palabras = " ".join(palabras.split())
wordcloud2 = WordCloud(background_color="white", min_font_size=5).generate(palabras)
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("P10/PAlabrasInstitución.png")
plt.close()

# ------------------------------------------------------------------------------------------
palabras2 = open_dataframe_column(df, 'ESTABLECIMIENTO')
palabras2 = " ".join(palabras2.split())
wordcloud3 = WordCloud(background_color="white", min_font_size=5).generate(palabras2)
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud3)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("P10/PAEstablecimiento.png")
plt.close()

# ------------------------------------------------------------------------------------------
df = pd.read_csv("P2/DonacionOrganos1.1.csv")
palabras = open_dataframe_column(df, 'ENTIDAD_FEDERATIVA')
palabras = " ".join(palabras.split())
wordcloud2 = WordCloud(background_color="white", min_font_size=5).generate(palabras)
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("P10/PAEntidadFederativa.png")
plt.close()