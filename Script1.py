
import pandas as pd
import requests
import io
def get_csv_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url).content
    return pd.read_csv(io.StringIO(r.decode('utf-8')))
df = get_csv_from_url("https://raw.githubusercontent.com/Cripez23/Dataset/main/database.csv")
df.to_csv("SismosMagnitud.csv", index=False)