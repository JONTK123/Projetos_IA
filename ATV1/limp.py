import pandas as pd

def limp(df: pd.DataFrame) -> pd.DataFrame:
    df_filtrado = df[
        (df["age"] >= 10000) & (df["age"] <= 30000) &
        (df["height"] >= 120) & (df["height"] <= 220) &
        (df["weight"] >= 30) & (df["weight"] <= 200) &
        (df["ap_hi"] >= 90) & (df["ap_hi"] <= 250) &
        (df["ap_lo"] >= 60) & (df["ap_lo"] <= 150)
    ].copy()

    return df_filtrado
