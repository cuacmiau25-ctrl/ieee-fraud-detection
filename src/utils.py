"""Utilidades compartidas por los notebooks del proyecto de fraude IEEE-CIS."""
import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce el uso de memoria de un DataFrame haciendo downcasting de tipos numéricos
    y convirtiendo columnas de tipo object a category.

    No cambia el contenido de los datos, solo la representación en memoria: cada columna
    numérica se reduce al tipo más pequeño que sigue representando sus valores sin perder
    precisión ni rango.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    int_types = [np.int8, np.int16, np.int32, np.int64]
    float_types = [np.float32, np.float64]

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            continue

        if str(col_type).startswith("category"):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if pd.api.types.is_integer_dtype(col_type):
            for t in int_types:
                info = np.iinfo(t)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(t)
                    break
        elif pd.api.types.is_float_dtype(col_type):
            if pd.isna(c_min) or pd.isna(c_max):
                df[col] = df[col].astype(np.float32)
            else:
                for t in float_types:
                    info = np.finfo(t)
                    if c_min >= info.min and c_max <= info.max:
                        df[col] = df[col].astype(t)
                        break

    for col in df.select_dtypes(include="object").columns:
        num_unique = df[col].nunique(dropna=False)
        num_total = len(df[col])
        if num_unique / max(num_total, 1) < 0.5:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(
            f"Memoria: {start_mem:.2f} MB -> {end_mem:.2f} MB "
            f"(reducción del {100 * (start_mem - end_mem) / start_mem:.1f}%)"
        )
    return df
