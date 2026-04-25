import numpy as np
import pandas as pd

def get_season(month):
    if month in [6,7,8,9]:
        return "SWM"
    elif month in [10,11,12]:
        return "NEM"
    elif month in [3,4,5]:
        return "PRE"
    else:
        return "WIN"

def seasonal_aggregate(dates, values):
    """
    dates  : list of datetime
    values : np.array (time,)
    """
    df = pd.DataFrame({
        "date": dates,
        "val": values
    })
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)

    # Sum rainfall per season
    seasonal = df.groupby("season")["val"].sum()
    return seasonal
