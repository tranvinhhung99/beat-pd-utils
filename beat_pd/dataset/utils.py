import pandas as pd
import os

import numpy as np
import torch

def interpolate_series(data, 
                       timestamp_step: float=0.02, 
                       start: int=0, 
                       end: int=60000, 
                       resolution: int=10):
    """ Linear interpolate for timeseries data
    
    timestamp_step: step to split timestamp
    start: start index
    end: End index
    resolution: precision of linear interpolate
    """
    df = data
    df['Timestamp'] = np.array(np.round(df['Timestamp'] / (timestamp_step / resolution)), dtype=int)

    new_timestamp = np.arange(start * resolution, end * resolution, 1, dtype=int)

    # drop duplicated
    final = df[np.bitwise_not(df.duplicated('Timestamp', keep='last'))]

    # resample and interpolate
    final = final.set_index("Timestamp").reindex(new_timestamp).interpolate().reset_index()
    final = final[final.Timestamp % resolution == 0] 

    # Normalize for other function    print(len(new_timestamp))
    final.Timestamp *= (timestamp_step / resolution)
    final.index //= resolution
    return final

def convert_raw_to_timeseries(raw_data: pd.DataFrame, interpolate=True) -> torch.Tensor:
    # Take xyz raw data and convert to signal
    data = raw_data
    if interpolate:
        data = interpolate_series(data)
    
    return torch.tensor([data.X, data.Y, data.Z])
    