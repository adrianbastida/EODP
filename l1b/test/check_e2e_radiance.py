
from pathlib import Path
import numpy as np
import xarray as xr

folder = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra"
    r"\EODP-TS-E2E\myoutput_l1b"
)

for i in range(4):
    band = f"VNIR-{i}"
    path = folder / f"l1b_toa_{band}.nc"

    with xr.open_dataset(path) as ds:
        toa = np.asarray(ds["toa"].values)

        print(f"\n{band}")
        print("Minimum:", np.nanmin(toa))
        print("Maximum:", np.nanmax(toa))
        print("Unique values:", len(np.unique(toa)))
