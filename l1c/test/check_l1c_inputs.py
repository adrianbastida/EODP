
from pathlib import Path
import numpy as np
import xarray as xr

base = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra\EODP_TER_2021"
)

folder_a = base / "EODP-TS-L1C" / "input" / "l1b_output"
folder_b = base / "EODP-TS-L1B" / "output"

print("\nL1C INPUT COMPARISON\n")

for i in range(4):
    band = f"VNIR-{i}"
    filename = f"l1b_toa_{band}.nc"

    file_a = folder_a / filename
    file_b = folder_b / filename

    with xr.open_dataset(file_a) as ds_a, \
         xr.open_dataset(file_b) as ds_b:

        common_vars = set(ds_a.data_vars) & set(ds_b.data_vars)

        print(f"\n{band}")
        print(f"Variables: {sorted(common_vars)}")

        if set(ds_a.data_vars) != set(ds_b.data_vars):
            print("Different variable sets")
            continue

        for var in sorted(common_vars):
            a = np.asarray(ds_a[var].values, dtype=float)
            b = np.asarray(ds_b[var].values, dtype=float)

            if a.shape != b.shape:
                print(f"{var}: Different shapes")
                continue

            if not np.array_equal(a, b, equal_nan=True):
                max_diff = np.nanmax(np.abs(a - b))
                print(f"{var}: DIFFERENT | Max diff: {max_diff:.8e}")
            else:
                print(f"{var}: IDENTICAL")

print("\nComparison completed.")
