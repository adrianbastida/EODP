
from pathlib import Path
import numpy as np
import xarray as xr

base = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra"
    r"\EODP_TER_2021\EODP-TS-L1C"
)

myoutput = base / "myoutput"
reference = base / "output"

REL_TOL = 1e-3  # %
ABS_TOL = 1e-8

print("\nL1C CROSS-VALIDATION")
print("-" * 80)
print(f"{'Band':<10} {'Variable':<12} {'Max Abs Diff':<18} "
      f"{'Max Rel Diff (%)':<20} {'Result'}")
print("-" * 80)

for i in range(4):
    band = f"VNIR-{i}"
    filename = f"l1c_toa_{band}.nc"

    myfile = myoutput / filename
    reffile = reference / filename

    if not myfile.exists() or not reffile.exists():
        print(f"{band}: FILE NOT FOUND")
        continue

    with xr.open_dataset(myfile) as ds_my, \
         xr.open_dataset(reffile) as ds_ref:

        for var in ["toa", "lat", "lon"]:

            if var not in ds_my or var not in ds_ref:
                print(f"{band:<10} {var:<12} MISSING VARIABLE")
                continue

            a = np.asarray(ds_my[var].values, dtype=float)
            b = np.asarray(ds_ref[var].values, dtype=float)

            if a.shape != b.shape:
                print(f"{band:<10} {var:<12} SHAPE MISMATCH")
                continue

            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                print(f"{band:<10} {var:<12} NON-FINITE VALUES")
                continue

            abs_diff = np.abs(a - b)
            max_abs = np.max(abs_diff)

            mask = np.abs(b) > ABS_TOL

            if np.any(mask):
                rel_diff = 100 * abs_diff[mask] / np.abs(b[mask])
                max_rel = np.max(rel_diff)
                rel_ok = np.all(rel_diff < REL_TOL)
            else:
                max_rel = 0.0
                rel_ok = True

            zero_ok = np.all(abs_diff[~mask] <= ABS_TOL)

            result = "PASS" if rel_ok and zero_ok else "FAIL"

            print(
                f"{band:<10} {var:<12} {max_abs:<18.8e} "
                f"{max_rel:<20.8e} {result}"
            )

print("\nL1C Cross-validation completed.")
