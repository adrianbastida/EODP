
from pathlib import Path
import numpy as np
import xarray as xr

# Folder containing your outputs
base = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra"
    r"\EODP_TER_2021\EODP-TS-ISM"
)

myoutput = base / "myoutput"
reference = base / "output"

# Acceptance criteria
REL_TOL = 1e-3  # Relative tolerance (%)
ABS_TOL = 1e-8  # For reference values close to zero

print("\nOPTICAL PHASE CROSS-VALIDATION\n")
print(f"{'Band':<10} {'Max Abs Diff':<18} "
      f"{'Max Rel Diff (%)':<20} {'Result'}")
print("-" * 60)

for i in range(4):
    band = f"VNIR-{i}"
    filename = f"ism_toa_optical_{band}.nc"

    myfile = myoutput / filename
    reffile = reference / filename

    if not myfile.exists() or not reffile.exists():
        print(f"{band:<10} FILE NOT FOUND")
        continue

    with xr.open_dataset(myfile) as ds_my, \
         xr.open_dataset(reffile) as ds_ref:

        variables = list(ds_ref.data_vars)

        if len(variables) != 1:
            print(f"{band}: Variables found: {variables}")
            print("Select the image variable before comparing.")
            continue

        var = variables[0]

        if var not in ds_my.data_vars:
            print(f"{band}: Variable '{var}' not found.")
            continue

        a = np.asarray(ds_my[var].values, dtype=float)
        b = np.asarray(ds_ref[var].values, dtype=float)

        if a.shape != b.shape:
            print(f"{band}: Different shapes: {a.shape}, {b.shape}")
            continue

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            print(f"{band}: Non-finite values detected.")
            continue

        abs_diff = np.abs(a - b)
        max_abs = np.max(abs_diff)

        # Relative error where reference values are significant
        mask = np.abs(b) > ABS_TOL

        if np.any(mask):
            rel_diff = 100 * abs_diff[mask] / np.abs(b[mask])
            max_rel = np.max(rel_diff)
            relative_ok = np.all(rel_diff < REL_TOL)
        else:
            max_rel = 0.0
            relative_ok = True

        # Absolute tolerance for near-zero reference values
        zero_ok = np.all(abs_diff[~mask] <= ABS_TOL)

        passed = relative_ok and zero_ok
        result = "PASS" if passed else "FAIL"

        print(f"{band:<10} {max_abs:<18.8e} "
              f"{max_rel:<20.8e} {result}")

print("\nCross-validation completed.")
