
from pathlib import Path
import numpy as np
import xarray as xr

# Directories
base = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra"
    r"\EODP_TER_2021\EODP-TS-ISM"
)

myoutput = base / "myoutput"
reference = base / "output"

# Tolerances
REL_TOL = 1e-3  # Percentage
ABS_TOL = 1e-8

# Products to validate
stages = {
    "OPTICAL STAGE": "ism_toa_optical",
    "DETECTION STAGE": "ism_toa_detection",
    "FINAL ISM OUTPUT": "ism_toa",
}

for stage, prefix in stages.items():

    print(f"\n{stage} CROSS-VALIDATION")
    print("-" * 68)
    print(f"{'Band':<10} {'Max Abs Diff':<18} "
          f"{'Max Rel Diff (%)':<20} {'Result'}")
    print("-" * 68)

    for i in range(4):
        band = f"VNIR-{i}"
        filename = f"{prefix}_{band}.nc"

        myfile = myoutput / filename
        reffile = reference / filename

        if not myfile.exists() or not reffile.exists():
            print(f"{band:<10} FILE NOT FOUND: {filename}")
            continue

        with xr.open_dataset(myfile) as ds_my, \
             xr.open_dataset(reffile) as ds_ref:

            variables = list(ds_ref.data_vars)

            if len(variables) != 1:
                print(f"{band}: Check variables: {variables}")
                continue

            var = variables[0]

            if var not in ds_my.data_vars:
                print(f"{band}: Variable {var} not found")
                continue

            a = np.asarray(ds_my[var].values, dtype=float)
            b = np.asarray(ds_ref[var].values, dtype=float)

            if a.shape != b.shape:
                print(f"{band}: SHAPE MISMATCH")
                continue

            if not np.all(np.isfinite(a)) or \
               not np.all(np.isfinite(b)):
                print(f"{band}: NON-FINITE VALUES")
                continue

            # Absolute difference
            abs_diff = np.abs(a - b)
            max_abs = np.max(abs_diff)

            # Relative difference
            mask = np.abs(b) > ABS_TOL

            if np.any(mask):
                rel_diff = (
                    100 * abs_diff[mask] / np.abs(b[mask])
                )
                max_rel = np.max(rel_diff)
                rel_ok = np.all(rel_diff < REL_TOL)
            else:
                max_rel = 0.0
                rel_ok = True

            # Absolute criterion near zero
            zero_ok = np.all(abs_diff[~mask] <= ABS_TOL)

            result = "PASS" if rel_ok and zero_ok else "FAIL"

            print(
                f"{band:<10} {max_abs:<18.8e} "
                f"{max_rel:<20.8e} {result}"
            )

print("\nISM Cross-validation completed.")
