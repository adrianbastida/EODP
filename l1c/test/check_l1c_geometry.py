
from pathlib import Path
import numpy as np
import xarray as xr

base = Path(
    r"C:\Users\adria\Desktop\carpeta\master\tierra"
    r"\EODP_TER_2021\EODP-TS-L1C"
)

REL_TOL = 1e-3  # %
ABS_TOL = 1e-8

print("\nL1C CROSS-VALIDATION (ALIGNED COORDINATES)")
print("-" * 76)
print(f"{'Band':<10} {'Max Abs Diff':<18} "
      f"{'Max Rel Diff (%)':<20} {'Result'}")
print("-" * 76)

for i in range(4):
    band = f"VNIR-{i}"
    filename = f"l1c_toa_{band}.nc"

    with xr.open_dataset(base / "myoutput" / filename) as my, \
         xr.open_dataset(base / "output" / filename) as ref:

        my_lat = np.asarray(my["lat"].values).ravel()
        my_lon = np.asarray(my["lon"].values).ravel()
        ref_lat = np.asarray(ref["lat"].values).ravel()
        ref_lon = np.asarray(ref["lon"].values).ravel()

        idx_my = np.lexsort((my_lon, my_lat))
        idx_ref = np.lexsort((ref_lon, ref_lat))

        if len(idx_my) != len(idx_ref):
            print(f"{band:<10} DIFFERENT NUMBER OF POINTS")
            continue

        if not (
            np.array_equal(my_lat[idx_my], ref_lat[idx_ref])
            and np.array_equal(my_lon[idx_my], ref_lon[idx_ref])
        ):
            print(f"{band:<10} GEOMETRY MISMATCH")
            continue

        a = np.asarray(my["toa"].values, dtype=float).ravel()[idx_my]
        b = np.asarray(ref["toa"].values, dtype=float).ravel()[idx_ref]

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            print(f"{band:<10} NON-FINITE VALUES")
            continue

        diff = np.abs(a - b)
        max_abs = np.max(diff)
        # Find the 5 pixels with the largest absolute errors
        #indices = np.argsort(diff)[-5:][::-1]

       #print(f"\n{band} - TOP 5 ERRORS")

        # for idx in indices:
        #  print(
        #       f"Latitude: {my_lat[idx_my][idx]:.6f}, "
        #       f"Longitude: {my_lon[idx_my][idx]:.6f}, "
        #        f"Absolute error: {diff[idx]:.8e}"
        #   )

        mask = np.abs(b) > ABS_TOL

        if np.any(mask):
            rel_diff = 100 * diff[mask] / np.abs(b[mask])
            max_rel = np.max(rel_diff)
            rel_ok = np.all(rel_diff < REL_TOL)
        else:
            max_rel = 0.0
            rel_ok = True

        zero_ok = np.all(diff[~mask] <= ABS_TOL)
        # Additional error analysis
        #  failed_pixels = np.sum(
        #   (diff > ABS_TOL) &
        #   (diff > REL_TOL / 100 * np.abs(b))
        #)

        #mean_abs = np.mean(diff)
        #median_abs = np.median(diff)

        #idx_max = np.argmax(diff)

        #print(f"\n{band} ERROR ANALYSIS")
        #print(f"Pixels exceeding tolerance: {failed_pixels}/{len(diff)}")
        #print(f"Mean absolute difference: {mean_abs:.8e}")
        #print(f"Median absolute difference: {median_abs:.8e}")
        #print(f"Reference at maximum error: {b[idx_max]:.8e}")
        #print(f"My output at maximum error: {a[idx_max]:.8e}")
        result = "PASS" if rel_ok and zero_ok else "FAIL"

        print(
            f"{band:<10} {max_abs:<18.8e} "
            f"{max_rel:<20.8e} {result}"
        )

print("\nL1C Cross-validation completed.")
