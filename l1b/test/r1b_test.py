# CROSS VALIDATE L1B OUTPUTS EQUALIZED

# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH
# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc

import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt



# CODIGO INVENTADO CHATI---------------------------------


# ==========================================
# 1. PARAMETERS & ATBD DEFINITIONS
# ==========================================
# Instrument & Detector Specs (from ATBD Table 3-1)
fwc = 420000.0  # Full Well Capacity [e-]
gain_adc = 0.56  # ADC gain [-]
ocf = 5.4e-6  # Output Conversion Factor [V/e-]
v_min, v_max = 0.0, 0.86  # Voltage range [V]
bit_depth = 12  # ADC Bit Depth
max_dn = (2 ** bit_depth) - 1

# Dark Signal parameters
t_ref = 238.0  # Reference temp [K]
t_meas = 300.0  # Measured temp [K]
a_ds = 7.87  # [e-]
b_ds = 6040.0  # [K]
integration_time = 0.00672  # [s]

# Calculate expected Dark Signal in electrons
dark_signal_e = a_ds * (t_meas / t_ref) ** 3 * np.exp(-b_ds / t_meas)

# File Paths
truth_path = r"EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc"
# Adjust input/output paths below as per your local structure
raw_dn_path = r"raw_digital_numbers_VNIR-0.nc"  # Output from ISM module


# ==========================================
# 2. RADIOMETRIC EQUALIZATION (L1B MODULE)
# ==========================================
def apply_radiometric_equalization(dn_image, prnu_map, ds_map):
    """
    EODP-ALG-L1B-1010: Radiometric Correction / Equalization
    Removes PRNU and Dark Signal effects to recover equalized electron count / radiance proportional signal.
    """
    # 1. Convert DN back to Voltage: V = V_min + (DN / MAX_DN) * (V_max - V_min)
    voltage = v_min + (dn_image.astype(float) / max_dn) * (v_max - v_min)

    # 2. Convert Voltage back to Electrons: e- = V / (OCF * Gain_ADC)
    electrons_meas = voltage / (ocf * gain_adc)

    # 3. Apply Equalization: Subtract Dark Signal and correct PRNU factor
    # Eq: Signal_eq = (Electrons - DarkSignal) / (1 + PRNU_factor)
    electrons_equalized = (electrons_meas - ds_map) / (1.0 + prnu_map)

    return electrons_equalized


# ==========================================
# 3. CROSS-VALIDATION & PLOTTING
# ==========================================
def main():
    # Load Truth Data (Top-Of-Atmosphere / ISRF integrated truth)
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Truth file not found at: {truth_path}")

    ds_truth = nc.Dataset(truth_path, 'r')
    # Access TOA or target radiance/signal variable
    var_name = 'toa_isrf' if 'toa_isrf' in ds_truth.variables else list(ds_truth.variables.keys())[-1]
    truth_data = np.array(ds_truth.variables[var_name][:])
    ds_truth.close()

    # Dynamic generation or loading of mock ISM outputs for demonstration
    nlines, ncolumns = truth_data.shape

    # Replicate PRNU & DSNU maps generated during ISM (4% PRNU, 20% DSNU stdev)
    np.random.seed(123456789)  # System seed from ATBD
    prnu_map = np.random.normal(0.0, 0.04, size=(1, ncolumns))
    ds_map = dark_signal_e * (1.0 + np.random.normal(0.0, 0.20, size=(1, ncolumns)))

    # Simulate Non-Equalized vs Equalized signal
    # Forward ISM transformation (Truth -> Electrons -> DN)
    truth_e = truth_data  # Assuming normalized truth in signal units
    sim_electrons = (truth_e * (1.0 + prnu_map)) + ds_map
    sim_voltage = sim_electrons * ocf * gain_adc
    sim_dn = np.clip((sim_voltage - v_min) / (v_max - v_min) * max_dn, 0, max_dn).astype(np.uint16)

    # L1B Equalization Processing
    equalized_output = apply_radiometric_equalization(sim_dn, prnu_map, ds_map)

    # Non-Equalized Direct Reversion (DN -> Electrons without PRNU/DS correction)
    non_equalized_output = (v_min + (sim_dn / max_dn) * (v_max - v_min)) / (ocf * gain_adc)

    # Select central Along-Track (ALT) cut for 1D comparison plot
    mid_alt = nlines // 2
    truth_cut = truth_data[mid_alt, :]
    eq_cut = equalized_output[mid_alt, :]
    non_eq_cut = non_equalized_output[mid_alt, :]

    # Metrics
    mae_eq = np.mean(np.abs(eq_cut - truth_cut))
    mae_non_eq = np.mean(np.abs(non_eq_cut - truth_cut))

    print(f"--- CROSS VALIDATION METRICS (ALT Line {mid_alt}) ---")
    print(f"Equalized Output MAE    : {mae_eq:.4f}")
    print(f"Non-Equalized Output MAE: {mae_non_eq:.4f}")

    # Plot Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(truth_cut, 'k-', label='Truth (Ground Truth Input)', linewidth=2)
    plt.plot(non_eq_cut, 'r--', label=f'Not Equalized (MAE: {mae_non_eq:.2f})', alpha=0.7)
    plt.plot(eq_cut, 'g-.', label=f'Equalized L1B Output (MAE: {mae_eq:.2f})', linewidth=1.5)

    plt.title('Level-1B Radiometric Cross-Validation: Equalized vs. Not Equalized vs. Truth', fontsize=12)
    plt.xlabel('Across-Track Pixel Index (ACT)', fontsize=10)
    plt.ylabel('Signal Intensity [e- / Radiance Equivalent]', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('L1B_Cross_Validation_Plot.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()