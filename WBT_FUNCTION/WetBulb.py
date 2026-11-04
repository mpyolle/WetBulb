# [Twb, Teq, epott, n_iter] = WetBulb(TemperatureC, Pressure, Humidity, [HumidityMode])
#
# Calculate wet-bulb temperature, equivalent temperature, and equivalent
# potential temperature from temperature, pressure, and relative or
# specific humidity.
#
#
# -------------------------------------------------------------------------
# Wet-Bulb Temperature, Equivalent Temperature, and Equivalent Potential
# Temperature Calculation Module
# -------------------------------------------------------------------------
# LINEAGE & CONTRIBUTORS:
# - Ported from HumanIndexMod (04-08-16) by Jonathan R. Buzan
# - MATLAB port by Robert Kopp (Last updated: June 08, 2016)
# - Ported to Python by Xianxiang Li (February 2, 2019)
# - Bug fix (iteration limit optimization) by Manuel Theurl (March 24, 2022)
# - Python Vectorized Optimizations & Enhancements by Marc Pyolle (2026)
#
# DOI : https://doi.org/10.5281/zenodo.23957860
#
# METHODOLOGY:
# Calculates Wet Bulb Temperature, Theta_wb, Theta_e, Moist Pot Temp, 
# Lifting Cond Temp, and Equiv Temp using Davies-Jones 2008 Method.
#
# Reference:  
#   - Bolton: The computation of equivalent potential temperature. Monthly Weather Review (1980)
#   - Davies-Jones: An efficient and accurate method for computing the wet-bulb temperature along pseudoadiabats. Monthly Weather Review (2008)
#   - Flatau et al: Polynomial fits to saturation vapor pressure. Journal of Applied Meteorology (1992)
#   - Sonntag: Important new values of the physical constants of 1986, vapor pressure formulations based on the ITS-90 and psychrometric formulae. Zeitschrift für Meteorologie (1990)
#
# --- Python Vectorized Optimizations by Marc Pyolle (2026) ---
# 1. Type Safety & Precision Shield:
#    - Enforced strict float64 casting across all input vectors (`TemperatureC`, 
#      `Pressure`, `Humidity`) to avoid hidden numerical downcasting and 
#      guarantee floating-point stability during intensive array calculations.
# 2. Physics-Based Domain Clamping:
#    - Replaced the original arbitrary damping step (`0.1 * delta`) in the slow 
#      loop with a hard physical constraint boundary (`np.minimum(Twb, T - C)`). 
#      Forces the solver to stay strictly within valid thermodynamic limits, 
#      preventing execution explosions and unphysical results.
# 3. Two-Tiered Vectorized Architecture Alignment:
#    - Realigned the logic of both solver stages. Fast convergence runs a global, 
#      fully-vectorized 3-iteration sweep. Slow convergence mode isolates the 
#      sub-domain of non-converged points via `subdo` indexing, preventing redundant 
#      recomputations over the entire grid.
# 4. Homogeneous Numerical Noise Barrier:
#    - Corrected the convergence threshold discrepancy: unified the limit to a 
#      strict 0.01°C for both the fast loop and sub-domain tracking. Eliminates 
#      infinite sub-epsilon oscillations and machine round-off errors ("staircase 
#      effect") against the physical saturation wall.
# -------------------------------------------------------------------------

import numpy as np

SHR_CONST_TKFRZ = 273.15

def calc_RH_from_T_Td(T, Td, mode=1):
    """
    Calculate relative humidity from dry-bulb temperature T (in degC) and dew-point temperature Td (in degC).
    
    Inputs:
        T: dry-bulb temperature (in degC)
        Td: dew-point temperature (in degC)
        mode: the mode to calculate RH
            mode = 0: more accurate equations (Sonntag 1994)
            mode = 1 (default): simpler Wexler formulae eq 10 from Bolton
    Outputs:
        RH: relative humidity in %, range 0-100.
        e: vapor pressure
        es: saturation vapor pressure
    """
    T = np.array(T, dtype=np.float64)
    Td = np.array(Td, dtype=np.float64)
    
    if mode == 0: 
        Tk = T + SHR_CONST_TKFRZ
        Tdk = Td + SHR_CONST_TKFRZ
        e = np.exp(-6096.9385 * Tdk**(-1) + 16.635794 - 2.711193e-2 * Tdk + 
                   1.673952e-5 * Tdk**2.0 + 2.433502 * np.log(Tdk))
        es = np.exp(-6096.9385 * Tk**(-1) + 16.635794 - 2.711193e-2 * Tk + 
                    1.673952e-5 * Tk**2.0 + 2.433502 * np.log(Tk))
    elif mode == 1: 
        e = 611.2 * np.exp((17.67 * Td) / (243.5 + Td)) # vapor pressure in Pa
        es = 611.2 * np.exp((17.67 * T) / (243.5 + T))   # vapor pressure in Pa

    RH = e / es * 100.0
    RH[RH > 100] = 100.0
    RH[RH < 0] = 0.0

    return RH, e, es


def QSat_2(T_k, p_t):
    """
    Computes saturation mixing ratio and the change in saturation mixing ratio with respect to temperature.
    Uses Bolton eqn 10, 39 and Davies-Jones eqns 2.3, A.1-A.10.

    Inputs:
        T_k: temperature (K)
        p_t: surface atmospheric pressure (Pa)

    Outputs:
        es_mb, rs, de_mbdT, dlnes_mbdT, rsdT, foftk, fdT
    """
    lambd_a = 3.504       # Inverse of Heat Capacity
    alpha = 17.67         # Constant to calculate vapour pressure
    beta = 243.5          # Constant to calculate vapour pressure
    epsilon = 0.6220      # Conversion between pressure/mixing ratio
    es_C = 6.112          # Vapour Pressure at Freezing STD (mb)
    vkp = 0.2854          # Heat Capacity
    y0 = 3036             # Constant
    y1 = 1.78             # Constant
    y2 = 0.448            # Constant
    Cf = SHR_CONST_TKFRZ  # Freezing Temp (K)
    refpres = 1000        # Reference Pressure (mb)

    p_tmb = p_t * 0.01
    tcfbdiff = T_k - Cf + beta
    es_mb = es_C * np.exp(alpha * (T_k - Cf) / tcfbdiff)
    dlnes_mbdT = alpha * beta / (tcfbdiff * tcfbdiff)
    pminuse = p_tmb - es_mb
    de_mbdT = es_mb * dlnes_mbdT
    d2e_mbdT2 = dlnes_mbdT * (de_mbdT - 2 * es_mb / tcfbdiff)

    ndimpress = (p_tmb / refpres)**vkp
    p0ndplam = refpres * ndimpress**lambd_a
    rs = epsilon * es_mb / (p0ndplam - es_mb + np.spacing(1))
    prersdt = epsilon * p_tmb / (pminuse * pminuse)
    rsdT = prersdt * de_mbdT
    d2rsdT2 = prersdt * (d2e_mbdT2 - de_mbdT * de_mbdT * (2 / pminuse))

    rsy2rs2 = rs + y2 * rs * rs
    oty2rs = 1 + 2.0 * y2 * rs
    y0tky1 = y0 / T_k - y1
    goftk = y0tky1 * (rs + y2 * rs * rs)
    gdT = -y0 * rsy2rs2 / (T_k * T_k) + y0tky1 * oty2rs * rsdT
    d2gdT2 = 2.0 * y0 * rsy2rs2 / (T_k * T_k * T_k) - 2.0 * y0 * rsy2rs2 * oty2rs * rsdT + \
             y0tky1 * 2.0 * y2 * rsdT * rsdT + y0tky1 * oty2rs * d2rsdT2

    foftk = ((Cf / T_k)**lambd_a) * (np.abs(1 - es_mb / p0ndplam))**(vkp * lambd_a) * \
            np.exp(-lambd_a * goftk)
    fdT = -lambd_a * (1.0 / T_k + vkp * de_mbdT / pminuse + gdT)
    d2fdT2 = lambd_a * (1.0 / (T_k * T_k) - vkp * de_mbdT * de_mbdT / (pminuse * pminuse) - \
                        vkp * d2e_mbdT2 / pminuse - d2gdT2)

    rs[rs > 1] = np.nan
    rs[rs < 0] = np.nan

    return es_mb, rs, de_mbdT, dlnes_mbdT, rsdT, foftk, fdT


def WetBulb(TemperatureC, Pressure, Humidity, HumidityMode=1):
    """ 
    INPUTS:
        TemperatureC: 2-m air temperature (degrees Celsius)
        Pressure: Atmospheric Pressure (Pa)
        Humidity: Humidity -- meaning depends on HumidityMode
        HumidityMode:
            0 (Default): specific humidity (kg/kg)
            1: relative humidity (%, max = 100)

    OUTPUTS:
        Twb: wet bulb temperature (C)
        Teq: Equivalent Temperature (K)
        epott: Equivalent Potential Temperature (K)
        n_iter: Number of iterations per point
    """
    TemperatureC = np.array(TemperatureC, dtype=np.float64)
    Pressure = np.array(Pressure, dtype=np.float64)
    Humidity = np.array(Humidity, dtype=np.float64)
    
    TemperatureK = TemperatureC + SHR_CONST_TKFRZ

    constA = 2675      
    grms = 1000        
    p0 = 1000          
    kappad = 0.2854    
    C = SHR_CONST_TKFRZ 
    pmb = Pressure * 0.01 
    T1 = TemperatureK

    es_mb, rs = QSat_2(TemperatureK, Pressure)[0:2]

    if HumidityMode == 0:
        qin = Humidity
        relhum = 100.0 * qin / rs
        vapemb = es_mb * relhum * 0.01
    elif HumidityMode == 1:
        relhum = Humidity
        qin = rs * relhum * 0.01
        vapemb = es_mb * relhum * 0.01

    mixr = qin * grms

    pnd = (pmb / p0)**kappad
    D = 1.0 / (0.1859 * pmb / p0 + 0.6512)
    k1 = -38.5 * pnd * pnd + 137.81 * pnd - 53.737
    k2 = -4.392 * pnd * pnd + 56.831 * pnd - 0.384

    tl = (1.0 / ((1.0 / (T1 - 55)) - (np.log(relhum / 100.0) / 2840.0))) + 55.0

    theta_dl = T1 * ((p0 / (pmb - vapemb))**kappad) * ((T1 / tl)**(mixr * 0.00028))
    epott = theta_dl * np.exp(((3.036 / tl) - 0.00178) * mixr * (1 + 0.000448 * mixr))
    Teq = epott * pnd
    X = (C / Teq)**3.504

    invalid = (Teq > 600) + (Teq < 100)
    hot = (Teq > 355.15)
    cold = ((X >= 1) * (X <= D))
    X[invalid == 1] = np.nan
    Teq[invalid == 1] = np.nan

    es_mb_teq, rs_teq, de_mbdTeq, dlnes_mbdTeq, rsdTeq, foftk_teq, fdTeq = QSat_2(Teq, Pressure)
    wb_temp = Teq - C - ((constA * rs_teq) / (1 + (constA * rs_teq * dlnes_mbdTeq)))
    sub = np.where(X <= D)
    wb_temp[sub] = (k1[sub] - 1.21 * cold[sub] - 1.45 * hot[sub] - (k2[sub] - 1.21 * cold[sub]) * X[sub] + (0.58 / X[sub]) * hot[sub])
    wb_temp[invalid == 1] = np.nan

    # --- Newton-Raphson fast ---
    maxiter = 3
    iter = 0
    delta = 1e6 * np.ones_like(wb_temp)

    n_iter = np.zeros_like(wb_temp, dtype=int)
    overshoot_mask = np.zeros_like(wb_temp, dtype=bool)

    while (np.max(np.abs(delta)) > 0.01) and (iter <= maxiter):
        es_mb_wb_temp, rs_wb_temp, de_mbdwb_temp, dlnes_mbdwb_temp, rsdwb_temp, foftk_wb_temp, fdwb_temp = QSat_2(wb_temp + C, Pressure)
        delta = (foftk_wb_temp - X) / fdwb_temp
        delta = np.where(delta < 10., delta, 10.)
        delta = np.where(delta > -10., delta, -10.)
        wb_temp = wb_temp - delta
        overshoot_mask |= (wb_temp > TemperatureK - C)
        wb_temp = np.minimum(wb_temp, TemperatureK - C)
        wb_temp[invalid == 1] = np.nan
        Twb = wb_temp
        iter = iter + 1
    
    n_iter[invalid != 1] = iter     

    # --- ConvergenceMode ---
    residu_max = np.nanmax(np.abs(delta))
    if residu_max > 0.01:
        convergence = 1e-2
        maxiter = 2000

        subdo = np.where(np.abs(delta) > convergence)

        iter_slow = 0
        while (len(subdo[0]) > 0) and (iter_slow < maxiter):
            iter_slow += 1
            n_iter[subdo] += 1

            es_mb_wb_temp, rs_wb_temp, de_mbdwb_temp, dlnes_mbdwb_temp, rsdwb_temp, foftk_wb_temp, fdwb_temp = QSat_2(
                wb_temp[subdo] + C, Pressure[subdo])
            delta_sub = (foftk_wb_temp - X[subdo]) / fdwb_temp
            delta_sub = np.where(delta_sub < 10., delta_sub, 10.)
            delta_sub = np.where(delta_sub > -10., delta_sub, -10.)
            wb_temp[subdo] -= delta_sub
            wb_temp[subdo] = np.minimum(wb_temp[subdo], TemperatureK[subdo] - C)
            wb_temp[invalid == 1] = np.nan
            delta[subdo] = delta_sub
            subdo = np.where(np.abs(delta) > convergence)

        Twb = wb_temp
        nb_fail = len(subdo[0])
        if nb_fail > 0:
            Twb[subdo] = np.nan

    return Twb, Teq, epott, n_iter


if __name__ == "__main__":
    tempC = np.array([31., 32., 33., 34.])
    tempd = np.array([26.57, 29.11, 29.26, 27.54])
    RH, e, es = calc_RH_from_T_Td(tempC, tempd, mode=1)
    print("RH = ", RH)

    Pres = np.array([101325] * 4)
    relHum = np.array([70., 80., 75., 60.])
    Hum_mode = 1

    Twb, Teq, epott, n_iter = WetBulb(tempC, Pres, relHum, Hum_mode)
    print("Twb = ", Twb)
