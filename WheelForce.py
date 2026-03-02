import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ===== USER-EDITABLE INPUT SECTION =======
# ==========================================
# Choose a material key from MATERIALS below, or use:
#   
# ==========================================
# Material selection system
# ==========================================
# Use:
#   SELECTED_MATERIALS = "ALL"
#   SELECTED_MATERIALS = "1"
#   SELECTED_MATERIALS = "1,5,8"
#   SELECTED_MATERIALS = [1,5,8]
#SELECTED_MATERIALS = "11,7,6,2"


# ==========================================
# Material selection system
# ==========================================
# Use:
SELECTED_MATERIALS = "ALL"
#   SELECTED_MATERIALS = "1"
#   SELECTED_MATERIALS = "1,5,8"
#   SELECTED_MATERIALS = [1,5,8]
#SELECTED_MATERIALS = "11,7,6,2,1,4"
   # e.g. "PLA Basic", "ABS", "316L AM", "Maraging MS1 (EOS)"

# Wheel geometry
OD_in = 8.0
bore_in = 3.0

# RPM sweep
RPM_MAX = 10000
RPM_SAMPLES = 2001

# ------------------ Clamp model (single bolt + flat donut ring) ------------------
CLAMP_TORQUE_FTLB = 20.0       # <-- edit (ft-lb)
BOLT_DIAMETER_MM = 8.0         # 13 mm hex head is typically M8
NUT_FACTOR_K = 0.20            # ~0.20 dry, ~0.15 lubricated

# Flat donut ring geometry (face contact)
BORE_ID_IN = 3.0               # wheel bore ID
DONUT_RADIAL_WIDTH_IN = 0.25   # ring contact radial width (ID->OD)/2
CAP_OD_IN = BORE_ID_IN + 2.0 * DONUT_RADIAL_WIDTH_IN  # computed OD

# ------------------ Thermal model (speed-linked) ------------------
THERMAL_MODEL = "power_law"     # "power_law" or "constant"
T_amb_C = 25.0
T_ref_C = 50                # placeholder at RPM_ref
RPM_ref = 3740
n_exp = 1.0

# If THERMAL_MODEL == "constant", use:
default_tooth_temp_C = 150.0

# ------------------ Flash temperature proxy (polymer tribology) ------------------
# Many polymer wear studies distinguish bulk (steady) temperature from flash temperature
# spikes at micro-contacts. Without detailed contact mechanics, we use a tunable proxy:
#   T_flash(RPM) = T_flash_ref * (v/v_ref)^m
# and combine:
#   T_effective = T_bulk + T_flash
# This keeps behavior realistic: higher surface speed -> higher flash spikes.
FLASH_MODEL = "power_law"       # "off" or "power_law"
T_flash_ref_C = 20            # flash spike at v_ref (Â°C) â tune later with IR/thermocouple
FLASH_V_EXP = 0.5               # exponent vs surface speed (0.5 is common proxy)


# ------------------ Wear / degradation proxy (per cut) ------------------
CUT_TIME_S = 30              # foam is being cut ~15 seconds per cycle
WEAR_ALPHA = 0.06              # ramp per Â°C above limit (tune later)
WEAR_BETA = 1.7                # exponent on thermal factor (tune later)

# Tg softening factor (wear/creep acceleration near Tg)
TG_DELTA_C = 10.0              # logistic transition width around Tg
TG_GAMMA = 3.0                 # multiplier strength (higher => bigger Tg effect)

# ==========================================

# Units/geometry
inch = 0.0254
ro = (OD_in / 2.0) * inch
ri = (bore_in / 2.0) * inch

# ---- Compute clamp_mpa from clamp torque + donut contact area ----
ftlb_to_Nm = 1.35582
T_Nm = CLAMP_TORQUE_FTLB * ftlb_to_Nm
D_bolt_m = BOLT_DIAMETER_MM / 1000.0
F_clamp = T_Nm / (NUT_FACTOR_K * D_bolt_m)  # T = K*F*D

D_cap_m  = CAP_OD_IN * inch
D_bore_m = BORE_ID_IN * inch
A_contact = (np.pi / 4.0) * (D_cap_m**2 - D_bore_m**2)   # m^2
clamp_mpa = (F_clamp / A_contact) / 1e6                  # MPa (average)

print(f"[Clamp] Cap OD={CAP_OD_IN:.2f} in (ID={BORE_ID_IN:.2f} in, radial width={DONUT_RADIAL_WIDTH_IN:.2f} in)")
print(f"[Clamp] Torque={CLAMP_TORQUE_FTLB:.1f} ft-lb ({T_Nm:.1f} NÂ·m), bolt={BOLT_DIAMETER_MM:.0f} mm, K={NUT_FACTOR_K:.2f}")
print(f"[Clamp] Clamp force â {F_clamp:,.0f} N, contact area â {A_contact*1e6:,.0f} mm^2")
print(f"[Clamp] Avg clamp pressure â {clamp_mpa:.2f} MPa  (used as clamp_mpa)")

# -----------------------------
# Centrifugal stress model for annular disk (free-free)
# -----------------------------
def annular_disk_stress(r, ri, ro, rho, nu, rpm):
    omega = 2.0 * np.pi * (rpm / 60.0)
    C = (3 + nu) / 8.0 * rho * omega**2
    D = (1 + 3 * nu) / 8.0 * rho * omega**2
    A = C * (ri**2 + ro**2)
    B = C * (ri**2) * (ro**2)
    sigma_r = A - B / (r**2) - C * (r**2)
    sigma_t = A + B / (r**2) - D * (r**2)
    return sigma_r, sigma_t

def von_mises(sig_r, sig_t):
    return np.sqrt(sig_r**2 - sig_r * sig_t + sig_t**2)

def first_crossing_x(x, y, target):
    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(x[0])
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

def tooth_temp_curve(rpm, ro_m):
    """
    Return (T_bulk, T_effective) in Â°C.
    T_bulk is the steady/running temperature proxy vs RPM.
    T_effective adds a flash-temperature proxy that scales with surface speed.
    """
    if THERMAL_MODEL.lower() == "constant":
        T_bulk = np.full_like(rpm, default_tooth_temp_C, dtype=float)
    else:
        rr = RPM_ref if RPM_ref > 1e-9 else 1.0
        rise = (T_ref_C - T_amb_C) * (np.maximum(rpm, 0.0) / rr) ** n_exp
        T_bulk = T_amb_C + rise

    # Flash temperature proxy based on surface speed v = omega*R
    if str(FLASH_MODEL).lower() == "off":
        return T_bulk, T_bulk

    omega = 2.0 * np.pi * (rpm / 60.0)
    v = omega * ro_m  # m/s
    omega_ref = 2.0 * np.pi * (RPM_ref / 60.0) if RPM_ref > 1e-9 else 1.0
    v_ref = omega_ref * ro_m
    v_ref = v_ref if v_ref > 1e-9 else 1.0

    T_flash = T_flash_ref_C * (np.maximum(v, 0.0) / v_ref) ** FLASH_V_EXP
    T_eff = T_bulk + T_flash
    return T_bulk, T_eff

# -----------------------------
# MATERIAL DATABASE
# -----------------------------
MATERIALS = {
    "PLA Basic":    {"rho": 1240, "nu": 0.35, "strength_mpa": 35.0,   "hardness_proxy": 1.00, "tg_C": 60.0,  "crit_temp_C": 57.0,  "crit_label": "Vicat 57Â°C"},
    "ABS":          {"rho": 1050, "nu": 0.35, "strength_mpa": 33.0,   "hardness_proxy": 0.90, "tg_C": 105.0, "crit_temp_C": 87.0,  "crit_label": "HDT 87Â°C"},
    "ASA":          {"rho": 1050, "nu": 0.35, "strength_mpa": 37.0,   "hardness_proxy": 0.95, "tg_C": 100.0, "crit_temp_C": 100.0, "crit_label": "HDT 100Â°C"},
    "PETG HF":      {"rho": 1280, "nu": 0.38, "strength_mpa": 34.0,   "hardness_proxy": 0.85, "tg_C": 80.0,  "crit_temp_C": 69.0,  "crit_label": "HDT 69Â°C"},
    "TPU 95A HF":   {"rho": 1220, "nu": 0.45, "strength_mpa": 27.3,   "hardness_proxy": 0.35, "tg_C": -30.0, "crit_temp_C": 183.0, "crit_label": "Melt 183Â°C (proxy)"},
    "PA6-CF":       {"rho": 1090, "nu": 0.39, "strength_mpa": 102.0,  "hardness_proxy": 1.40, "tg_C": 68.0,  "crit_temp_C": 186.0, "crit_label": "HDT 186Â°C"},
    "PPS-CF (Fiberon)": {"rho": 1380, "nu": 0.36, "strength_mpa": 100.0, "hardness_proxy": 1.60, "tg_C": 90.0, "crit_temp_C": 250.0, "crit_label": "HDT ~250Â°C (proxy)"},

    "316L AM":            {"rho": 8000, "nu": 0.29, "strength_mpa": 609.0,  "hardness_proxy": 6.00, "tg_C": None, "crit_temp_C": 600.0, "crit_label": "metal (no softening @150Â°C)"},
    "17-4PH AM (EOS)":    {"rho": 7800, "nu": 0.29, "strength_mpa": 1358.0, "hardness_proxy": 8.00, "tg_C": None, "crit_temp_C": 600.0, "crit_label": "metal (no softening @150Â°C)"},
    "17-4PH AM (H900)":   {"rho": 7800, "nu": 0.29, "strength_mpa": 1250.0, "hardness_proxy": 8.50, "tg_C": None, "crit_temp_C": 600.0, "crit_label": "metal (no softening @150Â°C)"},
    "Maraging MS1 (EOS)": {"rho": 8100, "nu": 0.29, "strength_mpa": 2080.0, "hardness_proxy": 10.0, "tg_C": None, "crit_temp_C": 600.0, "crit_label": "metal (no softening @150Â°C)"},
}


# Create numeric index mapping
MATERIAL_KEYS = list(MATERIALS.keys())
MATERIAL_INDEX = {i+1: MATERIAL_KEYS[i] for i in range(len(MATERIAL_KEYS))}

def resolve_selected_materials(selection):
    if isinstance(selection, str):
        if selection.strip().upper() == "ALL":
            return MATERIAL_KEYS
        nums = [int(s.strip()) for s in selection.split(",") if s.strip().isdigit()]
    elif isinstance(selection, (list, tuple)):
        nums = [int(n) for n in selection]
    else:
        return MATERIAL_KEYS

    names = []
    for n in nums:
        if n in MATERIAL_INDEX:
            names.append(MATERIAL_INDEX[n])
    return names

SELECTED_NAMES = resolve_selected_materials(SELECTED_MATERIALS)


def compute_vm_curve(mat, rpm, r, ri, ro):
    vm = np.zeros_like(rpm, dtype=float)
    for i, n in enumerate(rpm):
        if n == 0:
            continue
        sr, st = annular_disk_stress(r, ri, ro, mat["rho"], mat["nu"], n)
        vm[i] = float(np.max(von_mises(sr, st) / 1e6))  # MPa
    return vm

def wear_thermal_factor(T, Tcrit, tg_C=None):
    """
    Wear/dulling acceleration factor combining:
      1) HDT/Vicat-like threshold (shape-loss proxy): ramps up when T > Tcrit
      2) Tg softening: smooth logistic ramp centered at Tg (if provided), to mimic
         rapid modulus drop / creep rise near Tg seen in polymer tribology.
    """
    # (1) Above-crit ramp (HDT/Vicat proxy)
    dT = np.maximum(T - Tcrit, 0.0)
    f_crit = (1.0 + WEAR_ALPHA * dT) ** WEAR_BETA

    # (2) Tg logistic ramp (only for polymers where tg_C is not None)
    if tg_C is None:
        return f_crit

    # Logistic: 0 below Tg, ~1 above Tg, smooth width TG_DELTA_C
    # f_tg ranges from 1 to (1 + TG_GAMMA)
    x = (T - float(tg_C)) / max(TG_DELTA_C, 1e-9)
    s = 1.0 / (1.0 + np.exp(-x))
    f_tg = 1.0 + TG_GAMMA * s
    return f_crit * f_tg

# -----------------------------
# COMPUTE
# -----------------------------
rpm = np.linspace(0, RPM_MAX, RPM_SAMPLES)
r = np.linspace(ri * 1.001, ro, 1600)
T_bulk_C, T_eff_C = tooth_temp_curve(rpm, ro)
# Use effective temperature for wear/softening checks
tempC = T_eff_C

omega = 2.0 * np.pi * (rpm / 60.0)
surface_speed = omega * ro
slide_dist_per_cut = surface_speed * CUT_TIME_S  # meters per cut

# -----------------------------
# FIGURE 1: Stress + crossovers
# -----------------------------
plt.figure(figsize=(12.6, 8.0))
ax = plt.gca()

def plot_stress(name, mat, y_text=290):
    vm = compute_vm_curve(mat, rpm, r, ri, ro)
    line, = ax.plot(rpm, vm, linewidth=2.2, label=f"{name} VM (Ï={mat['rho']/1000:.2f} g/cc)")
    c = line.get_color()
    ax.plot(rpm, vm + clamp_mpa, linestyle="--", linewidth=1.4, color=c, alpha=0.85, label=f"{name} VM + clamp")
    ax.axhline(mat["strength_mpa"], linestyle=":", linewidth=1.2, color=c, alpha=0.9, label=f"{name} tensile {mat['strength_mpa']:.1f} MPa")

    fail_rpm = first_crossing_x(rpm, vm, mat["strength_mpa"])
    if fail_rpm is not None:
        ax.axvline(fail_rpm, linestyle=":", linewidth=1.2, color=c, alpha=0.75)
        ax.text(fail_rpm + 40, y_text, f"{name}â{int(round(fail_rpm))}rpm", rotation=90, fontsize=7.2, color=c, va="top")

    return vm

if SELECTED_MATERIALS == "ALL":
    for name in SELECTED_NAMES:
        mat = MATERIALS[name]
        plot_stress(name, mat)
    ax.set_title("ALL Materials: Centrifugal VM vs RPM + Stress Crossover + Thermal Model")
else:
    if False:
        raise ValueError("Material not found. Options:\n" + "\n".join(MATERIALS.keys()))
    for name in SELECTED_NAMES:
        plot_stress(name, MATERIALS[name])
    ax.set_title(f"{SELECTED_MATERIALS}: Stress crossover + Thermal model vs RPM")

ax.set_xlabel("RPM")
ax.set_ylabel("Von Mises Stress (MPa)")
ax.set_xlim(0, RPM_MAX)
ax.set_ylim(0, 300)
ax.grid(True)

axT = ax.twinx()
axT.spines["right"].set_position(("axes", 1.10))
axT.plot(rpm, T_bulk_C, linewidth=2.0, label=f"Bulk temp ({THERMAL_MODEL})")
axT.plot(rpm, T_eff_C,  linewidth=2.2, label=f"Effective temp (bulk+flash)")
axT.set_ylabel("Tooth-band temperature (Â°C)")
axT.set_ylim(0, max(260, float(np.max(tempC)) * 1.1))

model_line = "Thermal model: T=T_amb+(T_ref-T_amb)*(RPM/RPM_ref)^n"
params_line = f"T_amb={T_amb_C:.0f}Â°C, T_ref={T_ref_C:.0f}Â°C@{RPM_ref:.0f}rpm, n={n_exp:.2f}"
clamp_line = f"Clamp: {CLAMP_TORQUE_FTLB:.1f} ft-lb -> {clamp_mpa:.2f} MPa (avg)"
flash_line = f"Flash: {FLASH_MODEL} (T_flash_ref={T_flash_ref_C:.0f}Â°C @ v_ref, m={FLASH_V_EXP:.2f})"
ax.text(0.98, 0.98, model_line + "\n" + params_line + "\n" + flash_line + "\n" + clamp_line,
        transform=ax.transAxes, fontsize=7.4, ha="right", va="top",
        bbox=dict(boxstyle="round", alpha=0.9))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = axT.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=6.3, framealpha=0.95, ncol=2)

plt.tight_layout()
plt.show()

# -----------------------------
# FIGURE 2: Wear / degradation proxy per 15s cut
# -----------------------------
plt.figure(figsize=(12.6, 7.2))
axw = plt.gca()

def plot_wear(name, mat):
    thermal_factor = wear_thermal_factor(tempC, mat["crit_temp_C"], mat.get("tg_C", None))
    # Relative wear index per cut (arb units)
    hard = float(mat.get("hardness_proxy", 1.0))
    wear_index = (slide_dist_per_cut / np.maximum(hard, 1e-9)) * thermal_factor
    axw.plot(rpm, wear_index, linewidth=2.3, label=f"{name} wear index/cut")

if SELECTED_MATERIALS == "ALL":
    for name in SELECTED_NAMES:
        mat = MATERIALS[name]
        plot_wear(name, mat)
    axw.set_title(f"Relative wear/dulling risk per cut ({CUT_TIME_S:.0f} s) â speed + thermal acceleration")
else:
    for name in SELECTED_NAMES:
        plot_wear(name, MATERIALS[name])
    axw.set_title(f"{SELECTED_MATERIALS}: Relative wear/dulling risk per cut ({CUT_TIME_S:.0f} s)")

axw.set_xlabel("RPM")
axw.set_ylabel("Wear index per cut (arb. units)")
axw.set_xlim(0, RPM_MAX)
axw.grid(True)

wear_note = (
    f"Per-cut time = {CUT_TIME_S:.0f}s\n"
    f"Wear model: (sliding distance / hardness_proxy) * thermal_factor\n"
    f"thermal_factor = (1 + Î±Â·max(T-Tcrit,0))^Î² Â· (1 + Î³Â·logistic((T-Tg)/Î))\n"
    f"Î±={WEAR_ALPHA:.3f}, Î²={WEAR_BETA:.2f}\n"
    f"Temp model: bulk={THERMAL_MODEL}, flash={FLASH_MODEL} (T_ref={T_ref_C:.0f}Â°C@{RPM_ref:.0f}rpm, n={n_exp:.2f}; T_flash_ref={T_flash_ref_C:.0f}Â°C, m={FLASH_V_EXP:.2f})"
)
axw.text(0.98, 0.98, wear_note, transform=axw.transAxes, fontsize=8.2,
         ha="right", va="top", bbox=dict(boxstyle="round", alpha=0.9))

axw.legend(loc="upper left", fontsize=7.2, framealpha=0.95, ncol=2)

plt.tight_layout()
plt.show()

# -----------------------------
# Calibrate later (recommended)
# -----------------------------
# Measure mass loss (g) or tooth height loss (mm) after a known total cutting time at a known RPM.
# Fit one scalar coefficient so wear_index maps to g/cut or mm/cut.


print("\nMaterial Index Map:")
for idx, name in MATERIAL_INDEX.items():
    print(f"{idx}: {name}")