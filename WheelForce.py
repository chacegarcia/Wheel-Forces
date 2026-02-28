import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Given info (from user)
# -----------------------------
OD_in = 8.0
ro_in = OD_in/2
bore_in = 3.0
ri_in = bore_in/2

tooth_spacing_mm_low = 2.0
tooth_spacing_mm_high = 3.0  # spacing along surface (incl spiral-to-spiral)
N_engaged_low = 7
N_engaged_high = 10

P_kW = 1.1
torque_percent = 0.06
base_rpm_1800 = 1800.0
base_rpm_3600 = 3600.0

default_tooth_temp_C = 150.0  # user requested default frictional heating temp

# -----------------------------
# Conversions / constants
# -----------------------------
inch = 0.0254
ro = ro_in * inch
ri = ri_in * inch

# Steel properties for centrifugal stress model
rho = 7850.0
nu  = 0.30

def annular_disk_stress(r, ri, ro, rho, nu, rpm):
    omega = 2*np.pi*(rpm/60.0)
    C = (3+nu)/8.0 * rho * omega**2
    D = (1+3*nu)/8.0 * rho * omega**2
    A = C*(ri**2 + ro**2)
    B = C*(ri**2)*(ro**2)
    sigma_r = A - B/(r**2) - C*(r**2)
    sigma_t = A + B/(r**2) - D*(r**2)
    return sigma_r, sigma_t

def von_mises_plane_stress(sig_r, sig_t):
    return np.sqrt(sig_r**2 - sig_r*sig_t + sig_t**2)

def first_crossing_rpm(x, y, target):
    if y[0] >= target:
        return float(x[0])
    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    x0, x1 = x[i-1], x[i]
    y0, y1 = y[i-1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

# -----------------------------
# RPM sweep
# -----------------------------
rpm = np.linspace(0, 10000, 2001)

# Centrifugal stress (max across wheel) - analytic annular disk
r = np.linspace(ri*1.001, ro, 1600)
vm_max = np.zeros_like(rpm)
for i, n in enumerate(rpm):
    if n == 0:
        vm_max[i] = 0.0
        continue
    sr, st = annular_disk_stress(r, ri, ro, rho, nu, n)
    vm = von_mises_plane_stress(sr, st) / 1e6
    vm_max[i] = float(np.max(vm))

clamp_mpa = 35.0
combined_max = vm_max + clamp_mpa

# -----------------------------
# Plastic strength + thermal "critical" points (from Bambu TDS / store)
# strength MPa, critical_temp_C (use Vicat/HDT where available; else Tg; else melt), melt_C
# -----------------------------
materials = {
    "PLA Basic": {"strength":35.0, "color":"green",  "crit":60.0,  "melt":160.0, "crit_label":"Tg 60°C"},
    "ABS":       {"strength":33.0, "color":"orange", "crit":94.0,  "melt":200.0, "crit_label":"Vicat 94°C"},
    "ASA":       {"strength":37.0, "color":"purple", "crit":100.0, "melt":210.0, "crit_label":"HDT 100°C"},
    "PETG HF":   {"strength":34.0, "color":"brown",  "crit":70.0,  "melt":214.0, "crit_label":"Vicat 70°C"},
    "TPU 95A":   {"strength":29.6, "color":"cyan",   "crit":150.0, "melt":183.0, "crit_label":"(no Vicat/HDT)"},
    "PA6-CF":    {"strength":102.0,"color":"red",    "crit":68.0,  "melt":223.0, "crit_label":"Tg 68°C"},
}

rpm_at_strength = {name: first_crossing_rpm(rpm, vm_max, d["strength"])
                   for name, d in materials.items()}

# -----------------------------
# Tooth force (constant torque% assumption)
# -----------------------------
T_rated_1800 = 9550 * P_kW / base_rpm_1800
T_rated_3600 = 9550 * P_kW / base_rpm_3600
T_load_1800  = torque_percent * T_rated_1800
T_load_3600  = torque_percent * T_rated_3600

F_t_1800 = T_load_1800 / ro
F_t_3600 = T_load_3600 / ro

F_tooth_1800_lowN  = F_t_1800 / N_engaged_high
F_tooth_1800_highN = F_t_1800 / N_engaged_low
F_tooth_3600_lowN  = F_t_3600 / N_engaged_high
F_tooth_3600_highN = F_t_3600 / N_engaged_low

# -----------------------------
# Impact rate
# Teeth per rev from surface spacing range (2-3 mm along surface)
circumference_mm = 2*np.pi*(ro*1000.0)
teeth_per_rev_low  = circumference_mm / tooth_spacing_mm_high  # 3mm -> fewer teeth
teeth_per_rev_high = circumference_mm / tooth_spacing_mm_low   # 2mm -> more teeth

impacts_per_sec_low  = rpm/60.0 * teeth_per_rev_low
impacts_per_sec_high = rpm/60.0 * teeth_per_rev_high

# -----------------------------
# Thermal wear visualization
# We'll show a horizontal temp line at 150°C and horizontal "critical temp" lines per material.
# -----------------------------
plt.figure(figsize=(13.6, 7.8))
ax = plt.gca()

# Stress axis
ax.plot(rpm, vm_max, linewidth=3, label='Centrifugal Von Mises (max across wheel)')
ax.plot(rpm, combined_max, linewidth=3, label='Combined (Clamp + Centrifugal VM)')
ax.axhline(250, linestyle=':', linewidth=2, color='black', label='Steel Yield (~250 MPa)')

# Strength lines + failure markers (centrifugal-only)
for name, d in materials.items():
    strength = d["strength"]; color = d["color"]
    ax.axhline(strength, linestyle='--', linewidth=2, color=color, label=f'{name} strength: {strength:.1f} MPa')
    rc = rpm_at_strength[name]
    if rc is not None:
        ax.axvline(rc, linestyle=':', linewidth=1.8, color=color, alpha=0.85)
        ax.text(rc + 55, 255, f'{name} ≈ {int(round(rc))} rpm',
                rotation=90, color=color, fontsize=8, va='top')

ax.set_title('Wheel Stress vs RPM + Tooth Force + Impact Rate + Thermal Wear (default 150°C)', fontsize=14)
ax.set_xlabel('RPM')
ax.set_ylabel('Stress (MPa)')
ax.set_xlim(0, 10000)
ax.set_ylim(0, 300)
ax.grid(True)

# Tooth force axis
ax2 = ax.twinx()
ax2.plot(rpm, np.full_like(rpm, F_tooth_1800_highN), linewidth=2,
         label=f'Force/tooth @OD (1800 base, N=7) ≈ {F_tooth_1800_highN:.2f} N')
ax2.plot(rpm, np.full_like(rpm, F_tooth_1800_lowN), linewidth=2,
         label=f'Force/tooth @OD (1800 base, N=10) ≈ {F_tooth_1800_lowN:.2f} N')
ax2.fill_between(rpm, F_tooth_1800_lowN, F_tooth_1800_highN, alpha=0.10, label='1800 base: tooth force band (N=7–10)')

ax2.plot(rpm, np.full_like(rpm, F_tooth_3600_highN), linewidth=2,
         label=f'Force/tooth @OD (3600 base, N=7) ≈ {F_tooth_3600_highN:.2f} N')
ax2.plot(rpm, np.full_like(rpm, F_tooth_3600_lowN), linewidth=2,
         label=f'Force/tooth @OD (3600 base, N=10) ≈ {F_tooth_3600_lowN:.2f} N')
ax2.fill_between(rpm, F_tooth_3600_lowN, F_tooth_3600_highN, alpha=0.10, label='3600 base: tooth force band (N=7–10)')
ax2.set_ylabel('Force per tooth (N)')
ax2.set_ylim(0, max(F_tooth_1800_highN, F_tooth_3600_highN) * 6 + 0.2)

# Impacts/sec axis (offset)
ax3 = ax.twinx()
ax3.spines['right'].set_position(('axes', 1.10))
ax3.plot(rpm, impacts_per_sec_low, linestyle='-.', linewidth=2, alpha=0.9,
         label=f'Impact rate (3mm spacing): strikes/sec')
ax3.plot(rpm, impacts_per_sec_high, linestyle='-.', linewidth=2, alpha=0.9,
         label=f'Impact rate (2mm spacing): strikes/sec')
ax3.set_ylabel('Strikes per second (at contact point)')
ax3.set_ylim(0, np.max(impacts_per_sec_high) * 1.05)

# Temperature axis (second offset)
ax4 = ax.twinx()
ax4.spines['right'].set_position(('axes', 1.20))

# Default tooth temperature line across RPM
ax4.plot(rpm, np.full_like(rpm, default_tooth_temp_C), linewidth=2,
         label=f'Default tooth-band temp: {default_tooth_temp_C:.0f}°C')

# Material critical temperature lines
for name, d in materials.items():
    ax4.axhline(d["crit"], linestyle=':', linewidth=2, color=d["color"],
                alpha=0.9, label=f'{name} thermal limit ({d["crit_label"]}) ≈ {d["crit"]:.0f}°C')

ax4.set_ylabel('Temperature (°C)')
ax4.set_ylim(0, max(260, default_tooth_temp_C*1.25))

# Info box
over_temp = [name for name,d in materials.items() if default_tooth_temp_C >= d["crit"]]
info_box = (
    f"Wheel: OD=8\" (ro=4\"), bore=3\" (ri=1.5\")\n"
    f"Motor: {P_kW:.1f} kW | torque% ≈ {torque_percent*100:.0f}% (assumed constant)\n"
    f"Engaged teeth: N=7–10\n"
    f"Surface tooth spacing: ~2–3 mm → teeth/rev ≈ {teeth_per_rev_low:.0f}–{teeth_per_rev_high:.0f}\n"
    f"Thermal wear model: tooth temp fixed at {default_tooth_temp_C:.0f}°C (placeholder)\n"
    f"At {default_tooth_temp_C:.0f}°C exceeds thermal limit for: " + (", ".join(over_temp) if over_temp else "none")
)
ax.text(0.98, 0.98, info_box,
        transform=ax.transAxes,
        fontsize=7.7,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', alpha=0.9))

# Legend (big, but compact)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax.legend(lines1 + lines2 + lines3 + lines4,
          labels1 + labels2 + labels3 + labels4,
          loc='upper left', fontsize=6.7, framealpha=0.95, ncols=2)

plt.tight_layout()
outpath = "/mnt/data/wheel_stress_vs_rpm_with_thermal_wear_overlay_150C.png"
plt.savefig(outpath, dpi=200)
outpath
