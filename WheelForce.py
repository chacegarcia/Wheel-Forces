import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER INPUTS
# -----------------------------
OD_in = 8.0
bore_in = 3.0
tooth_spacing_mm_low = 2.0
tooth_spacing_mm_high = 3.0
N_engaged_low = 7
N_engaged_high = 10

P_kW = 1.1
torque_percent = 0.06
base_rpm_1800 = 1800.0
base_rpm_3600 = 3600.0

default_tooth_temp_C = 150.0  # Placeholder thermal estimate
clamp_mpa = 35.0

# -----------------------------
# GEOMETRY
# -----------------------------
inch = 0.0254
ro = (OD_in/2) * inch
ri = (bore_in/2) * inch

rho = 7850.0
nu  = 0.30

# -----------------------------
# FUNCTIONS
# -----------------------------
def annular_disk_stress(r, ri, ro, rho, nu, rpm):
    omega = 2*np.pi*(rpm/60.0)
    C = (3+nu)/8.0 * rho * omega**2
    D = (1+3*nu)/8.0 * rho * omega**2
    A = C*(ri**2 + ro**2)
    B = C*(ri**2)*(ro**2)
    sigma_r = A - B/(r**2) - C*(r**2)
    sigma_t = A + B/(r**2) - D*(r**2)
    return sigma_r, sigma_t

def von_mises(sig_r, sig_t):
    return np.sqrt(sig_r**2 - sig_r*sig_t + sig_t**2)

# -----------------------------
# RPM RANGE
# -----------------------------
rpm = np.linspace(0, 10000, 2001)
r = np.linspace(ri*1.001, ro, 1600)

vm_max = []
for n in rpm:
    if n == 0:
        vm_max.append(0)
        continue
    sr, st = annular_disk_stress(r, ri, ro, rho, nu, n)
    vm = von_mises(sr, st)/1e6
    vm_max.append(np.max(vm))
vm_max = np.array(vm_max)
combined_vm = vm_max + clamp_mpa

# -----------------------------
# MATERIAL DATA
# -----------------------------
materials = {
    "PLA Basic": {"strength":35.0, "color":"green",  "crit":60.0},
    "ABS":       {"strength":33.0, "color":"orange", "crit":94.0},
    "ASA":       {"strength":37.0, "color":"purple", "crit":100.0},
    "PETG HF":   {"strength":34.0, "color":"brown",  "crit":70.0},
    "TPU 95A":   {"strength":29.6, "color":"cyan",   "crit":150.0},
    "PA6-CF":    {"strength":102.0,"color":"red",    "crit":68.0},
}

# -----------------------------
# TOOTH FORCE (constant torque%)
# -----------------------------
T_rated_1800 = 9550 * P_kW / base_rpm_1800
T_rated_3600 = 9550 * P_kW / base_rpm_3600
T_load_1800  = torque_percent * T_rated_1800
T_load_3600  = torque_percent * T_rated_3600

F_t_1800 = T_load_1800 / ro
F_t_3600 = T_load_3600 / ro

F_tooth_1800_low  = F_t_1800 / N_engaged_high
F_tooth_1800_high = F_t_1800 / N_engaged_low
F_tooth_3600_low  = F_t_3600 / N_engaged_high
F_tooth_3600_high = F_t_3600 / N_engaged_low

# -----------------------------
# IMPACT RATE
# -----------------------------
circumference_mm = 2*np.pi*(ro*1000.0)
teeth_per_rev_low  = circumference_mm / tooth_spacing_mm_high
teeth_per_rev_high = circumference_mm / tooth_spacing_mm_low

impacts_low  = rpm/60.0 * teeth_per_rev_low
impacts_high = rpm/60.0 * teeth_per_rev_high

# -----------------------------
# PLOTTING
# -----------------------------
fig, ax = plt.subplots(figsize=(13,8))

# Stress
ax.plot(rpm, vm_max, linewidth=3, label="Centrifugal VM")
ax.plot(rpm, combined_vm, linewidth=3, label="Combined VM")
ax.axhline(250, linestyle=":", color="black", label="Steel Yield")

for name, d in materials.items():
    ax.axhline(d["strength"], linestyle="--",
               color=d["color"], label=f"{name} strength")

ax.set_xlim(0,10000)
ax.set_ylim(0,300)
ax.set_xlabel("RPM")
ax.set_ylabel("Stress (MPa)")
ax.grid(True)

# Tooth Force axis
ax2 = ax.twinx()
ax2.plot(rpm, np.full_like(rpm, F_tooth_1800_high),
         label="Tooth Force 1800 (N=7)")
ax2.plot(rpm, np.full_like(rpm, F_tooth_3600_high),
         label="Tooth Force 3600 (N=7)")
ax2.set_ylabel("Force per Tooth (N)")

# Impact rate axis
ax3 = ax.twinx()
ax3.spines["right"].set_position(("axes",1.1))
ax3.plot(rpm, impacts_high, linestyle="-.",
         label="Impacts/sec (2mm spacing)")
ax3.set_ylabel("Impacts/sec")

# Temperature axis
ax4 = ax.twinx()
ax4.spines["right"].set_position(("axes",1.2))
ax4.plot(rpm, np.full_like(rpm, default_tooth_temp_C),
         label="Tooth Temp (150°C)")
for name, d in materials.items():
    ax4.axhline(d["crit"], linestyle=":",
                color=d["color"],
                label=f"{name} thermal limit")
ax4.set_ylabel("Temperature (°C)")

# Combined legend (FIXED ncol)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()

ax.legend(lines1+lines2+lines3+lines4,
          labels1+labels2+labels3+labels4,
          loc="upper left",
          fontsize=7,
          framealpha=0.9,
          ncol=2)

plt.title("Wheel Stress + Tooth Force + Impact Rate + Thermal Wear Model")
plt.tight_layout()
plt.show()