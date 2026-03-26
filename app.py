import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MDOF Building Dynamics Explorer", layout="wide")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
G = 9.81  # m/s^2


def ensure_positive(values: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    arr = np.array(values, dtype=float)
    arr[arr <= 0] = fallback
    return arr


# ---------------------------------------------------------
# Structural matrices
# ---------------------------------------------------------
def build_mass_matrix(weights_kN: np.ndarray) -> np.ndarray:
    masses = weights_kN / G
    return np.diag(masses)



def build_stiffness_matrix(story_stiffness_kN_per_m: np.ndarray) -> np.ndarray:
    n = len(story_stiffness_kN_per_m)
    K = np.zeros((n, n), dtype=float)

    for i in range(n):
        ki = story_stiffness_kN_per_m[i]
        kip1 = story_stiffness_kN_per_m[i + 1] if i + 1 < n else 0.0

        K[i, i] = ki + kip1
        if i < n - 1:
            K[i, i + 1] = -kip1
            K[i + 1, i] = -kip1

    return K



def solve_modes(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real_if_close(eigvals)
    eigvecs = np.real_if_close(eigvecs)

    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals[eigvals < 0] = 0.0
    omega = np.sqrt(eigvals)

    for i in range(eigvecs.shape[1]):
        mode = eigvecs[:, i]
        max_abs = np.max(np.abs(mode))
        if max_abs > 0:
            eigvecs[:, i] = mode / max_abs
        if eigvecs[-1, i] < 0:
            eigvecs[:, i] *= -1

    return omega, eigvecs



def modal_properties(M: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((M.shape[0], 1))
    gamma = []
    modal_mass = []
    eff_modal_mass = []

    for i in range(phi.shape[1]):
        mode = phi[:, [i]]
        num = (mode.T @ M @ ones).item()
        den = (mode.T @ M @ mode).item()

        g_i = num / den if abs(den) > 1e-12 else 0.0
        m_i = den
        m_eff_i = (num ** 2) / den if abs(den) > 1e-12 else 0.0

        gamma.append(g_i)
        modal_mass.append(m_i)
        eff_modal_mass.append(m_eff_i)

    return np.array(gamma, dtype=float), np.array(modal_mass, dtype=float), np.array(eff_modal_mass, dtype=float)



def compute_periods(omega: np.ndarray) -> np.ndarray:
    periods = np.zeros_like(omega)
    mask = omega > 0
    periods[mask] = 2 * np.pi / omega[mask]
    return periods


# ---------------------------------------------------------
# Seismic helpers
# ---------------------------------------------------------
def approximate_code_base_shear(total_weight_kN: float, T1: float, SDS: float, R: float, Ie: float) -> float:
    """
    Simplified classroom base shear estimate retained from the original app.
    The vertical ELF distribution below is improved, but this V calculation
    remains intentionally simplified unless the user manually overrides it.
    """
    T_use = max(T1, 0.05)
    Cs = SDS / max(R / Ie, 1e-9)
    period_factor = min(1.0, max(0.4, 1.0 / math.sqrt(T_use)))
    Cs_adj = min(max(Cs * period_factor, 0.01), 0.30)
    return Cs_adj * total_weight_kN



def vertical_distribution_exponent(T: float) -> float:
    if T <= 0.5:
        return 1.0
    if T >= 2.5:
        return 2.0
    return 1.0 + (T - 0.5) / 2.0



def compute_top_concentrated_force(T1: float, V_kN: float, include_top_force: bool) -> float:
    """
    Simple teaching implementation of the ELF top concentrated force Ft.
    Commonly taken as 0.07 T V, limited to 0.25 V, and omitted for shorter periods.
    """
    if not include_top_force or T1 <= 0.7 or V_kN <= 0:
        return 0.0
    return min(0.07 * T1 * V_kN, 0.25 * V_kN)



def distribute_lateral_forces(
    weights_kN: np.ndarray,
    heights_m: np.ndarray,
    V_kN: float,
    T1: float,
    include_top_force: bool = True,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    ELF vertical distribution closer to NSCP/ASCE format:
    Fx = Cvx * (V - Ft) + Ft at roof
    where Cvx = wx hx^k / sum(wx hx^k)
    """
    if V_kN <= 0:
        return np.zeros_like(weights_kN), 0.0, 0.0, np.zeros_like(weights_kN)

    k = vertical_distribution_exponent(T1)
    wxhk = weights_kN * (heights_m ** k)
    denom = np.sum(wxhk)
    if denom <= 0:
        return np.zeros_like(weights_kN), k, 0.0, np.zeros_like(weights_kN)

    cvx = wxhk / denom
    Ft = compute_top_concentrated_force(T1, V_kN, include_top_force)
    Fx = cvx * max(V_kN - Ft, 0.0)
    Fx[-1] += Ft
    return Fx, k, Ft, cvx



def story_shear_from_floor_forces(floor_forces_kN: np.ndarray) -> np.ndarray:
    n = len(floor_forces_kN)
    Vstory = np.zeros(n)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running += floor_forces_kN[i]
        Vstory[i] = running
    return Vstory



def design_spectrum_asce_nscp(T: float, SDS: float, SD1: float, TL: float) -> float:
    T = max(T, 1e-6)
    if SDS <= 1e-12:
        return 0.0

    Ts = SD1 / SDS if SDS > 1e-12 else 1.0
    T0 = 0.2 * Ts

    if T <= T0:
        return SDS * (0.4 + 0.6 * T / max(T0, 1e-9))
    if T <= Ts:
        return SDS
    if T <= TL:
        return SD1 / T
    return SD1 * TL / (T ** 2)



def design_spectrum_ubc97(T: float, Ca: float, Cv: float) -> float:
    T = max(T, 1e-6)
    if Ca <= 1e-12:
        return 0.0

    Ts = Cv / max(2.5 * Ca, 1e-12)
    T0 = 0.2 * Ts

    if T <= T0:
        return Ca * (1.0 + 1.5 * T / max(T0, 1e-9))
    if T <= Ts:
        return 2.5 * Ca
    return Cv / T



def design_spectrum_sa(
    T: float,
    spectrum_family: str,
    SDS: float,
    SD1: float,
    TL: float,
    Ca: float,
    Cv: float,
) -> float:
    if spectrum_family == "UBC 97":
        return design_spectrum_ubc97(T, Ca, Cv)
    return design_spectrum_asce_nscp(T, SDS, SD1, TL)



def modal_lateral_forces(
    M: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
    periods: np.ndarray,
    Sa_modes_g: np.ndarray,
    R: float,
    Ie: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_story, n_modes = phi.shape
    mass_diag = np.diag(M)
    mode_force_matrix = np.zeros((n_story, n_modes), dtype=float)
    mode_base_shear = np.zeros(n_modes, dtype=float)

    reduction = max(R / Ie, 1e-9)
    for r in range(n_modes):
        mode_shape = phi[:, r]
        coeff = gamma[r] * (Sa_modes_g[r] / reduction)
        Fx_r = coeff * mass_diag * mode_shape * G
        mode_force_matrix[:, r] = Fx_r
        mode_base_shear[r] = np.sum(Fx_r)

    return mode_force_matrix, mode_base_shear



def combine_srss(values_by_mode: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(values_by_mode ** 2, axis=1))



def cqc_correlation_matrix(omega: np.ndarray, damping_ratio: float) -> np.ndarray:
    n = len(omega)
    rho = np.eye(n)
    z = damping_ratio
    for i in range(n):
        for j in range(n):
            if i == j:
                rho[i, j] = 1.0
            else:
                beta = omega[j] / max(omega[i], 1e-12)
                num = 8 * z * z * beta * (1 + beta) * (beta ** 1.5)
                den = (1 - beta ** 2) ** 2 + 4 * z * z * beta * (1 + beta) ** 2
                rho[i, j] = num / max(den, 1e-12)
    return rho



def combine_cqc(values_by_mode: np.ndarray, omega: np.ndarray, damping_ratio: float) -> np.ndarray:
    rho = cqc_correlation_matrix(omega, damping_ratio)
    n_story, n_modes = values_by_mode.shape
    out = np.zeros(n_story, dtype=float)

    for i in range(n_story):
        total = 0.0
        for r in range(n_modes):
            for s in range(n_modes):
                total += rho[r, s] * values_by_mode[i, r] * values_by_mode[i, s]
        out[i] = math.sqrt(max(total, 0.0))
    return out



def scale_to_target_base_shear(floor_forces: np.ndarray, target_base_shear: float) -> Tuple[np.ndarray, float]:
    current = np.sum(floor_forces)
    if current <= 1e-12:
        return floor_forces.copy(), 1.0
    scale = target_base_shear / current
    return floor_forces * scale, scale


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def frame_plot(n_story: int, story_h: float, lateral: np.ndarray | None = None, title: str = "Frame View"):
    x_left = 0.0
    x_right = 6.0
    y = np.arange(0, n_story + 1) * story_h

    dx = np.zeros(n_story + 1)
    if lateral is not None:
        dx[1:] = lateral

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_left + dx, y=y, mode="lines+markers", name="Left Column"))
    fig.add_trace(go.Scatter(x=x_right + dx, y=y, mode="lines+markers", name="Right Column"))

    for i in range(1, n_story + 1):
        fig.add_trace(
            go.Scatter(
                x=[x_left + dx[i], x_right + dx[i]],
                y=[y[i], y[i]],
                mode="lines",
                showlegend=False,
            )
        )

    fig.update_layout(title=title, xaxis_title="Horizontal Position / Relative Sway", yaxis_title="Height (m)", height=520)
    return fig



def mode_shape_plot(phi: np.ndarray, story_h: float, mode_number: int):
    n = phi.shape[0]
    y = np.arange(1, n + 1) * story_h
    x = phi[:, mode_number - 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[0, x], y=np.r_[0, y], mode="lines+markers", name=f"Mode {mode_number}"))
    fig.update_layout(title=f"Mode Shape {mode_number}", xaxis_title="Normalized Lateral Displacement", yaxis_title="Height (m)", height=420)
    return fig



def force_plot(floor_forces_kN: np.ndarray, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=floor_forces_kN,
            y=[f"Floor {i+1}" for i in range(len(floor_forces_kN))],
            orientation="h",
            text=np.round(floor_forces_kN, 2),
            textposition="outside",
            name="Fx",
        )
    )
    fig.update_layout(title=title, xaxis_title="Force (kN)", height=420)
    return fig



def multi_mode_force_plot(mode_force_matrix: np.ndarray):
    fig = go.Figure()
    n_story, n_modes = mode_force_matrix.shape
    floors = [f"Floor {i+1}" for i in range(n_story)]
    for r in range(n_modes):
        fig.add_trace(
            go.Bar(
                x=mode_force_matrix[:, r],
                y=floors,
                orientation="h",
                name=f"Mode {r+1}",
            )
        )
    fig.update_layout(barmode="group", title="Dynamic Floor Force Contribution by Mode", xaxis_title="Force (kN)", height=480)
    return fig



def spectrum_plot(spectrum_family: str, SDS: float, SD1: float, TL: float, Ca: float, Cv: float):
    T_vals = np.linspace(0.0, max(4.0, TL + 1.0), 300)
    Sa_vals = [design_spectrum_sa(T, spectrum_family, SDS, SD1, TL, Ca, Cv) for T in T_vals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_vals, y=Sa_vals, mode="lines", name=spectrum_family))
    fig.update_layout(title=f"{spectrum_family} Design Response Spectrum", xaxis_title="Period T (s)", yaxis_title="Sa (g)", height=420)
    return fig


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🏢 MDOF Building Dynamics Explorer")
st.caption(
    "Interactive classroom tool for modal properties, modal seismic forces, SRSS/CQC combination, "
    "static base shear matching, ELF vertical distribution, and response-spectrum comparison."
)

with st.sidebar:
    st.header("Building Inputs")
    n_story = st.selectbox("Number of storeys", [2, 3, 4, 5], index=2)
    story_h = st.number_input("Typical storey height (m)", min_value=2.5, max_value=6.0, value=3.0, step=0.1)

    st.subheader("Seismic Inputs")
    code_basis = st.selectbox("Teaching basis", ["NSCP / IBC / UBC ELF + Response Spectrum", "Modal dynamics only"])
    spectrum_family = st.selectbox("Response spectrum format", ["ASCE / NSCP", "UBC 97"], index=0)

    if spectrum_family == "ASCE / NSCP":
        SDS = st.number_input("SDS", min_value=0.05, max_value=2.50, value=0.75, step=0.05)
        SD1 = st.number_input("SD1", min_value=0.02, max_value=2.50, value=0.45, step=0.05)
        TL = st.number_input("TL (s)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
        Ca = 0.0
        Cv = 0.0
    else:
        Ca = st.number_input("Ca", min_value=0.05, max_value=2.50, value=0.40, step=0.05)
        Cv = st.number_input("Cv", min_value=0.05, max_value=5.00, value=0.64, step=0.05)
        SDS = 2.5 * Ca
        SD1 = Cv
        TL = st.number_input("Reference long-period plot limit TL (s)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)

    R = st.number_input("Response modification factor R", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
    Ie = st.number_input("Importance factor Ie", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    damping_percent = st.number_input("Damping ratio (%) for CQC", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    combine_method = st.selectbox("Modal combination method", ["SRSS", "CQC"])
    match_static_base_shear = st.checkbox("Scale combined dynamic results to match static ELF base shear", value=True)

    st.subheader("Static ELF Distribution")
    include_top_force = st.checkbox("Include roof top concentrated force Ft when applicable", value=True)

    st.subheader("Base Shear Control")
    base_shear_mode = st.selectbox(
        "Base shear input mode",
        [
            "Auto calculate both",
            "Manual static only",
            "Manual dynamic only",
            "Manual static and dynamic",
        ],
        index=0,
    )

    manual_static_base_shear = None
    manual_dynamic_base_shear = None

    if base_shear_mode in ["Manual static only", "Manual static and dynamic"]:
        manual_static_base_shear = st.number_input(
            "Manual Static Base Shear Vstatic (kN)",
            min_value=0.0,
            value=1000.0,
            step=10.0,
        )

    if base_shear_mode in ["Manual dynamic only", "Manual static and dynamic"]:
        manual_dynamic_base_shear = st.number_input(
            "Manual Dynamic Base Shear Vdynamic (kN)",
            min_value=0.0,
            value=900.0,
            step=10.0,
        )

    st.subheader("Visualization")
    scale_factor = st.slider("Mode/frame exaggeration", min_value=0.5, max_value=30.0, value=8.0, step=0.5)
    selected_mode = st.selectbox("Mode to visualize", list(range(1, n_story + 1)), index=0)

st.markdown("### Floor Input Table")
default_weights = [900.0, 900.0, 900.0, 900.0, 900.0][:n_story]
default_k = [35000.0, 32000.0, 29000.0, 26000.0, 23000.0][:n_story]

input_df = pd.DataFrame(
    {
        "Floor": [f"Floor {i+1}" for i in range(n_story)],
        "Weight_kN": default_weights,
        "Story_Stiffness_kN_per_m": default_k,
    }
)

edited_df = st.data_editor(input_df, use_container_width=True, num_rows="fixed", key="floor_input_editor")
weights_kN = ensure_positive(edited_df["Weight_kN"].to_numpy(dtype=float), fallback=100.0)
stiffness_kNpm = ensure_positive(edited_df["Story_Stiffness_kN_per_m"].to_numpy(dtype=float), fallback=1000.0)
heights_m = np.arange(1, n_story + 1, dtype=float) * story_h

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------
M = build_mass_matrix(weights_kN)
K = build_stiffness_matrix(stiffness_kNpm)
omega, phi = solve_modes(M, K)
periods = compute_periods(omega)
gamma, modal_mass, eff_modal_mass = modal_properties(M, phi)

mass_diag = np.diag(M)
total_mass = np.sum(mass_diag)
eff_mass_ratio = eff_modal_mass / total_mass if total_mass > 0 else np.zeros_like(eff_modal_mass)
cum_eff_mass_ratio = np.cumsum(eff_mass_ratio)

T1 = periods[0] if len(periods) > 0 else 0.0
W_total = float(np.sum(weights_kN))

# ----------------------------
# STATIC BASE SHEAR
# ----------------------------
Vbase_static_auto = (
    approximate_code_base_shear(W_total, T1, SDS, R, Ie)
    if code_basis != "Modal dynamics only"
    else 0.0
)

if base_shear_mode in ["Manual static only", "Manual static and dynamic"]:
    Vbase_static = float(manual_static_base_shear)
    static_source = "Manual"
else:
    Vbase_static = Vbase_static_auto
    static_source = "Auto"

if Vbase_static > 0:
    static_floor_forces, elf_k, Ft_static, cvx = distribute_lateral_forces(
        weights_kN,
        heights_m,
        Vbase_static,
        T1,
        include_top_force=include_top_force,
    )
else:
    static_floor_forces = np.zeros(n_story)
    elf_k = vertical_distribution_exponent(T1)
    Ft_static = 0.0
    cvx = np.zeros(n_story)

static_story_shear = story_shear_from_floor_forces(static_floor_forces)

# ----------------------------
# DYNAMIC BASE SHEAR
# ----------------------------
Sa_modes_g = (
    np.array(
        [design_spectrum_sa(T, spectrum_family, SDS, SD1, TL, Ca, Cv) for T in periods],
        dtype=float,
    )
    if code_basis != "Modal dynamics only"
    else np.zeros_like(periods)
)

mode_force_matrix, mode_base_shear = modal_lateral_forces(M, phi, gamma, periods, Sa_modes_g, R, Ie)
mode_story_shear_matrix = np.column_stack(
    [story_shear_from_floor_forces(mode_force_matrix[:, r]) for r in range(n_story)]
)

if combine_method == "SRSS":
    dynamic_floor_combined_auto = combine_srss(mode_force_matrix)
    dynamic_story_shear_combined_auto = combine_srss(mode_story_shear_matrix)
else:
    dynamic_floor_combined_auto = combine_cqc(mode_force_matrix, omega, damping_percent / 100.0)
    dynamic_story_shear_combined_auto = combine_cqc(mode_story_shear_matrix, omega, damping_percent / 100.0)

Vbase_dynamic_auto = float(np.sum(dynamic_floor_combined_auto))

if base_shear_mode in ["Manual dynamic only", "Manual static and dynamic"]:
    Vbase_dynamic = float(manual_dynamic_base_shear)
    dynamic_source = "Manual"

    if Vbase_dynamic_auto > 1e-12:
        dynamic_floor_combined, dyn_user_scale = scale_to_target_base_shear(dynamic_floor_combined_auto, Vbase_dynamic)
        dynamic_story_shear_combined = story_shear_from_floor_forces(dynamic_floor_combined)
    else:
        dynamic_floor_combined = dynamic_floor_combined_auto.copy()
        dynamic_story_shear_combined = dynamic_story_shear_combined_auto.copy()
        dyn_user_scale = 1.0
else:
    Vbase_dynamic = Vbase_dynamic_auto
    dynamic_floor_combined = dynamic_floor_combined_auto.copy()
    dynamic_story_shear_combined = dynamic_story_shear_combined_auto.copy()
    dynamic_source = "Auto"
    dyn_user_scale = 1.0

# ----------------------------
# OPTIONAL MATCH TO STATIC
# ----------------------------
if match_static_base_shear and Vbase_static > 0 and np.sum(dynamic_floor_combined) > 1e-12:
    dynamic_floor_scaled, scale_ratio = scale_to_target_base_shear(dynamic_floor_combined, Vbase_static)
    dynamic_story_scaled = story_shear_from_floor_forces(dynamic_floor_scaled)
else:
    dynamic_floor_scaled = dynamic_floor_combined.copy()
    dynamic_story_scaled = dynamic_story_shear_combined.copy()
    scale_ratio = 1.0

selected_mode_shape = phi[:, selected_mode - 1]
selected_mode_visual = selected_mode_shape * scale_factor

# ---------------------------------------------------------
# Results overview
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Fundamental Period T1 (s)", f"{T1:.4f}")
col2.metric(f"Static Base Shear ({static_source}) (kN)", f"{Vbase_static:,.2f}")
col3.metric(f"Dynamic Base Shear ({dynamic_source}) (kN)", f"{Vbase_dynamic:,.2f}")
col4.metric("Scale Factor to Static", f"{scale_ratio:.3f}")
col5.metric("1st Mode Mass Participation", f"{eff_mass_ratio[0] * 100:.2f}%")

res_tab1, res_tab2, res_tab3, res_tab4, res_tab5, res_tab6 = st.tabs(
    ["Visualization", "Modal Properties", "Static ELF", "Dynamic Modal Forces", "Spectrum", "Matrices & Downloads"]
)

with res_tab1:
    a, b = st.columns([1.2, 1])
    with a:
        st.plotly_chart(frame_plot(n_story, story_h, title="Undeformed Frame"), use_container_width=True, key="undeformed_frame_chart")
        st.plotly_chart(
            frame_plot(n_story, story_h, lateral=selected_mode_visual, title=f"Mode {selected_mode} Visualization"),
            use_container_width=True,
            key=f"mode_visualization_chart_{selected_mode}",
        )
    with b:
        st.plotly_chart(mode_shape_plot(phi, story_h, selected_mode), use_container_width=True, key=f"selected_mode_shape_chart_{selected_mode}")
        viz_df = pd.DataFrame(
            {
                "Floor": [f"Floor {i+1}" for i in range(n_story)],
                "Height_m": heights_m,
                f"Mode_{selected_mode}_Shape": np.round(selected_mode_shape, 5),
                "Scaled_Display_Displacement": np.round(selected_mode_visual, 5),
                f"Mode_{selected_mode}_Force_kN": np.round(mode_force_matrix[:, selected_mode - 1], 3),
            }
        )
        st.dataframe(viz_df, use_container_width=True)

with res_tab2:
    modal_df = pd.DataFrame(
        {
            "Mode": np.arange(1, n_story + 1),
            "Omega_rad_per_s": np.round(omega, 5),
            "Period_s": np.round(periods, 5),
            "Sa_g": np.round(Sa_modes_g, 5),
            "Gamma": np.round(gamma, 5),
            "Modal_Mass": np.round(modal_mass, 5),
            "Effective_Modal_Mass": np.round(eff_modal_mass, 5),
            "Eff_Mass_Ratio_%": np.round(eff_mass_ratio * 100, 3),
            "Cumulative_%": np.round(cum_eff_mass_ratio * 100, 3),
            "Mode_Base_Shear_kN": np.round(mode_base_shear, 3),
        }
    )
    st.dataframe(modal_df, use_container_width=True)

    mode_cols = st.columns(min(n_story, 4))
    for i in range(n_story):
        with mode_cols[i % len(mode_cols)]:
            st.plotly_chart(mode_shape_plot(phi, story_h, i + 1), use_container_width=True, key=f"mode_shape_chart_{i+1}")

    mode_shape_df = pd.DataFrame(phi, columns=[f"Mode {i+1}" for i in range(n_story)])
    mode_shape_df.insert(0, "Floor", [f"Floor {i+1}" for i in range(n_story)])
    st.markdown("#### Mode Shape Matrix")
    st.dataframe(np.round(mode_shape_df, 5), use_container_width=True)

with res_tab3:
    if code_basis != "Modal dynamics only":
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(force_plot(static_floor_forces, "Static ELF Floor Forces"), use_container_width=True, key="static_force_plot")
        with c2:
            static_df = pd.DataFrame(
                {
                    "Floor": [f"Floor {i+1}" for i in range(n_story)],
                    "Height_m": np.round(heights_m, 3),
                    "Weight_kN": np.round(weights_kN, 3),
                    "Cvx": np.round(cvx, 5),
                    "ELF_Fx_kN": np.round(static_floor_forces, 3),
                    "ELF_Story_Shear_kN": np.round(static_story_shear, 3),
                }
            )
            st.dataframe(static_df, use_container_width=True)

        info1, info2, info3 = st.columns(3)
        info1.metric("ELF vertical exponent k", f"{elf_k:.3f}")
        info2.metric("Roof top force Ft (kN)", f"{Ft_static:,.2f}")
        info3.metric("Sum Fx (kN)", f"{np.sum(static_floor_forces):,.2f}")

        st.info(
            "Static ELF vertical distribution now uses Cvx = wx hx^k / Σ(wx hx^k), "
            "with optional roof top concentrated force Ft added to the top floor when applicable. "
            "The auto base shear V remains a simplified teaching estimate unless manually overridden."
        )
    else:
        st.warning("Static ELF is disabled in modal dynamics only mode.")

with res_tab4:
    st.markdown("#### Per-Mode Dynamic Floor Forces")
    top1, top2 = st.columns([1.1, 0.9])
    with top1:
        st.plotly_chart(multi_mode_force_plot(mode_force_matrix), use_container_width=True, key="multi_mode_force_plot")
    with top2:
        selected_force_df = pd.DataFrame(
            {
                "Floor": [f"Floor {i+1}" for i in range(n_story)],
                f"Mode_{selected_mode}_Force_kN": np.round(mode_force_matrix[:, selected_mode - 1], 3),
                f"Mode_{selected_mode}_StoryShear_kN": np.round(mode_story_shear_matrix[:, selected_mode - 1], 3),
            }
        )
        st.dataframe(selected_force_df, use_container_width=True)

    per_mode_force_df = pd.DataFrame(mode_force_matrix, columns=[f"Mode {i+1}" for i in range(n_story)])
    per_mode_force_df.insert(0, "Floor", [f"Floor {i+1}" for i in range(n_story)])
    st.dataframe(np.round(per_mode_force_df, 3), use_container_width=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.plotly_chart(force_plot(dynamic_floor_combined, f"Combined Dynamic Floor Forces ({combine_method})"), use_container_width=True, key="combined_dynamic_force_plot")
        if match_static_base_shear and code_basis != "Modal dynamics only":
            st.plotly_chart(
                force_plot(dynamic_floor_scaled, f"Scaled Dynamic Floor Forces matched to Static Base Shear ({combine_method})"),
                use_container_width=True,
                key="scaled_dynamic_force_plot",
            )
    with d2:
        dynamic_df = pd.DataFrame(
            {
                "Floor": [f"Floor {i+1}" for i in range(n_story)],
                "Dynamic_Combined_Fx_kN": np.round(dynamic_floor_combined, 3),
                "Dynamic_Combined_StoryShear_kN": np.round(dynamic_story_shear_combined, 3),
                "Scaled_Dynamic_Fx_kN": np.round(dynamic_floor_scaled, 3),
                "Scaled_Dynamic_StoryShear_kN": np.round(dynamic_story_scaled, 3),
            }
        )
        st.dataframe(dynamic_df, use_container_width=True)

    st.markdown("#### Base Shear Comparison Table")
    compare_df = pd.DataFrame(
        {
            "Item": [
                "Static Base Shear",
                "Dynamic Base Shear",
                "Dynamic after User Override",
                "Dynamic after Static Matching",
            ],
            "Auto_Value_kN": [
                Vbase_static_auto,
                Vbase_dynamic_auto,
                Vbase_dynamic_auto,
                float(np.sum(dynamic_floor_combined)),
            ],
            "Final_Value_kN": [
                Vbase_static,
                Vbase_dynamic,
                float(np.sum(dynamic_floor_combined)),
                float(np.sum(dynamic_floor_scaled)),
            ],
        }
    )
    st.dataframe(compare_df, use_container_width=True)

    st.markdown(
        f"**Interpretation:** each mode gives a different vertical pattern of inertia force. The app combines them using **{combine_method}**, then "
        f"{'scales the combined dynamic result to the static ELF base shear.' if match_static_base_shear else 'keeps the unscaled dynamic result.'}"
    )

with res_tab5:
    s1, s2 = st.columns([1.2, 0.8])
    with s1:
        st.plotly_chart(
            spectrum_plot(spectrum_family, SDS, SD1, TL, Ca, Cv),
            use_container_width=True,
            key="spectrum_plot",
        )
    with s2:
        if spectrum_family == "ASCE / NSCP":
            Ts = SD1 / SDS if SDS > 1e-12 else 0.0
            T0 = 0.2 * Ts
            spec_df = pd.DataFrame(
                {
                    "Parameter": ["SDS", "SD1", "T0", "Ts", "TL"],
                    "Value": [SDS, SD1, T0, Ts, TL],
                }
            )
            st.dataframe(np.round(spec_df, 5), use_container_width=True)
            st.caption("Sa(T): linear rise to T0, plateau to Ts, 1/T branch to TL, and 1/T² branch beyond TL.")
        else:
            Ts = Cv / max(2.5 * Ca, 1e-12)
            T0 = 0.2 * Ts
            spec_df = pd.DataFrame(
                {
                    "Parameter": ["Ca", "Cv", "T0", "Ts", "Plateau Sa"],
                    "Value": [Ca, Cv, T0, Ts, 2.5 * Ca],
                }
            )
            st.dataframe(np.round(spec_df, 5), use_container_width=True)
            st.caption("Sa(T): linear rise to T0, plateau = 2.5Ca to Ts, then Cv/T descending branch.")

with res_tab6:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### Mass Matrix M")
        st.dataframe(pd.DataFrame(np.round(M, 5)), use_container_width=True)
    with m2:
        st.markdown("#### Stiffness Matrix K")
        st.dataframe(pd.DataFrame(np.round(K, 5)), use_container_width=True)

    export_modal = modal_df.copy()
    export_mode_shapes = mode_shape_df.copy()
    export_static = pd.DataFrame(
        {
            "Floor": [f"Floor {i+1}" for i in range(n_story)],
            "Height_m": heights_m,
            "Weight_kN": weights_kN,
            "Cvx": cvx,
            "ELF_k": np.full(n_story, elf_k),
            "ELF_Ft_kN": np.full(n_story, Ft_static),
            "ELF_Fx_kN": static_floor_forces,
            "ELF_Story_Shear_kN": static_story_shear,
        }
    )
    export_dynamic = pd.DataFrame(
        {
            "Floor": [f"Floor {i+1}" for i in range(n_story)],
            **{f"Mode_{i+1}_Fx_kN": mode_force_matrix[:, i] for i in range(n_story)},
            "Combined_Fx_Auto_kN": dynamic_floor_combined_auto,
            "Combined_Fx_Final_kN": dynamic_floor_combined,
            "Scaled_to_Static_Fx_kN": dynamic_floor_scaled,
            "Scaled_to_Static_Story_Shear_kN": dynamic_story_scaled,
        }
    )
    export_base_shear_summary = pd.DataFrame(
        {
            "Parameter": [
                "Response Spectrum Format",
                "Static Base Shear Auto",
                "Static Base Shear Final",
                "Dynamic Base Shear Auto",
                "Dynamic Base Shear Final",
                "Dynamic User Scale Factor",
                "Dynamic-to-Static Scale Factor",
                "Static Source",
                "Dynamic Source",
            ],
            "Value": [
                spectrum_family,
                Vbase_static_auto,
                Vbase_static,
                Vbase_dynamic_auto,
                Vbase_dynamic,
                dyn_user_scale,
                scale_ratio,
                static_source,
                dynamic_source,
            ],
        }
    )

    st.download_button(
        "Download modal_properties.csv",
        export_modal.to_csv(index=False).encode("utf-8"),
        file_name="modal_properties.csv",
        mime="text/csv",
        key="download_modal_csv",
    )
    st.download_button(
        "Download mode_shapes.csv",
        export_mode_shapes.to_csv(index=False).encode("utf-8"),
        file_name="mode_shapes.csv",
        mime="text/csv",
        key="download_mode_shapes_csv",
    )
    st.download_button(
        "Download static_elf.csv",
        export_static.to_csv(index=False).encode("utf-8"),
        file_name="static_elf.csv",
        mime="text/csv",
        key="download_static_csv",
    )
    st.download_button(
        "Download dynamic_modal_forces.csv",
        export_dynamic.to_csv(index=False).encode("utf-8"),
        file_name="dynamic_modal_forces.csv",
        mime="text/csv",
        key="download_dynamic_csv",
    )
    st.download_button(
        "Download base_shear_summary.csv",
        export_base_shear_summary.to_csv(index=False).encode("utf-8"),
        file_name="base_shear_summary.csv",
        mime="text/csv",
        key="download_base_shear_summary_csv",
    )

st.markdown("---")
st.markdown(
    "**Teaching use:** change floor weights, storey stiffness, response-spectrum format, damping, base shear mode, and mode combination method. "
    "Students can then compare static ELF forces, individual modal dynamic forces, combined SRSS/CQC results, and spreadsheet-imposed base shear values."
)
