

# ===== DL REGRESSION PLAYGROUND (Unified Sidebar, K1 Unique Keys) =====
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="DL Regression Playground", page_icon="🧪", layout="wide")
st.title("🧪 Deep Learning Regression Playground — Unified Sidebar (K1 Keys)")

st.write("""
This file uses ONE unified sidebar for both 1D and 2D regression,
with auto-updating (A1-β) and globally unique keys (K1).
""")

# --------------------------------------------------------------------------
# ⚠️ RESTORED ARCHITECTURE DIAGRAM FUNCTION (MANDATORY)
# --------------------------------------------------------------------------
def arch_html(desc, title="Architecture"):
    html = (
        f"<div style='font-weight:700;font-size:18px;margin-bottom:8px'>{title}</div>"
        "<div style='display:flex;gap:12px;overflow-x:auto;padding:10px;"
        "border:1px solid #eee;border-radius:10px;background:#fafafa'>"
    )

    for i, layer in enumerate(desc):
        t = layer["type"].capitalize()
        sub = (
            f"{layer['units']}u {layer.get('act','')}"
            if layer["type"] == "dense"
            else f"{layer['units']} dims"
        )

        html += (
            "<div style='min-width:130px;background:white;border:1px solid #ddd;"
            "border-radius:10px;padding:10px;text-align:center;'>"
            f"<div style='font-weight:700'>{t}</div>"
            f"<div style='color:#555;font-size:12px'>{sub}</div>"
            "</div>"
        )

        if i < len(desc) - 1:
            html += "<div style='font-size:20px;color:#888;display:flex;align-items:center'>→</div>"

    return html + "</div>"

# --------------------------------------------------------------------------
# TABS (Segment 2 = 1D, Segment 3 = 2D)
# --------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📈 1D Regression", "🌐 2D Regression"])

# ========== SEGMENT 2 — 1D REGRESSION WITH UNIFIED SIDEBAR (K1 UNIQUE KEYS) ==========
with tab1:
    st.header("📈 1D Regression")

    # --- SIDEBAR: 1D Controls -------------------------------------------------
    sidebar = st.sidebar
    sidebar.markdown("## 🧩 1D Controls")

    # Collapsible: Function
    with sidebar.expander("1D — Function", expanded=False):
        func_choice_1d = st.selectbox(
            "Select Function",
            ["sinusoid","polynomial","exponential","logarithmic","rational","piecewise"],
            key="1d_func_choice"
        )

        if func_choice_1d == "sinusoid":
            amp_1d   = st.slider("Amplitude", 0.1, 5.0, 1.0, key="1d_func_amp")
            freq_1d  = st.slider("Frequency (Hz)", 0.1, 5.0, 1.0, key="1d_func_freq")
            phase_1d = st.slider("Phase (rad)", -3.14, 3.14, 0.0, key="1d_func_phase")
            bias_1d  = st.slider("Vertical shift", -2.0, 2.0, 0.0, key="1d_func_bias")

        if func_choice_1d == "piecewise":
            saw_amp_1d   = st.slider("Amplitude", 0.1, 5.0, 1.0, key="1d_func_saw_amp")
            saw_freq_1d  = st.slider("Frequency", 0.1, 10.0, 1.0, key="1d_func_saw_freq")
            saw_bias_1d  = st.slider("Vertical shift", -2.0, 2.0, 0.0, key="1d_func_saw_bias")

    # Collapsible: Domain
    with sidebar.expander("1D — Domain", expanded=False):
        xmin_1d = st.number_input("x min", value=-3.0, key="1d_domain_xmin")
        xmax_1d = st.number_input("x max", value=3.0, key="1d_domain_xmax")

    # Collapsible: Data
    with sidebar.expander("1D — Data", expanded=False):
        N_1d = st.slider("Samples", 200, 5000, 1500, key="1d_data_N")
        noise_1d = st.slider("Noise σ", 0.0, 1.0, 0.3, key="1d_data_noise")
        test_ratio_1d = st.slider("Test ratio", 0.05, 0.5, 0.2, key="1d_data_ratio")
        seed_1d = st.number_input("Seed", value=0, key="1d_data_seed")

    # Collapsible: Architecture
    with sidebar.expander("1D — Architecture", expanded=False):
        num_layers_1d = st.slider("Hidden layers", 0, 12, 3, key="1d_arch_layers")
        hidden_sizes_1d = [
            st.number_input(f"Layer {i+1} neurons", 1, 1024, 64, key=f"1d_arch_hlayer_{i}")
            for i in range(num_layers_1d)
        ]
        epochs_1d = st.slider("Epochs", 50, 4000, 600, key="1d_arch_epochs")
        lr_1d = st.selectbox("Learning Rate", [0.1,0.03,0.01,0.003,0.001], index=2, key="1d_arch_lr")
        use_gpu_1d = st.checkbox("Use GPU", value=True, key="1d_arch_gpu")

    # Collapsible: Display
    with sidebar.expander("1D — Display", expanded=False):
        show_test_1d = st.checkbox("Show Test Data", value=True, key="1d_display_test")
        show_true_1d = st.checkbox("Show True Function", value=True, key="1d_display_true")

    # Collapsible: Train
    with sidebar.expander("1D — Train", expanded=False):
        train_button_1d = st.button("🚀 Train 1D Model", key="1d_train_button")

    # --- Define target function ----------------------------------------------
    def f1d(x):
        if func_choice_1d == "sinusoid":
            return amp_1d * torch.sin(2 * np.pi * freq_1d * x + phase_1d) + bias_1d
        if func_choice_1d == "polynomial":
            return 0.5*x + 0.2*(x**2)
        if func_choice_1d == "exponential":
            return torch.exp(0.3*x)
        if func_choice_1d == "logarithmic":
            return torch.log(torch.clamp(x + 4, 1e-6))
        if func_choice_1d == "rational":
            return x / (1 + x**2)
        if func_choice_1d == "piecewise":
            # normalized sawtooth at frequency saw_freq_1d
            phase = torch.remainder(x * saw_freq_1d, 1.0)  # in [0,1)
            saw = 2 * torch.abs(phase - 0.5) - 0.5         # centered
            return saw_amp_1d * saw + saw_bias_1d

    # --- Generate data --------------------------------------------------------
    torch.manual_seed(seed_1d)
    x1d = torch.linspace(xmin_1d, xmax_1d, N_1d).unsqueeze(1)
    y_clean_1d = f1d(x1d)
    y1d = y_clean_1d + noise_1d * torch.randn_like(y_clean_1d)
    perm = torch.randperm(N_1d)
    x1d, y1d = x1d[perm], y1d[perm]
    split_1d = int((1 - test_ratio_1d) * N_1d)
    x_train_1d, x_test_1d = x1d[:split_1d], x1d[split_1d:]
    y_train_1d, y_test_1d = y1d[:split_1d], y1d[split_1d:]

    # --- Build model ----------------------------------------------------------
    desc_1d = [{"type":"input","units":1}]
    layers_1d = []
    prev = 1
    for h in hidden_sizes_1d:
        layers_1d += [nn.Linear(prev, h), nn.ReLU()]
        desc_1d.append({"type":"dense","units":h,"act":"ReLU"})
        prev = h
    layers_1d.append(nn.Linear(prev, 1))
    desc_1d.append({"type":"output","units":1})
    model_1d = nn.Sequential(*layers_1d)

    # --- Train ---------------------------------------------------------------
    if train_button_1d:
        device = torch.device("cuda" if (use_gpu_1d and torch.cuda.is_available()) else "cpu")
        model_1d = model_1d.to(device)
        Xt = x_train_1d.to(device)
        yt = y_train_1d.to(device)
        crit = nn.MSELoss()
        opt = optim.Adam(model_1d.parameters(), lr=lr_1d)
        losses_1d = []
        for _ in range(epochs_1d):
            opt.zero_grad()
            pred = model_1d(Xt)
            L = crit(pred, yt)
            L.backward(); opt.step()
            losses_1d.append(L.item())
        st.session_state["1d_pred"] = model_1d(torch.linspace(xmin_1d, xmax_1d, 800).unsqueeze(1).to(device)).detach().cpu().numpy().squeeze()
        st.session_state["1d_losses"] = losses_1d
        st.success("1D training complete!")
        st.rerun()

    # --- Plot 1D regression ---------------------------------------------------
    xline = torch.linspace(xmin_1d, xmax_1d, 800).unsqueeze(1)
    ytrue_line = f1d(xline).numpy()

    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(x_train_1d, y_train_1d, c="blue", alpha=0.3, label="Train")
    if show_test_1d:
        ax.scatter(x_test_1d, y_test_1d, c="orange", alpha=0.4, label="Test")
    if show_true_1d:
        ax.plot(xline, ytrue_line, c="green", lw=3, label="True function")
    if "1d_pred" in st.session_state:
        ax.plot(xline, st.session_state["1d_pred"], c="red", lw=3, label="Learned model")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # --- Architecture diagram --------------------------------------------------
    st.components.v1.html(arch_html(desc_1d, "1D Architecture"), height=220)

    # --- Training curve --------------------------------------------------------
    if "1d_losses" in st.session_state:
        figL, axL = plt.subplots(figsize=(5,3))
        axL.plot(st.session_state["1d_losses"], c="blue")
        axL.set_title("1D Training Error Evolution")
        axL.grid(alpha=0.3)
        st.pyplot(figL)

# ========== SEGMENT 3 — 2D REGRESSION WITH UNIFIED SIDEBAR (K1 UNIQUE KEYS) ==========
with tab2:
    st.header("🌐 2D Regression")

    # --- SIDEBAR: 2D Controls -------------------------------------------------
    sidebar = st.sidebar
    sidebar.markdown("## 🌍 2D Controls")

    # Collapsible: Function
    with sidebar.expander("2D — Function", expanded=False):
        func_choice_2d = st.selectbox(
            "Select Function",
            ["sin(x1)*cos(x2)", "sin(x1*x2)", "exp(-x1² - x2²)", "sin(x1)+cos(x2)", "sin(x1²+x2²)"],
            key="2d_func_choice")

    # Collapsible: Domain
    with sidebar.expander("2D — Domain", expanded=False):
        xmin_2d = st.number_input("x1 min", value=-3.0, key="2d_domain_xmin")
        xmax_2d = st.number_input("x1 max", value=3.0, key="2d_domain_xmax")
        ymin_2d = st.number_input("x2 min", value=-3.0, key="2d_domain_ymin")
        ymax_2d = st.number_input("x2 max", value=3.0, key="2d_domain_ymax")

    # Collapsible: Data
    with sidebar.expander("2D — Data", expanded=False):
        N_2d = st.slider("Samples", 500, 10000, 2500, key="2d_data_N")
        noise_2d = st.slider("Noise σ", 0.0, 1.0, 0.2, key="2d_data_noise")
        test_ratio_2d = st.slider("Test ratio", 0.05, 0.5, 0.2, key="2d_data_ratio")
        seed_2d = st.number_input("Seed", value=0, key="2d_data_seed")

    # Collapsible: Architecture
    with sidebar.expander("2D — Architecture", expanded=False):
        num_layers_2d = st.slider("Hidden layers", 0, 12, 4, key="2d_arch_layers")
        hidden_sizes_2d = [
            st.number_input(f"Layer {i+1} neurons", 1, 1024, 64, key=f"2d_arch_hlayer_{i}")
            for i in range(num_layers_2d)
        ]
        epochs_2d = st.slider("Epochs", 50, 4000, 600, key="2d_arch_epochs")
        lr_2d = st.selectbox("Learning Rate", [0.1,0.03,0.01,0.003,0.001], index=2, key="2d_arch_lr")
        use_gpu_2d = st.checkbox("Use GPU", value=True, key="2d_arch_gpu")

    # Collapsible: Display
    with sidebar.expander("2D — Display", expanded=False):
        show_test_2d = st.checkbox("Show Test Data", value=True, key="2d_display_test")
        show_true_2d = st.checkbox("Show True Function", value=True, key="2d_display_true")

    # Collapsible: Train
    with sidebar.expander("2D — Train", expanded=False):
        train_button_2d = st.button("🚀 Train 2D Model", key="2d_train_button")

    # --- Define 2D target function -------------------------------------------
    def f2d(x1, x2):
        if func_choice_2d == "sin(x1)*cos(x2)":
            return torch.sin(x1) * torch.cos(x2)
        if func_choice_2d == "sin(x1*x2)":
            return torch.sin(x1 * x2)
        if func_choice_2d == "exp(-x1² - x2²)":
            return torch.exp(-(x1**2 + x2**2))
        if func_choice_2d == "sin(x1)+cos(x2)":
            return torch.sin(x1) + torch.cos(x2)
        if func_choice_2d == "sin(x1²+x2²)":
            return torch.sin(x1**2 + x2**2)

    # --- Generate 2D data -----------------------------------------------------
    torch.manual_seed(seed_2d)
    x1 = torch.empty(N_2d).uniform_(xmin_2d, xmax_2d)
    x2 = torch.empty(N_2d).uniform_(ymin_2d, ymax_2d)
    z = f2d(x1, x2) + noise_2d * torch.randn(N_2d)

    perm = torch.randperm(N_2d)
    x1, x2, z = x1[perm], x2[perm], z[perm]

    split_2d = int((1 - test_ratio_2d) * N_2d)
    X_train_2d = torch.stack([x1[:split_2d], x2[:split_2d]], dim=1)
    y_train_2d = z[:split_2d]
    X_test_2d  = torch.stack([x1[split_2d:], x2[split_2d:]], dim=1)
    y_test_2d  = z[split_2d:]

    # --- Build 2D model -------------------------------------------------------
    desc_2d = [{"type":"input","units":2}]
    layers_2d = []
    prev = 2
    for h in hidden_sizes_2d:
        layers_2d += [nn.Linear(prev, h), nn.ReLU()]
        desc_2d.append({"type":"dense","units":h,"act":"ReLU"})
        prev = h
    layers_2d.append(nn.Linear(prev, 1))
    desc_2d.append({"type":"output","units":1})
    model_2d = nn.Sequential(*layers_2d)

    # --- Grid for visualization ----------------------------------------------
    grid = 50
    gx = torch.linspace(xmin_2d, xmax_2d, grid)
    gy = torch.linspace(ymin_2d, ymax_2d, grid)
    GX, GY = torch.meshgrid(gx, gy, indexing="xy")
    GZ_true = f2d(GX, GY).numpy()

    # --- Training -------------------------------------------------------------
    if train_button_2d:
        device = torch.device("cuda" if (use_gpu_2d and torch.cuda.is_available()) else "cpu")
        model_2d = model_2d.to(device)
        Xt = X_train_2d.to(device)
        yt = y_train_2d.unsqueeze(1).to(device)
        crit = nn.MSELoss()
        opt = optim.Adam(model_2d.parameters(), lr=lr_2d)
        losses_2d = []
        for _ in range(epochs_2d):
            model_2d.train(); opt.zero_grad()
            pred = model_2d(Xt)
            L = crit(pred, yt)
            L.backward(); opt.step()
            losses_2d.append(L.item())
        with torch.no_grad():
            grid_inputs = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=1).to(device)
            pred_surface = model_2d(grid_inputs).cpu().numpy().reshape(grid, grid)
        st.session_state["2d_pred_surface"] = pred_surface
        st.session_state["2d_losses"] = losses_2d
        st.success("2D training complete!")
        st.rerun()

    # --- Layout: 3D + Heatmap side-by-side -----------------------------------
    col3d, colheat = st.columns([1,1])

    with col3d:
        st.subheader("🌋 3D Surface")
        traces = []
        if show_true_2d:
            traces.append(go.Surface(x=GX.numpy(), y=GY.numpy(), z=GZ_true, colorscale='Viridis', opacity=0.8, showscale=False))
        if "2d_pred_surface" in st.session_state:
            traces.append(go.Surface(x=GX.numpy(), y=GY.numpy(), z=st.session_state["2d_pred_surface"], colorscale='Inferno', opacity=0.6, showscale=False))
        fig3d = go.Figure(data=traces)
        if show_test_2d:
            fig3d.add_trace(go.Scatter3d(x=x1[:split_2d].numpy(), y=x2[:split_2d].numpy(), z=y_train_2d.numpy(), mode='markers', marker=dict(size=2,color='white')))
        fig3d.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with colheat:
        st.subheader("🟪 True Heatmap")
        fig_h, ax_h = plt.subplots(figsize=(6,5))
        if show_true_2d:
            im = ax_h.imshow(GZ_true, extent=[xmin_2d, xmax_2d, ymin_2d, ymax_2d], origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax_h)
        if show_test_2d:
            ax_h.scatter(x1[:split_2d], x2[:split_2d], c='white', s=6, alpha=0.6)
        st.pyplot(fig_h)

        if "2d_pred_surface" in st.session_state:
            st.subheader("🟥 Learned Heatmap")
            fig_h2, ax_h2 = plt.subplots(figsize=(6,5))
            im2 = ax_h2.imshow(st.session_state["2d_pred_surface"], extent=[xmin_2d, xmax_2d, ymin_2d, ymax_2d], origin='lower', cmap='inferno')
            plt.colorbar(im2, ax=ax_h2)
            st.pyplot(fig_h2)

    # --- Architecture diagram -------------------------------------------------
    st.components.v1.html(arch_html(desc_2d, "2D Architecture"), height=220)

    # --- Training loss curve --------------------------------------------------
    if "2d_losses" in st.session_state:
        figL, axL = plt.subplots(figsize=(5,3))
        axL.plot(st.session_state["2d_losses"], c="blue")
        axL.set_title("2D Training Error Evolution")
        axL.grid(alpha=0.3)
        st.pyplot(figL)
