import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ===============================================================
("🧪 Deep Learning Regression Playground (PyTorch)")# PAGE CONFIG
st.write("Always-on plot shows the data + true function. Press **Train Model** to overlay the learned function.")

# ===============================================================
# SIDEBAR — DATA & FUNCTION
# ===============================================================
st.sidebar.header("📈 Data & Function")

func_choice = st.sidebar.selectbox(
    "Function",
    ["sinusoid", "polynomial", "exponential", "logarithmic",
     "rational", "piecewise (sawtooth-like)"]
)

# DEFAULT DOMAINS
domains = {
    "sinusoid": (-3, 3),
    "polynomial": (-2, 2),
    "exponential": (-2, 2),
    "logarithmic": (0.1, 5),
    "rational": (-3, 3),
    "piecewise (sawtooth-like)": (-3, 3),
}

xmin, xmax = domains[func_choice]
xmin = st.sidebar.number_input("x min", value=float(xmin))
xmax = st.sidebar.number_input("x max", value=float(xmax))

# FUNCTION PARAMETERS
st.sidebar.subheader("🔧 Function params")

amp=freq=phase=bias=c0=c1=c2=c3=A=k=b_log=a=base_e=p=q=r=period=slope=bias_pw=None

if func_choice == "sinusoid":
    amp   = st.sidebar.slider("Amplitude", 0.1, 3.0, 1.0)
    freq  = st.sidebar.slider("Frequency", 0.1, 5.0, 1.0)
    phase = st.sidebar.slider("Phase", -np.pi, np.pi, 0.0)
    bias  = st.sidebar.slider("Bias", -2.0, 2.0, 0.0)

elif func_choice == "polynomial":
    c0 = st.sidebar.slider("c0", -2, 2, 0)
    c1 = st.sidebar.slider("c1", -3, 3, 1)
    c2 = st.sidebar.slider("c2", -3, 3, 0)
    c3 = st.sidebar.slider("c3", -3, 3, 0)

elif func_choice == "exponential":
    A = st.sidebar.slider("A", 0.1, 3.0, 1.0)
    k = st.sidebar.slider("k", -2.0, 2.0, 0.5)
    b_log = st.sidebar.slider("Bias", -2.0, 2.0, 0.0)

elif func_choice == "logarithmic":
    a = st.sidebar.slider("a", 0.1, 3.0, 1.0)
    b_log = st.sidebar.slider("Bias", -2.0, 2.0, 0.0)
    base_e = st.sidebar.checkbox("Use ln", value=True)

elif func_choice == "rational":
    p = st.sidebar.slider("p", 0.1, 5.0, 1.0)
    q = st.sidebar.slider("q", 0.1, 5.0, 1.0)
    r = st.sidebar.slider("Bias", -2.0, 2.0, 0.0)

elif func_choice == "piecewise (sawtooth-like)":
    period = st.sidebar.slider("Period", 0.5, 5.0, 2.0)
    slope  = st.sidebar.slider("Slope", 0.1, 3.0, 1.0)
    bias_pw= st.sidebar.slider("Bias", -2.0, 2.0, 0.0)

# ===============================================================
# DATA SETTINGS
# ===============================================================
st.sidebar.subheader("🧪 Data Settings")
N          = st.sidebar.slider("Sample size", 200, 5000, 1500)
noise      = st.sidebar.slider("Noise σ", 0.0, 1.0, 0.3)
test_ratio = st.sidebar.slider("Test ratio", 0.05, 0.5, 0.2)
seed       = st.sidebar.number_input("Seed", value=0, step=1)

# ===============================================================
# ARCHITECTURE
# ===============================================================
st.sidebar.header("🧠 Architecture")

arch_mode = st.sidebar.radio("Architecture Mode", ["Preset", "Custom"], horizontal=True)

preset_arch = st.sidebar.selectbox(
    "Preset model",
    ["shallow", "deep", "wide", "dropout", "batchnorm"],
    disabled=(arch_mode != "Preset")
)

st.sidebar.subheader("Custom architecture")

num_layers = st.sidebar.slider("Hidden layers", 0, 12, 3, disabled=(arch_mode!="Custom"))

hidden_sizes = []
if arch_mode=="Custom":
    for i in range(num_layers):
        h = st.sidebar.number_input(f"Layer {i+1} neurons", 1, 1024, 64, key=f"layer{i}")
        hidden_sizes.append(int(h))

epochs = st.sidebar.slider("Epochs", 50, 3000, 500, 50)
lr     = st.sidebar.selectbox("Learning Rate", [0.1,0.03,0.01,0.003,0.001], index=2)
use_gpu= st.sidebar.checkbox("Use GPU", value=True)

# ===============================================================
# TARGET FUNCTION
# ===============================================================
def build_target_function():
    if func_choice=="sinusoid":
        return lambda x: amp*torch.sin(freq*x + phase) + bias
    if func_choice=="polynomial":
        return lambda x: c0 + c1*x + c2*x**2 + c3*x**3
    if func_choice=="exponential":
        return lambda x: A*torch.exp(k*x) + b_log
    if func_choice=="logarithmic":
        return lambda x: a*(torch.log(torch.clamp(x,1e-6)) if base_e else torch.log10(torch.clamp(x,1e-6))) + b_log
    if func_choice=="rational":
        return lambda x: p*x/(x**2 + q) + r
    if func_choice=="piecewise (sawtooth-like)":
        return lambda x: slope*(2*torch.abs((torch.remainder(x,period)/period)-0.5)-0.5) + bias_pw

target_fn = build_target_function()

# ===============================================================
# MODEL BUILDER + ARCH DESCRIPTOR
# ===============================================================
def build_model():
    if arch_mode=="Preset":
        name = preset_arch
        if name=="shallow":
            desc = [{"type":"input","units":1},
                    {"type":"dense","units":32,"act":"ReLU"},
                    {"type":"output","units":1}]
            model = nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,1))

        elif name=="deep":
            desc=[{"type":"input","units":1}]
            layers=[]
            prev=1
            for _ in range(3):
                layers += [nn.Linear(prev,64), nn.ReLU()]
                desc.append({"type":"dense","units":64,"act":"ReLU"})
                prev=64
            layers.append(nn.Linear(prev,1))
            desc.append({"type":"output","units":1})
            model = nn.Sequential(*layers)

        elif name=="wide":
            model = nn.Sequential(
                nn.Linear(1,256), nn.ReLU(),
                nn.Linear(256,256), nn.ReLU(),
                nn.Linear(256,1)
            )
            desc=[
                {"type":"input","units":1},
                {"type":"dense","units":256,"act":"ReLU"},
                {"type":"dense","units":256,"act":"ReLU"},
                {"type":"output","units":1},
            ]

        elif name=="dropout":
            model = nn.Sequential(
                nn.Linear(1,128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,1)
            )
            desc=[
                {"type":"input","units":1},
                {"type":"dense","units":128,"act":"ReLU"},
                {"type":"dropout","p":0.3},
                {"type":"dense","units":128,"act":"ReLU"},
                {"type":"dropout","p":0.3},
                {"type":"output","units":1},
            ]

        elif name=="batchnorm":
            model = nn.Sequential(
                nn.Linear(1,64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64,64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64,1)
            )
            desc=[
                {"type":"input","units":1},
                {"type":"dense","units":64,"norm":"BN","act":"ReLU"},
                {"type":"dense","units":64,"norm":"BN","act":"ReLU"},
                {"type":"output","units":1},
            ]

        return model, desc, name

    # CUSTOM ARCHITECTURE
    desc=[{"type":"input","units":1}]
    layers=[]
    prev=1
    for h in hidden_sizes:
        layers += [nn.Linear(prev,h), nn.ReLU()]
        desc.append({"type":"dense","units":h,"act":"ReLU"})
        prev=h
    layers.append(nn.Linear(prev,1))
    desc.append({"type":"output","units":1})
    return nn.Sequential(*layers), desc, "custom"


# ===============================================================
# HTML ARCHITECTURE VISUALIZATION
# ===============================================================
def render_architecture(desc, title):
    html=f"<div style='font-weight:600;margin-bottom:8px'>{title}</div>"
    html+="<div style='display:flex;gap:12px;overflow-x:auto;padding:10px;border:1px solid #eee;border-radius:8px;background:#fafafa'>"

    for i,layer in enumerate(desc):
        if layer["type"]=="input":
            header="Input"
            sub=f"{layer['units']} feature"
        elif layer["type"]=="output":
            header="Output"
            sub=f"{layer['units']} target"
        elif layer["type"]=="dense":
            header="Dense"
            sub=f"{layer['units']}u · {layer.get('act','')}"
        elif layer["type"]=="dropout":
            header="Dropout"
            sub=f"p={layer['p']}"

        html+=f"""
        <div style='min-width:130px;border:1px solid #ddd;border-radius:10px;
                    background:white;padding:10px;text-align:center;
                    box-shadow:0 2px 4px rgba(0,0,0,0.05)'>
            <div style='font-weight:700'>{header}</div>
            <div style='color:#777;font-size:13px'>{sub}</div>
        </div>
        """
        if i < len(desc)-1:
            html+="<div style='font-size:22px;color:#aaa;display:flex;align-items:center'>→</div>"

    html+="</div>"
    return html

# ===============================================================
# SESSION STATE
# ===============================================================
if "trained" not in st.session_state:
    st.session_state.trained=False
    st.session_state.pred=None
    st.session_state.mse=None
    st.session_state.losses=None
    st.session_state.arch=None
    st.session_state.cfg=None

def capture_cfg():
    return (func_choice,xmin,xmax,N,noise,test_ratio,seed)

new_cfg=capture_cfg()
if st.session_state.cfg!=new_cfg:
    st.session_state.trained=False
    st.session_state.cfg=new_cfg

# ===============================================================
# GENERATE DATA
# ===============================================================
torch.manual_seed(seed)
np.random.seed(seed)

X = torch.linspace(xmin, xmax, N).unsqueeze(1)
y_clean = target_fn(X)
y = y_clean + noise*torch.randn_like(y_clean)

perm=torch.randperm(N)
X, y = X[perm], y[perm]

split=int((1-test_ratio)*N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================================================
# ALWAYS SHOW DATA + TRUE FUNCTION
# ===============================================================
X_line = torch.linspace(xmin, xmax, 800).unsqueeze(1)
y_true = target_fn(X_line).numpy()

col_left, col_right = st.columns([1.6,1])

with col_left:
    fig,ax=plt.subplots(figsize=(10,6))
    ax.scatter(X_train,y_train,c="#1f77b4",alpha=0.3,label="Train")
    ax.scatter(X_test ,y_test ,c="#ff7f0e",alpha=0.4,label="Test")
    ax.plot(X_line,y_true,c="#2ca02c",lw=2,label="True")

    if st.session_state.trained and st.session_state.pred is not None:
        ax.plot(X_line, st.session_state.pred, c="#d62728", lw=2.2,
                label=f"Learned ({st.session_state.arch})")

    ax.legend()
    ax.set_title(f"Data + True Function — {func_choice}")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

# ===============================================================
# RIGHT COLUMN — ARCH + RESULTS
# ===============================================================
model, desc, arch_name = build_model()
st.components.v1.html(render_architecture(desc, f"Architecture • {arch_name}"), height=240)

with col_right:
    if st.session_state.trained:
        st.subheader("📊 Test MSE")
        st.metric("MSE", f"{st.session_state.mse:.6f}")

        fig2,ax2=plt.subplots(figsize=(6,3))
        ax2.plot(st.session_state.losses)
        ax2.set_title("Training Loss")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Click **Train Model** to learn the function.")

# ===============================================================
# TRAIN BUTTON
# ===============================================================
if st.sidebar.button("🚀 Train Model"):
    device=torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model=model.to(device)

    Xtr,ytr=X_train.to(device),y_train.to(device)
    Xte,yte=X_test.to(device),y_test.to(device)

    criterion=nn.MSELoss()
    optimzr=optim.Adam(model.parameters(),lr=lr)

    losses=[]
    for _ in range(epochs):
        model.train()
        optimzr.zero_grad()
        pred=model(Xtr)
        loss=criterion(pred,ytr)
        loss.backward()
        optimzr.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        mse=criterion(model(Xte),yte).item()
        pred_line=model(X_line.to(device)).cpu().numpy().squeeze()

    st.session_state.trained=True
    st.session_state.pred=pred_line
    st.session_state.mse=mse
    st.session_state.losses=losses
    st.session_state.arch=arch_name

    st.rerun()
# ===============================================================
st.set_page_config(page_title="DL Regression Playground", page_icon="🧪", layout="wide")
