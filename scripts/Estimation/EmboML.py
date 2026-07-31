import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# ============================================================
# 1. Synthetic data from a latent Gaussian model
# ============================================================

N = 512
true_w, true_b = 2.0, -1.0
sigma_x = torch.tensor(0.3)

z_true = torch.randn(N, 1)
x = true_w * z_true + true_b + sigma_x * torch.randn(N, 1)

# ============================================================
# 2. Generative model p_theta(x|z)
# ============================================================

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, z):
        return self.linear(z)  # mean of p(x|z)

# ============================================================
# 3. Variational encoder q_phi(z|x)
# ============================================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2)  # outputs [mu, log_var]
        )

    def forward(self, x):
        stats = self.net(x)
        mu = stats[:, :1]
        log_var = stats[:, 1:]
        return mu, log_var

dec = Decoder()
enc = Encoder()

opt = optim.Adam(list(dec.parameters()) + list(enc.parameters()), lr=1e-2)

# ============================================================
# 4. Helper: sample z using the reparameterization trick
# ============================================================

def sample_q(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps

# ============================================================
# 5. ELBO Formulation A: Reconstruction - KL
# ============================================================

def elbo_recon_kl(x_batch):
    mu_z, log_var_z = enc(x_batch)
    z = sample_q(mu_z, log_var_z)

    # log p(x|z)
    x_mean = dec(z)
    log_px_given_z = -0.5 * (
        (x_batch - x_mean)**2 / sigma_x**2 +
        torch.log(2 * torch.pi * sigma_x**2)
    ).sum(dim=1)

    # KL(q||p)
    kl = 0.5 * (torch.exp(log_var_z) + mu_z**2 - 1 - log_var_z).sum(dim=1)

    return (log_px_given_z - kl).mean()

# ============================================================
# 6. ELBO Formulation B: Joint log - entropy
# ============================================================

def elbo_joint_entropy(x_batch):
    mu_z, log_var_z = enc(x_batch)
    z = torch.tensor(sample_q(mu_z, log_var_z))

    # log p(x|z)
    x_mean = dec(z)
    log_px_given_z = -0.5 * (
        (x_batch - x_mean)**2 / sigma_x**2 +
        torch.log(2 * torch.pi * sigma_x**2)
    ).sum(dim=1)

    # log p(z)
    log_pz = -0.5 * (z**2 + torch.log(2 * torch.tensor(torch.pi))).sum(dim=1)

    # log q(z|x)
    log_qz_given_x = -0.5 * (
        (z - mu_z)**2 / torch.exp(log_var_z) +
        log_var_z +
        torch.log(2 * torch.tensor(torch.pi))
    ).sum(dim=1)

    return (log_px_given_z + log_pz - log_qz_given_x).mean()

# ============================================================
# 7. ELBO Formulation C: log p(x,z) - log q(z|x)
#    (Variational Free Energy)
# ============================================================

def elbo_free_energy(x_batch):
    mu_z, log_var_z = enc(x_batch)
    z = sample_q(mu_z, log_var_z)

    # log p(x|z)
    x_mean = dec(z)
    log_px_given_z = -0.5 * (
        (x_batch - x_mean)**2 / sigma_x**2 +
        torch.log(2 * torch.pi * sigma_x**2)
    ).sum(dim=1)

    # log p(z)
    log_pz = -0.5 * (z**2 + torch.log(2 * torch.pi)).sum(dim=1)

    # log q(z|x)
    log_qz_given_x = -0.5 * (
        (z - mu_z)**2 / torch.exp(log_var_z) +
        log_var_z +
        torch.log(2 * torch.pi)
    ).sum(dim=1)

    # ELBO = E_q[log p(x,z) - log q(z|x)]
    return (log_px_given_z + log_pz - log_qz_given_x).mean()

# ============================================================
# 8. Training loop (choose any ELBO)
# ============================================================

for epoch in range(5000):
    opt.zero_grad()

    # Choose one of the three:
    #elbo = elbo_recon_kl(x)
    elbo = elbo_joint_entropy(x)
    # elbo = elbo_free_energy(x)

    loss = -elbo
    loss.backward()
    opt.step()

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}, -ELBO: {loss.item():.4f}")

# ============================================================
# 9. Inspect learned parameters (MLE estimate)
# ============================================================

w_learned = dec.linear.weight.item()
b_learned = dec.linear.bias.item()

print(f"True w={true_w:.3f}, learned w={w_learned:.3f}")
print(f"True b={true_b:.3f}, learned b={b_learned:.3f}")
