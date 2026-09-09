"""
Small Feedforward Neural Network (2 inputs -> hidden layer -> 1 output)
trained with manual gradient descent.

Instead of calling loss.backward() and letting torch.optim update the
weights, we:
  1. Let autograd compute dLoss/dW, dLoss/db for every parameter (the
     "derivatives") via loss.backward().
  2. Read those derivatives from .grad and apply the gradient descent
     update rule ourselves: theta = theta - lr * dLoss/dtheta
"""

import torch

torch.manual_seed(0)

# ---------------------------------------------------------------
# 1. Toy data: 2 continuous inputs -> 1 continuous output
#    (target function: y = 3*x1 - 2*x2 + 1, plus a little noise)
# ---------------------------------------------------------------
n_samples = 200
X = torch.randn(n_samples, 2)
true_w = torch.tensor([3.0, -2.0])
y = X @ true_w + 1.0 + 0.1 * torch.randn(n_samples)
y = y.unsqueeze(1)  # shape (n_samples, 1)

# ---------------------------------------------------------------
# 2. Parameters of a single hidden layer network
#    input(2) -> hidden(H) -> output(1)
#    Each parameter has requires_grad=True so autograd tracks it
#    and computes its derivative w.r.t. the loss.
# ---------------------------------------------------------------
H = 8  # hidden units
lr = 0.05
epochs = 300

# Same random init used for BOTH runs, so the comparison is fair.
init_W1 = torch.randn(2, H) * 0.5
init_b1 = torch.zeros(H)
init_W2 = torch.randn(H, 1) * 0.5
init_b2 = torch.zeros(1)


def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1          # linear
    a1 = torch.tanh(z1)       # nonlinearity
    z2 = a1 @ W2 + b2         # linear -> continuous output
    return z2


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


# =================================================================
# 3a. RUN 1 — manual gradient descent (no torch.optim)
# =================================================================
W1 = init_W1.clone().requires_grad_()
b1 = init_b1.clone().requires_grad_()
W2 = init_W2.clone().requires_grad_()
b2 = init_b2.clone().requires_grad_()
params = [W1, b1, W2, b2]

manual_losses = []
for epoch in range(epochs):
    pred = forward(X, W1, b1, W2, b2)
    loss = mse_loss(pred, y)

    for p in params:                 # clear old derivatives
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()                  # autograd computes dLoss/dp

    with torch.no_grad():            # manual update rule
        for p in params:
            p -= lr * p.grad         # theta_new = theta_old - lr * dLoss/dtheta

    manual_losses.append(loss.item())
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"[manual] epoch {epoch:4d} | loss = {loss.item():.5f} "
              f"| |dLoss/dW1| = {W1.grad.norm().item():.4f} "
              f"| |dLoss/dW2| = {W2.grad.norm().item():.4f}")

# =================================================================
# 3b. RUN 2 — identical network, but updated via torch.optim.SGD
# =================================================================
W1o = init_W1.clone().requires_grad_()
b1o = init_b1.clone().requires_grad_()
W2o = init_W2.clone().requires_grad_()
b2o = init_b2.clone().requires_grad_()
params_o = [W1o, b1o, W2o, b2o]

optimizer = torch.optim.SGD(params_o, lr=lr)  # plain SGD == same math as above

optim_losses = []
for epoch in range(epochs):
    pred = forward(X, W1o, b1o, W2o, b2o)
    loss = mse_loss(pred, y)

    optimizer.zero_grad()   # same role as the manual p.grad.zero_() loop
    loss.backward()         # autograd computes dLoss/dp, same as run 1
    optimizer.step()        # same role as the manual "p -= lr * p.grad" loop

    optim_losses.append(loss.item())
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"[optim ] epoch {epoch:4d} | loss = {loss.item():.5f} "
              f"| |dLoss/dW1| = {W1o.grad.norm().item():.4f} "
              f"| |dLoss/dW2| = {W2o.grad.norm().item():.4f}")

# =================================================================
# 4. Compare the two training runs
# =================================================================
print("\n--- Comparison ---")
print(f"Manual GD  final loss: {manual_losses[-1]:.6f}")
print(f"torch.optim final loss: {optim_losses[-1]:.6f}")
max_diff = max(abs(m - o) for m, o in zip(manual_losses, optim_losses))
print(f"Max |loss difference| across all {epochs} epochs: {max_diff:.2e}")
print("(With plain SGD and identical init/data, the two curves should be "
      "numerically identical up to floating point rounding: torch.optim.SGD "
      "with default settings performs exactly the update p -= lr * p.grad.)")
