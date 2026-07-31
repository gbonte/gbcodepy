import numpy as np
import matplotlib.pyplot as plt

n_train = 40
n_test = 2000
max_features = 2000
n_runs = 100

feature_range = np.arange(1, max_features + 1)
all_test_errors = np.zeros((n_runs, max_features))
all_train_errors = np.zeros((n_runs, max_features))

for run in range(n_runs):
    np.random.seed(run)

    X_train_full = np.random.randn(n_train, max_features)
    X_test_full = np.random.randn(n_test, max_features)

    true_w = np.zeros(max_features)
    true_w[:5] = np.random.randn(5)

    y_train = X_train_full @ true_w + 0.5 * np.random.randn(n_train)
    y_test = X_test_full @ true_w + 0.5 * np.random.randn(n_test)

    for d in feature_range:
        X_train = X_train_full[:, :d]
        X_test = X_test_full[:, :d]

        # Explicit least squares using pseudoinverse (minimum-norm solution)
        # w = X^+ y
        w_hat = np.linalg.pinv(X_train) @ y_train

        y_train_pred = X_train @ w_hat
        y_test_pred = X_test @ w_hat

        train_err = np.mean((y_train - y_train_pred) ** 2)
        test_err = np.mean((y_test - y_test_pred) ** 2)

        all_train_errors[run, d-1] = train_err
        all_test_errors[run, d-1] = test_err

mean_train_error = all_train_errors.mean(axis=0)
mean_test_error = all_test_errors.mean(axis=0)

plt.figure()
plt.plot(feature_range, mean_train_error, label='Average Train error')
plt.plot(feature_range, mean_test_error, label='Average Test error')
plt.axvline(n_train, linestyle='--', label='Interpolation threshold (d=n_train)')
plt.xlabel('Number of features (model complexity)')
plt.ylabel('Mean squared error')
plt.ylim(0, 10)
plt.title('Double Descent with Explicit Least Squares (pseudoinverse)')
plt.legend()
plt.show()
