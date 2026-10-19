"""Cost function + gradient descent + decision boundary, from scratch (syllabus: 'Cost function, Optimization, Decision boundary')."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features.build_features import labelled, NUM


def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def cost(w, X, y): p = sigmoid(X @ w); return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))   # binary cross-entropy


def gd(X, y, lr, iters=400):
    w, hist = np.zeros(X.shape[1]), []
    for _ in range(iters):
        w -= lr * X.T @ (sigmoid(X @ w) - y) / len(y)        # gradient of the cost
        hist.append(cost(w, X, y))
    return w, hist


def main():
    df = labelled()
    Xs = StandardScaler().fit_transform(df[NUM]); X = np.c_[np.ones(len(Xs)), Xs]
    y = (df.difficulty == "Easy").astype(float).values          # task: Easy vs not-Easy
    plt.figure(figsize=(6, 4))
    for lr in [0.01, 0.1, 0.5, 2.0]:
        w, h = gd(X, y, lr); plt.plot(h, label=f"lr={lr}")
    plt.xlabel("iteration"); plt.ylabel("cost (cross-entropy)"); plt.legend(); plt.title("Gradient descent: learning-rate effect")
    plt.tight_layout(); plt.savefig("report/gd_cost_curve.png", dpi=140); plt.close()
    w, h = gd(X, y, 0.5, 2000)
    sk = LogisticRegression(C=1e6, max_iter=5000).fit(Xs, y)
    acc = lambda p: float(((p > .5) == y).mean())
    print(f"final cost {h[-1]:.4f} | my GD accuracy {acc(sigmoid(X @ w)):.3f} | sklearn accuracy {sk.score(Xs, y):.3f}")
    print("coefficient correlation (mine vs sklearn):", round(float(np.corrcoef(w[1:], sk.coef_[0])[0, 1]), 3))
    z = PCA(2).fit_transform(Xs); lr2 = LogisticRegression(class_weight="balanced").fit(z, y)
    gx, gy = np.meshgrid(np.linspace(z[:, 0].min() - 1, z[:, 0].max() + 1, 200), np.linspace(z[:, 1].min() - 1, z[:, 1].max() + 1, 200))
    pr = lr2.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
    plt.figure(figsize=(5.5, 4.5)); plt.contourf(gx, gy, pr, levels=[0, .5, 1], alpha=.25, colors=["tab:red", "tab:green"])
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap="RdYlGn", s=10, edgecolors="none"); plt.title("Decision boundary (Easy vs not) in PCA space")
    plt.tight_layout(); plt.savefig("report/decision_boundary.png", dpi=140); plt.close()
    print("saved report/gd_cost_curve.png and report/decision_boundary.png")
