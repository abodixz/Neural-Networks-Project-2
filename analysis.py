"""Statistics (confidence intervals, McNemar, seen/unseen split, cost analysis) and figures for the paper."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from common import load_data, split, metrics
R, F = "results", "figures"
L = lambda n: json.load(open(os.path.join(R, n)))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7.5})
out = {}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [c - h, c + h]
def normal_ci(k, n, z=1.96):
    p = k / n; h = z * np.sqrt(p * (1 - p) / n); return [p - h, p + h]

# ---- final model: CI on test error, seeds, CV ----
fin = L("final.json"); yte = np.array(fin["y_test"]); n = len(yte)
p_nn = np.array(fin["test_runs"][0]["probs"]); p_rf = np.array(fin["rf_test_probs"])
e_nn = ((p_nn >= .5) != yte); e_rf = ((p_rf >= .5) != yte)
errs = [r["test"]["error"] for r in fin["test_runs"]]
out["ci"] = dict(n_test=n,
    nn_seed0=dict(errors=int(e_nn.sum()), error=float(e_nn.mean()), wilson95=wilson(e_nn.sum(), n), normal95=normal_ci(e_nn.sum(), n)),
    rf=dict(errors=int(e_rf.sum()), error=float(e_rf.mean()), wilson95=wilson(e_rf.sum(), n), normal95=normal_ci(e_rf.sum(), n)),
    nn_10seeds=dict(mean_error=float(np.mean(errs)), sd=float(np.std(errs, ddof=1)), min=float(min(errs)), max=float(max(errs)),
                    t95_of_mean=[float(np.mean(errs) - stats.t.ppf(.975, 9) * np.std(errs, ddof=1) / np.sqrt(10)),
                                 float(np.mean(errs) + stats.t.ppf(.975, 9) * np.std(errs, ddof=1) / np.sqrt(10))]))
b = int((e_nn & ~e_rf).sum()); c = int((~e_nn & e_rf).sum())
out["mcnemar_nn_seed0_vs_rf"] = dict(nn_wrong_rf_right=b, nn_right_rf_wrong=c, both_wrong=int((e_nn & e_rf).sum()),
                                     exact_p=float(stats.binomtest(b, b + c, .5).pvalue) if b + c else 1.0)
# paired difference CI (normal approx on per-instance difference)
diff = e_nn.astype(float) - e_rf.astype(float)
out["paired_error_diff_nn_minus_rf"] = dict(mean=float(diff.mean()), ci95=[float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(n)), float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(n))])
# seen / unseen test rows (identical feature vector present in training data)
X, y, _ = load_data(); s = split(X, y); seen_set = set(map(tuple, s["Xtr_full"]))
seen = np.array([tuple(r) in seen_set for r in s["Xte"]]); assert (s["yte"] == yte).all()
out["seen_unseen"] = dict(n_seen=int(seen.sum()), n_unseen=int((~seen).sum()),
    nn_err_seen=float(e_nn[seen].mean()), nn_err_unseen=float(e_nn[~seen].mean()),
    rf_err_seen=float(e_rf[seen].mean()), rf_err_unseen=float(e_rf[~seen].mean()),
    nn_err_unseen_wilson=wilson(e_nn[~seen].sum(), (~seen).sum()), rf_err_unseen_wilson=wilson(e_rf[~seen].sum(), (~seen).sum()))
cvf = fin["cv"]; out["nn_cv"] = dict(f1=[c_["test"]["f1"] for c_ in cvf], acc=[c_["test"]["accuracy"] for c_ in cvf])
out["nn_test_mean"] = {k: float(np.mean([r["test"][k] for r in fin["test_runs"]])) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
out["nn_test_sd"] = {k: float(np.std([r["test"][k] for r in fin["test_runs"]], ddof=1)) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
out["rf_test"] = metrics(yte, p_rf)
# legitimate-as-positive F1 (Project 1 convention may have used this)
from sklearn.metrics import f1_score
out["f1_legit_positive"] = dict(nn_seed0=float(f1_score(1 - yte, (p_nn < .5).astype(int))), rf=float(f1_score(1 - yte, (p_rf < .5).astype(int))))

# ---- misclassification costs: C_FN=5, C_FP=1 ----
co = L("costs.json"); CFN, CFP = 5, 1; rows = []
for r in co["runs"]:
    t = r["test"]; rows.append(dict(method=f"pos_weight={r['pos_weight']}", pw=r["pos_weight"], seed=r["seed"], fp=t["fp"], fn=t["fn"],
                                    recall=t["recall"], precision=t["precision"], acc=t["accuracy"], cost=CFN * t["fn"] + CFP * t["fp"]))
yv = np.array(co["y_val"]); yt = np.array(co["y_test"])
for r in co["runs"]:
    if r["pos_weight"] != 1: continue
    pv, pt = np.array(r["val_probs"]), np.array(r["test_probs"])
    ths = np.linspace(0.01, 0.99, 99); vc = [CFN * ((pv < t) & (yv == 1)).sum() + CFP * ((pv >= t) & (yv == 0)).sum() for t in ths]
    for name, th in [("threshold 1/6 (Bayes)", CFP / (CFP + CFN)), ("threshold tuned on val", float(ths[int(np.argmin(vc))]))]:
        m = metrics(yt, pt, th); rows.append(dict(method=name, pw=1, seed=r["seed"], th=th, fp=m["fp"], fn=m["fn"], recall=m["recall"],
                                                  precision=m["precision"], acc=m["accuracy"], cost=CFN * m["fn"] + CFP * m["fp"]))
cdf = pd.DataFrame(rows); out["costs"] = cdf.groupby("method", sort=False)[["fp", "fn", "recall", "precision", "acc", "cost"]].mean().reset_index().to_dict("records")
cdf.to_csv(os.path.join(R, "costs_summary.csv"), index=False)

json.dump(out, open(os.path.join(R, "summary_stats.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# ================= figures =================
g = L("grid.json")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
names = {"sgd": "SGD", "sgd_momentum": "SGD+momentum", "sgd_nesterov_onecycle": "Nesterov+one-cycle", "adam": "Adam"}
for o, col in zip(names, ["C0", "C1", "C2", "C3"]):
    r = [x for x in g if x["opt"] == o and x["depth"] == 2 and x["width"] == 256 and x["seed"] == 0][0]
    ax[0].plot(np.arange(1, len(r["val_loss_curve"]) + 1), r["val_loss_curve"], col, lw=1, label=names[o])
ax[0].set_ylim(0.05, 0.5); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation BCE loss"); ax[0].set_title("(a) Optimisers, 2x256 (val)"); ax[0].legend(loc="upper right", fontsize=6.5, framealpha=1)
of = L("overfit.json")
for r, col, lab in [(of[1], "C3", "3x256, no reg."), (of[2], "C2", "3x256, dropout 0.3 + wd"), (of[0], "C0", "1x16")]:
    h = r["hist"]; ax[1].plot(h["val_loss"], col, lw=1, label=lab); ax[1].plot(h["train_loss"], col, lw=.8, ls=":")
ax[1].set_ylim(0, .4); ax[1].set_xlabel("epoch"); ax[1].set_title("(b) No early stopping, Adam"); ax[1].legend(loc="upper right", fontsize=6.5, framealpha=1)
v = L("vanishing.json")
for a, col in zip(["sigmoid", "tanh", "relu"], ["C0", "C1", "C3"]):
    r = [x for x in v if x["act"] == a and x["depth"] == 10 and x["opt"] == "sgd_momentum"][0]
    gn = np.array(r["hist"]["grad_norms"][0]); ax[2].semilogy(np.arange(1, len(gn) + 1), gn, col + "o-", ms=3, lw=1, label=a)
ax[2].set_xlabel("weight layer (1 = input)"); ax[2].set_ylabel("gradient norm, first batch"); ax[2].set_title("(c) 10 hidden layers"); ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(F, "fig1_training_dynamics.png"), dpi=220); plt.close()

lo = L("local_optima.json"); ds = pd.DataFrame([dict(frac=x["frac"], n=x["n_train"], nn=x["nn"]["f1"], rf=x["rf"]["f1"]) for x in L("datasize.json")])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for it, col in zip(lo["interpolation"], ["C0", "C1", "C2"]):
    t = [p["t"] for p in it["curve"]]; ax[0].plot(t, [p["val_loss"] for p in it["curve"]], col, lw=1, label=f"restarts {it['pair'][0]}-{it['pair'][1]} (val)")
    ax[0].plot(t, [p["train_loss"] for p in it["curve"]], col, lw=.8, ls=":")
ax[0].set_xlabel("interpolation coefficient t"); ax[0].set_ylabel("BCE loss"); ax[0].set_title("(a) Linear path between two minima"); ax[0].legend(frameon=False)
gs = ds.groupby("n").agg(["mean", "std"])
for k, col, lab in [("nn", "C3", "MLP (selected)"), ("rf", "C2", "Random Forest")]:
    ax[1].errorbar(gs.index, gs[k]["mean"], yerr=gs[k]["std"], color=col, marker="o", ms=3, lw=1, capsize=2, label=lab)
ax[1].set_xscale("log"); ax[1].set_xlabel("training instances"); ax[1].set_ylabel("test F1"); ax[1].set_title("(b) Training-set size"); ax[1].legend(frameon=False)
ms = pd.DataFrame([dict(rate=x["rate"], ind=x["nn_indicator"]["f1"], zero=x["nn_zero"]["f1"], rf=x["rf_nan"]["f1"]) for x in L("missing.json")]).groupby("rate").mean()
ax[2].plot(ms.index * 100, ms["ind"], "C3o-", ms=3, lw=1, label="MLP, 0 + indicator"); ax[2].plot(ms.index * 100, ms["zero"], "C1s--", ms=3, lw=1, label="MLP, 0 only")
ax[2].plot(ms.index * 100, ms["rf"], "C2^-", ms=3, lw=1, label="RF, native NaN"); ax[2].set_xlabel("% values missing (MCAR)"); ax[2].set_ylabel("test F1"); ax[2].set_title("(c) Missing values"); ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(F, "fig2_optima_datasize_missing.png"), dpi=220); plt.close()
print("figures written")
