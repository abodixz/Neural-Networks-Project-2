"""Oscillation index of validation loss per optimiser (mean |delta val loss| between consecutive epochs, first 50 epochs)."""
import json, numpy as np, pandas as pd
g = json.load(open("results/grid.json")); rows = []
for r in g:
    v = np.array(r["val_loss_curve"][:50]); t = np.array(r["train_loss_curve"][:50])
    rows.append(dict(opt=r["opt"], depth=r["depth"], width=r["width"], osc_val=float(np.mean(np.abs(np.diff(v)))),
                     osc_train=float(np.mean(np.abs(np.diff(t)))), spikes=int(np.sum(v[1:] > 1.5 * np.minimum.accumulate(v)[:-1])),
                     train_loss_ep10=float(t[min(9, len(t)-1)])))
df = pd.DataFrame(rows); s = df.groupby("opt")[["osc_val", "osc_train", "spikes", "train_loss_ep10"]].mean()
s2 = df.groupby(["opt", "width"])[["osc_val", "spikes"]].mean().unstack()
print(s.round(4)); print(s2.round(4)); s.to_csv("results/oscillation_by_optimizer.csv")
