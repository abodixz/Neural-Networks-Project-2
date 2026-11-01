import json, pandas as pd, numpy as np
g = json.load(open("results/grid.json"))
rows = [dict(opt=r["opt"], depth=r["depth"], width=r["width"], seed=r["seed"], val_loss=r["best_val_loss"],
             val_acc=r["val"]["accuracy"], val_f1=r["val"]["f1"], val_auc=r["val"]["roc_auc"], train_acc=r["train"]["accuracy"],
             train_loss=r["train_loss_at_best"], best_ep=r["best_epoch"], ep=r["epochs_run"], t=r["time_s"], params=r["n_params"]) for r in g]
df = pd.DataFrame(rows); df.to_csv("results/grid_runs.csv", index=False)
agg = df.groupby(["opt","depth","width"]).agg(val_loss=("val_loss","mean"), val_loss_sd=("val_loss","std"), val_acc=("val_acc","mean"),
      val_acc_sd=("val_acc","std"), val_f1=("val_f1","mean"), train_acc=("train_acc","mean"), best_ep=("best_ep","mean"),
      ep=("ep","mean"), t=("t","mean"), params=("params","first")).reset_index()
agg["gap"] = agg.train_acc - agg.val_acc
agg.to_csv("results/grid_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(agg.sort_values("val_loss").round(4).to_string())
print(df.groupby("opt")[["val_loss","val_acc","best_ep","t"]].mean().round(4))
print(df.groupby("depth")[["val_loss","val_acc","train_acc","best_ep"]].mean().round(4))
print(df.groupby("width")[["val_loss","val_acc","train_acc","best_ep"]].mean().round(4))
