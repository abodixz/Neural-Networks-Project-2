"""Held-out test evaluation of reference configurations (selection was done on validation only).
10 seeds each; RF / HistGBM / LogReg as non-neural references on the identical split."""
import json, os, numpy as np, torch
from multiprocessing import Pool
from common import *

CFGS = {"MLP 1x16 (shallow baseline, Adam)": ("adam", 1, 16, 0.0, 0.0),
        "MLP 1x256 (best unregularised, Adam)": ("adam", 1, 256, 0.0, 0.0),
        "MLP 2x256 (SGD Nesterov + one-cycle)": ("sgd_nesterov_onecycle", 2, 256, 0.0, 0.0),
        "MLP 4x256 (SGD + momentum)": ("sgd_momentum", 4, 256, 0.0, 0.0),
        "MLP 3x256 + dropout 0.3, wd 1e-4 (selected, Adam)": ("adam", 3, 256, 0.3, 1e-4)}

def job(args):
    torch.set_num_threads(1)
    name, seed = args; opt, depth, width, do, wd = CFGS[name]
    X, y, _ = load_data(); s = split(X, y); f = lambda a: a.astype(np.float32)
    m, h, info = train(f(s["Xtr"]), s["ytr"], f(s["Xva"]), s["yva"], depth=depth, width=width, opt=opt, dropout=do, weight_decay=wd, seed=seed)
    return dict(name=name, seed=seed, cfg=CFGS[name], **info, test=metrics(s["yte"], predict(m, f(s["Xte"]))))

if __name__ == "__main__":
    jobs = [(n, s) for n in CFGS for s in range(10)]
    with Pool(6) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "test_reference.json"), "w"), indent=1, default=float)
    print("done", len(res))
