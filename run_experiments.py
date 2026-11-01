"""Runs all experiments for the paper. Usage: python run_experiments.py [stage ...]
All results are written to results/*.json. Seeds are fixed and recorded per run."""
import sys, json, itertools, os, time
import numpy as np, torch
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from common import *

R = os.path.join(HERE, "results")
SEEDS = [0, 1, 2, 3, 4]
OPTS = ["sgd", "sgd_momentum", "sgd_nesterov_onecycle", "adam"]
DEPTHS = [1, 2, 3, 4]
WIDTHS = [16, 64, 256]

def save(name, obj):
    with open(os.path.join(R, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)

def data(enc="raw", dedup=False):
    X, y, _ = load_data(dedup); s = split(X, y); e = Encoder(enc).fit(s["Xtr"])
    return {k: (e.transform(v) if k.startswith("X") else v) for k, v in s.items()}

def rf():
    return RandomForestClassifier(n_estimators=164, min_samples_split=2, min_samples_leaf=1, n_jobs=1, random_state=0)

# ---------- stage: baselines (tree / linear sanity checks on the same split) ----------
def baselines():
    X, y, _ = load_data(); s = split(X, y); out = {}
    models = {"RandomForest": rf(), "HistGBM": HistGradientBoostingClassifier(random_state=0),
              "LogisticRegression": LogisticRegression(max_iter=2000)}
    for n, m in models.items():
        t0 = time.time(); m.fit(s["Xtr_full"], s["ytr_full"])
        out[n] = dict(test=metrics(s["yte"], m.predict_proba(s["Xte"])[:, 1]), fit_time_s=time.time() - t0)
        m.fit(s["Xtr"], s["ytr"]); out[n]["val"] = metrics(s["yva"], m.predict_proba(s["Xva"])[:, 1])
        cv = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(s["Xtr_full"], s["ytr_full"]):
            m.fit(s["Xtr_full"][tr], s["ytr_full"][tr]); cv.append(metrics(s["ytr_full"][te], m.predict_proba(s["Xtr_full"][te])[:, 1])["f1"])
        out[n]["cv_f1"] = cv
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "phishing_websites.csv"))
    feats = df.drop(columns=["result"])
    out["data"] = dict(n=len(df), n_features=feats.shape[1], n_phishing=int((df.result == -1).sum()),
                       n_legit=int((df.result == 1).sum()), exact_duplicate_rows=int(df.duplicated().sum()),
                       unique_rows=int(len(df.drop_duplicates())), unique_feature_vectors=int(len(feats.drop_duplicates())),
                       conflicting_feature_vectors=int((df.drop_duplicates().groupby(list(feats.columns)).size() > 1).sum()),
                       levels_per_feature={c: sorted(map(int, feats[c].unique())) for c in feats.columns})
    Xtr_set = set(map(tuple, s["Xtr_full"])); out["data"]["test_rows_with_identical_train_vector"] = int(sum(tuple(r) in Xtr_set for r in s["Xte"]))
    save("baselines.json", out); print(json.dumps({k: v for k, v in out.items() if k != "data"}, indent=1, default=float)); print(out["data"])

# ---------- stage: grid over optimizer x depth x width (raw coding) ----------
def _grid_job(args):
    torch.set_num_threads(1)
    opt, depth, width, seed, enc = args
    d = data(enc)
    m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=width, opt=opt, seed=seed)
    return dict(opt=opt, depth=depth, width=width, seed=seed, enc=enc, **info,
                val=metrics(d["yva"], predict(m, d["Xva"])), train=metrics(d["ytr"], predict(m, d["Xtr"])),
                val_loss_curve=h["val_loss"], train_loss_curve=h["train_loss"])

def grid():
    jobs = list(itertools.product(OPTS, DEPTHS, WIDTHS, SEEDS, ["raw"]))
    with Pool(6) as p: res = p.map(_grid_job, jobs, chunksize=1)
    save("grid.json", res)

def encodings():
    cfgs = [("adam", 2, 64), ("adam", 3, 256), ("sgd_momentum", 2, 64), ("sgd_nesterov_onecycle", 3, 64)]
    jobs = [(o, dp, w, s, e) for (o, dp, w) in cfgs for s in SEEDS for e in ["raw", "standard", "onehot"]]
    with Pool(6) as p: res = p.map(_grid_job, jobs, chunksize=1)
    for r in res: r.pop("val_loss_curve"); r.pop("train_loss_curve")
    save("encodings.json", res)

# ---------- stage: overfitting without early stopping ----------
def _of_job(args):
    torch.set_num_threads(1)
    depth, width, dropout, wd, seed = args
    d = data("raw")
    m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=width, opt="adam",
                       dropout=dropout, weight_decay=wd, seed=seed, max_epochs=300, early_stop=False)
    return dict(depth=depth, width=width, dropout=dropout, weight_decay=wd, seed=seed, **info, hist=h)

def overfit():
    jobs = [(1, 16, 0, 0, 0), (3, 256, 0, 0, 0), (3, 256, 0.3, 1e-4, 0), (4, 256, 0, 0, 0)]
    with Pool(4) as p: res = p.map(_of_job, jobs)
    save("overfit.json", res)

# ---------- stage: vanishing gradients ----------
def _vg_job(args):
    torch.set_num_threads(1)
    act, depth, opt = args
    d = data("raw")
    m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=64, opt=opt, act=act,
                       seed=0, max_epochs=150, early_stop=False, grad_log=True)
    return dict(act=act, depth=depth, opt=opt, **info, val=metrics(d["yva"], predict(m, d["Xva"])), hist=h)

def vanishing():
    jobs = [(a, dp, o) for a in ["sigmoid", "tanh", "relu"] for dp in [2, 10] for o in ["sgd_momentum", "adam"]]
    with Pool(6) as p: res = p.map(_vg_job, jobs)
    save("vanishing.json", res)

# ---------- stage: local optima (many restarts + linear interpolation) ----------
def local_optima(cfg):
    torch.set_num_threads(4)
    d = data("raw"); opt, depth, width, dropout, wd = cfg; runs, models = [], []
    for s in range(20):
        m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=100 + s)
        runs.append(dict(seed=100 + s, **info, val=metrics(d["yva"], predict(m, d["Xva"])))); models.append(m)
    lossf = torch.nn.BCEWithLogitsLoss(); interp = []
    for (a, b) in [(0, 1), (0, 2), (1, 2)]:
        sa, sb = models[a].state_dict(), models[b].state_dict(); mm = make_mlp(d["Xtr"].shape[1], depth, width, dropout=dropout)
        curve = []
        for t in np.linspace(0, 1, 21):
            mm.load_state_dict({k: (1 - t) * sa[k] + t * sb[k] for k in sa})
            curve.append(dict(t=float(t), train_loss=eval_loss(mm, d["Xtr"], d["ytr"], lossf), val_loss=eval_loss(mm, d["Xva"], d["yva"], lossf)))
        interp.append(dict(pair=[a, b], curve=curve))
    save("local_optima.json", dict(config=cfg, runs=runs, interpolation=interp))

# ---------- stage: final model on test (+ repeated seeds, + 3-fold CV like Project 1) ----------
def final(cfg):
    torch.set_num_threads(4)
    opt, depth, width, dropout, wd = cfg; d = data("raw"); out = dict(config=cfg, test_runs=[])
    for s in range(10):
        m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=s)
        p = predict(m, d["Xte"])
        out["test_runs"].append(dict(seed=s, **info, test=metrics(d["yte"], p), probs=p.tolist() if s == 0 else None,
                                     hist=h if s == 0 else None))
    out["y_test"] = d["yte"].tolist()
    r = rf().fit(d["Xtr_full"], d["ytr_full"]); out["rf_test_probs"] = r.predict_proba(d["Xte"])[:, 1].tolist()
    cv = []
    for k, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(d["Xtr_full"], d["ytr_full"])):
        Xa, Xv, ya, yv = train_test_split(d["Xtr_full"][tr], d["ytr_full"][tr], test_size=1/9, stratify=d["ytr_full"][tr], random_state=k)
        m, h, info = train(Xa, ya, Xv, yv, depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=k)
        cv.append(dict(fold=k, **info, test=metrics(d["ytr_full"][te], predict(m, d["Xtr_full"][te]))))
    out["cv"] = cv
    save("final.json", out)

# ---------- stage: misclassification costs ----------
def costs(cfg):
    torch.set_num_threads(4)
    opt, depth, width, dropout, wd = cfg; d = data("raw"); out = []
    for pw in [0.5, 1, 2, 5, 10]:
        for s in range(5):
            m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=s, pos_weight=pw)
            out.append(dict(pos_weight=pw, seed=s, best_epoch=info["best_epoch"], val=metrics(d["yva"], predict(m, d["Xva"])),
                            test=metrics(d["yte"], predict(m, d["Xte"])),
                            test_probs=predict(m, d["Xte"]).tolist() if pw == 1 else None,
                            val_probs=predict(m, d["Xva"]).tolist() if pw == 1 else None))
    save("costs.json", dict(config=cfg, runs=out, y_test=d["yte"].tolist(), y_val=d["yva"].tolist()))

# ---------- stage: training-set size (as Project 1 Table 3) ----------
def _size_job(args):
    torch.set_num_threads(1)
    frac, seed, cfg = args; opt, depth, width, dropout, wd = cfg; d = data("raw")
    if frac < 1:
        idx, _ = train_test_split(np.arange(len(d["ytr"])), train_size=frac, stratify=d["ytr"], random_state=seed)
    else:
        idx = np.arange(len(d["ytr"]))
    m, h, info = train(d["Xtr"][idx], d["ytr"][idx], d["Xva"], d["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=seed)
    r = rf().set_params(random_state=seed).fit(d["Xtr"][idx], d["ytr"][idx])
    return dict(frac=frac, seed=seed, n_train=int(len(idx)), best_epoch=info["best_epoch"],
                nn=metrics(d["yte"], predict(m, d["Xte"])), rf=metrics(d["yte"], r.predict_proba(d["Xte"])[:, 1]))

def datasize(cfg):
    jobs = [(f, s, cfg) for f in [0.05, 0.25, 0.5, 1.0] for s in SEEDS]
    with Pool(6) as p: res = p.map(_size_job, jobs)
    save("datasize.json", res)

# ---------- stage: missing values (MCAR in train/val/test) ----------
def _miss_job(args):
    torch.set_num_threads(1)
    rate, seed, cfg = args; opt, depth, width, dropout, wd = cfg
    X, y, _ = load_data(); s = split(X, y); rng = np.random.default_rng(seed); out = dict(rate=rate, seed=seed)
    masks = {k: rng.random(s[k].shape) < rate for k in ["Xtr", "Xva", "Xte", "Xtr_full"]}
    enc = lambda k: np.concatenate([np.where(masks[k], 0, s[k]), masks[k]], 1).astype(np.float32)
    m, h, info = train(enc("Xtr"), s["ytr"], enc("Xva"), s["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=seed)
    out["nn_indicator"] = metrics(s["yte"], predict(m, enc("Xte")))
    enc0 = lambda k: np.where(masks[k], 0, s[k]).astype(np.float32)
    m, h, info = train(enc0("Xtr"), s["ytr"], enc0("Xva"), s["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=seed)
    out["nn_zero"] = metrics(s["yte"], predict(m, enc0("Xte")))
    nan = lambda k: np.where(masks[k], np.nan, s[k].astype(float))
    r = rf().set_params(random_state=seed).fit(nan("Xtr_full"), s["ytr_full"])
    out["rf_nan"] = metrics(s["yte"], r.predict_proba(nan("Xte"))[:, 1])
    return out

def missing(cfg):
    jobs = [(r, s, cfg) for r in [0.1, 0.3, 0.5, 0.8] for s in [0, 1, 2]]
    with Pool(6) as p: res = p.map(_miss_job, jobs)
    save("missing.json", res)

# ---------- stage: de-duplicated dataset ----------
def dedup(cfg):
    torch.set_num_threads(4)
    opt, depth, width, dropout, wd = cfg; X, y, _ = load_data(dedup=True); s = split(X, y); out = dict(n=int(len(y)), nn=[], rf=[])
    for sd in SEEDS:
        m, h, info = train(s["Xtr"].astype(np.float32), s["ytr"], s["Xva"].astype(np.float32), s["yva"], depth=depth, width=width, opt=opt, dropout=dropout, weight_decay=wd, seed=sd)
        out["nn"].append(metrics(s["yte"], predict(m, s["Xte"].astype(np.float32))))
        r = rf().set_params(random_state=sd).fit(s["Xtr_full"], s["ytr_full"]); out["rf"].append(metrics(s["yte"], r.predict_proba(s["Xte"])[:, 1]))
    save("dedup.json", out)

# ---------- stage: regularisation search on validation (dropout x weight decay), Adam ----------
def _reg_job(args):
    torch.set_num_threads(1)
    depth, dropout, wd, seed = args
    d = data("raw")
    m, h, info = train(d["Xtr"], d["ytr"], d["Xva"], d["yva"], depth=depth, width=256, opt="adam", dropout=dropout, weight_decay=wd, seed=seed)
    return dict(opt="adam", depth=depth, width=256, dropout=dropout, weight_decay=wd, seed=seed, **info,
                val=metrics(d["yva"], predict(m, d["Xva"])), train=metrics(d["ytr"], predict(m, d["Xtr"])))

def regularization():
    jobs = [(dp, do, wd, s) for dp in [1, 2, 3] for do in [0.2, 0.3, 0.5] for wd in [0.0, 1e-4] for s in SEEDS]
    with Pool(6) as p: res = p.map(_reg_job, jobs, chunksize=1)
    save("regularization.json", res)

if __name__ == "__main__":
    cfg = json.load(open(os.path.join(R, "selected_config.json")))["config"] if os.path.exists(os.path.join(R, "selected_config.json")) else None
    for st in sys.argv[1:]:
        t0 = time.time(); print("== stage", st, flush=True)
        fn = globals()[st]
        fn(tuple(cfg)) if fn.__code__.co_argcount == 1 else fn()
        print("   done in %.1fs" % (time.time() - t0), flush=True)
