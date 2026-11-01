"""Shared utilities for the neural-net phishing project.
Dataset: UCI Phishing Websites (id 327), 11,055 x 30, features in {-1,0,1}.
Label coding: phishing (result == -1) -> 1 (positive class), legitimate (result == 1) -> 0.
"""
import time, json, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_SEED = 42

def load_data(dedup=False):
    df = pd.read_csv(os.path.join(HERE, "phishing_websites.csv"))
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=["result"]).values.astype(np.int64)
    y = (df["result"].values == -1).astype(np.int64)  # phishing = 1
    return X, y, list(df.columns[:-1])

def split(X, y, seed=SPLIT_SEED):
    """90/10 stratified train/test (as in Project 1), then 1/9 of train -> validation => 80/10/10."""
    Xtr_full, Xte, ytr_full, yte = train_test_split(X, y, test_size=0.10, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_full, ytr_full, test_size=1/9, stratify=ytr_full, random_state=seed)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, Xtr_full=Xtr_full, ytr_full=ytr_full)

class Encoder:
    """raw: keep {-1,0,1}; standard: z-score fitted on train; onehot: one indicator per observed level."""
    def __init__(self, kind="raw"):
        self.kind = kind
    def fit(self, X):
        if self.kind == "standard":
            self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd == 0] = 1
        elif self.kind == "onehot":
            self.levels = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self
    def transform(self, X):
        if self.kind == "raw":
            return X.astype(np.float32)
        if self.kind == "standard":
            return ((X - self.mu) / self.sd).astype(np.float32)
        cols = [(X[:, j:j+1] == lv[None, :]).astype(np.float32) for j, lv in enumerate(self.levels)]
        return np.concatenate(cols, 1)

def make_mlp(n_in, depth, width, act="relu", dropout=0.0):
    A = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[act]
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), A()]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def make_opt(name, params, epochs, steps_per_epoch, weight_decay=0.0):
    sched = None
    if name == "sgd":
        opt = torch.optim.SGD(params, lr=0.05, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=weight_decay)
    elif name == "sgd_nesterov_onecycle":
        opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)
    elif name == "adam":
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=weight_decay)
    else:
        raise ValueError(name)
    return opt, sched

def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat), f1=f1_score(y, yhat), roc_auc=roc_auc_score(y, p),
                error=1 - accuracy_score(y, yhat), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))

@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X)).squeeze(1)).numpy()

@torch.no_grad()
def eval_loss(model, X, y, lossf):
    model.eval()
    return float(lossf(model(torch.from_numpy(X)).squeeze(1), torch.from_numpy(y.astype(np.float32))))

def train(Xtr, ytr, Xva, yva, depth=2, width=64, opt="adam", act="relu", dropout=0.0, weight_decay=0.0,
          max_epochs=300, patience=25, batch=128, seed=0, pos_weight=1.0, early_stop=True, grad_log=False):
    """Mini-batch training with BCE loss; early stopping on validation loss with best-weight restore."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_mlp(Xtr.shape[1], depth, width, act, dropout)
    spe = int(np.ceil(len(Xtr) / batch))
    optim, sched = make_opt(opt, model.parameters(), max_epochs, spe, weight_decay)
    if sched is not None:
        early_stop = False  # schedule defined over full budget; best checkpoint still restored
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    plain = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    hist = dict(train_loss=[], val_loss=[], val_acc=[], grad_norms=[])
    best, best_ep, wait, t0 = np.inf, 0, 0, time.time()
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(spe):
            idx = perm[i*batch:(i+1)*batch]
            optim.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            if grad_log and i == 0:
                hist["grad_norms"].append([float(m.weight.grad.norm()) for m in model if isinstance(m, nn.Linear)])
            optim.step()
            if sched is not None: sched.step()
        trl = eval_loss(model, Xtr, ytr, plain); val = eval_loss(model, Xva, yva, plain)
        hist["train_loss"].append(trl); hist["val_loss"].append(val)
        hist["val_acc"].append(float(((predict(model, Xva) >= 0.5) == yva).mean()))
        if val < best - 1e-5:
            best, best_ep, wait = val, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if early_stop and wait >= patience:
                break
    model.load_state_dict(best_state)
    info = dict(best_epoch=best_ep + 1, epochs_run=ep + 1, best_val_loss=best,
                train_loss_at_best=hist["train_loss"][best_ep], time_s=time.time() - t0,
                n_params=sum(p.numel() for p in model.parameters()))
    return model, hist, info
