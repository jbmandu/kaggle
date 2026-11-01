# %% [markdown]
# Airline Satisfaction v7 — ORIGINAL CORRETO (arseniyshutko/binary-aviation-satisfaction-129k, data.csv) + match exato de linha no original + kNN + RealMLP x2 + LGB 4095 + OOF v6/v5/v4/v3, stack
# Mesmas dobras da v3: StratifiedKFold(5, shuffle=True, random_state=42)
# %%
import subprocess, sys
try:
    import pytabkit  # noqa
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytabkit', 'dill', 'msgpack'], check=False)
import glob, os, time, numpy as np, pandas as pd, lightgbm as lgb, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from catboost import CatBoostClassifier
try:
    from pytabkit import RealMLP_TD_Classifier; HAS_MLP = True
except Exception as e:
    print('pytabkit indisponível:', e); HAS_MLP = False

files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
comp_dir = os.path.dirname([f for f in files if 's6e10' in f and f.endswith('train.csv')][0])
train = pd.read_csv(f'{comp_dir}/train.csv', index_col='id')
test  = pd.read_csv(f'{comp_dir}/test.csv',  index_col='id')
norm = lambda c: ''.join(ch for ch in str(c).lower() if ch.isalnum())
def load_orig(paths):
    if not paths: return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    ren = {c: t for c in d.columns for t in list(train.columns) + ['satisfaction'] if norm(c) == norm(t)}
    return d.rename(columns=ren)
shutko = [f for f in files if os.path.basename(f) == 'data.csv' and 's6e10' not in f]
teej = [f for f in files if os.path.dirname(f) != comp_dir and os.path.basename(f) in ('train.csv', 'test.csv')]
print('shutko', shutko, '| teej', teej, flush=True)
orig = load_orig(shutko) if shutko else load_orig(teej)
print('orig cols faltando vs train:', [c for c in train.columns if c not in orig.columns], '| extras:', [c for c in orig.columns if c not in train.columns], flush=True)
for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Cleanliness', 'Flight Distance']:
    if c in orig and c in train: print('diag', c, 'train', train[c].describe()[['mean', 'max']].round(2).tolist(), 'orig', orig[c].describe()[['mean', 'max']].round(2).tolist())
T = 'satisfaction'
to01 = lambda s: s.astype(str).str.lower().isin(['true', '1', 'satisfied']).astype(int)
y = to01(train.pop(T)).values
orig = orig.reindex(columns=list(train.columns) + [T]).dropna(subset=[T]) if len(orig) else orig
yo = to01(orig.pop(T)).values if len(orig) else None
print(train.shape, test.shape, orig.shape, 'MLP', HAS_MLP, flush=True)

# %% features
cols = list(train.columns)
cats = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
nums = [c for c in cols if c not in cats]
AD, DD = 'Arrival Delay in Minutes', 'Departure Delay in Minutes'
for d in (train, test, orig):
    if len(d): d[AD] = d[AD].fillna(d[DD])
ALL = pd.concat([train, test], ignore_index=True)

def build(df):
    out = pd.DataFrame(index=df.index)
    for c in nums: out[c] = df[c].astype(float)
    for c in nums: out['cnt_' + c] = df[c].map(ALL[c].value_counts()).fillna(0).astype(float)
    if len(orig):
        prior = yo.mean()
        for c in cols:
            g = pd.DataFrame({'k': orig[c].astype(str), 't': yo}).groupby('k')['t'].agg(['mean', 'count'])
            out['orig_' + c] = df[c].astype(str).map((g['mean'] * g['count'] + prior * 20) / (g['count'] + 20)).astype(float).fillna(prior)
    for c in cols: out['c_' + c] = df[c].astype(str)   # todos os valores exatos como categoria
    return out

X, Xt = build(train), build(test)
ccols = [c for c in X.columns if c.startswith('c_')]
for c in ccols:
    cat_t = pd.CategoricalDtype(sorted(set(X[c]) | set(Xt[c])))
    X[c] = X[c].astype(cat_t); Xt[c] = Xt[c].astype(cat_t)
# match exato da linha inteira (e sem atrasos) no original: média e contagem do rótulo
prior = yo.mean()
for nm, kc in (('full', cols), ('nodelay', [c for c in cols if c not in (AD, DD)])):
    ko = orig[kc].astype(str).agg('|'.join, axis=1)
    g = pd.DataFrame({'k': ko, 't': yo}).groupby('k')['t'].agg(['sum', 'count'])
    for D, df in ((X, train), (Xt, test)):
        k = df[kc].astype(str).agg('|'.join, axis=1)
        sm, ct = k.map(g['sum']).fillna(0).values, k.map(g['count']).fillna(0).values
        D['m_' + nm] = (sm + prior * 2) / (ct + 2); D['mc_' + nm] = ct
    print('match', nm, 'cobertura train', round((X['mc_' + nm] > 0).mean(), 4), 'test', round((Xt['mc_' + nm] > 0).mean(), 4), flush=True)
print('features', X.shape, flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
oof, pred = {}, {}


# %% kNN contra o dataset original (só rótulos do original: sem vazamento)
def emb(df):
    Z = [df[c].astype(float).values[:, None] / 5.0 for c in nums if c not in ('Age', 'Flight Distance', AD, DD)]
    Z += [df['Age'].values[:, None] / 40.0, df['Flight Distance'].values[:, None] / 1500.0,
          np.log1p(df[DD].values)[:, None] / 3.0, np.log1p(df[AD].values)[:, None] / 3.0]
    for c in cats:
        for v in sorted(train[c].astype(str).unique()): Z.append((df[c].astype(str).values == v).astype(float)[:, None])
    return np.hstack(Z).astype(np.float32)
KNN = {}
if len(orig):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    O = torch.tensor(emb(orig), device=dev); Yo = torch.tensor(yo, dtype=torch.float32, device=dev)
    for name, df in (('tr', train), ('te', test)):
        Q = emb(df); out = np.zeros((len(Q), 4), np.float32); t0 = time.time()
        for i in range(0, len(Q), 4096):
            q = torch.tensor(Q[i:i+4096], device=dev)
            d, idx = torch.cdist(q, O).topk(100, largest=False)
            lab = Yo[idx]
            out[i:i+4096] = torch.stack([lab[:, :10].mean(1), lab[:, :30].mean(1), lab.mean(1), d[:, 0]], 1).cpu().numpy()
        KNN[name] = out; print('knn', name, f'{time.time()-t0:.0f}s', flush=True)
    for j, nm in enumerate(['knn10', 'knn30', 'knn100', 'knn_d1']):
        X[nm] = KNN['tr'][:, j]; Xt[nm] = KNN['te'][:, j]
    print('knn10 AUC', round(roc_auc_score(y, X['knn10']), 5), flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
oof, pred = {}, {}

# %% RealMLP (agora com features kNN), 2 variações
if HAS_MLP:
    for seed, ep, lr_, hs in ((7, 4, 0.04, [512, 256, 128]), (8, 4, 0.04, [512, 256, 128])):
        n = f'mlp{seed}s'; oof[n] = np.zeros(len(X)); pred[n] = np.zeros(len(Xt))
        for k, (tr, va) in enumerate(folds):
            t0 = time.time()
            m = RealMLP_TD_Classifier(device='cuda', random_state=seed * 10 + k, verbosity=0,
                                      n_epochs=ep, batch_size=512, n_ens=4, hidden_sizes=hs,
                                      embedding_size=8, max_one_hot_cat_size=9, p_drop=0.05, lr=lr_,
                                      use_ls=False, val_metric_name='1-auc_ovr', use_early_stopping=False)
            m.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_col_names=ccols)
            oof[n][va] = m.predict_proba(X.iloc[va])[:, 1]; pred[n] += m.predict_proba(Xt)[:, 1] / 5
            print(f'{n} fold {k}: {roc_auc_score(y[va], oof[n][va]):.5f} ({time.time()-t0:.0f}s)', flush=True)
        print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5), flush=True)

# %% LightGBM com bins finos (4095) + TE in-fold de Flight Distance
n = 'lgb4095s'; oof[n] = np.zeros(len(X)); pred[n] = np.zeros(len(Xt))
lcols = [c for c in X.columns if not c.startswith('c_')]
LX = X[lcols].copy(); LXt = Xt[lcols].copy()
for c in cats: LX[c] = X['c_' + c]; LXt[c] = Xt['c_' + c]
def te(keys_tr, ytr, keys_ap, a=10):
    p = ytr.mean(); g = pd.DataFrame({'k': keys_tr, 't': ytr}).groupby('k')['t'].agg(['sum', 'count'])
    return pd.Series(keys_ap).map((g['sum'] + p * a) / (g['count'] + a)).fillna(p).values
for k, (tr, va) in enumerate(folds):
    t0 = time.time()
    A, B, C = LX.iloc[tr].copy(), LX.iloc[va].copy(), LXt.copy()
    fd = LX['Flight Distance'].values; ytr = y[tr]
    col = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=k).split(tr, ytr):
        col[b] = te(fd[tr][a], ytr[a], fd[tr][b])
    A['te_fd'] = col; B['te_fd'] = te(fd[tr], ytr, fd[va]); C['te_fd'] = te(fd[tr], ytr, Xt['Flight Distance'].values)
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.04, num_leaves=127, min_child_samples=60, max_bin=4095,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=15.0, cat_smooth=5,
                           n_estimators=10000, verbose=-1)
    m.fit(A, ytr, eval_set=[(B, y[va])], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[n][va] = m.predict_proba(B)[:, 1]; pred[n] += m.predict_proba(C)[:, 1] / 5
    print(f'{n} fold {k}: {roc_auc_score(y[va], oof[n][va]):.5f} ({time.time()-t0:.0f}s)', flush=True)
print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5), flush=True)

# %% OOF anteriores (v6 já inclui v5/v4/v3)
fp = glob.glob('/kaggle/input/**/oof_v6.csv', recursive=True) or glob.glob('/kaggle/input/**/oof_v5.csv', recursive=True)
if fp:
    op = pd.read_csv(fp[0]); tp = pd.read_csv(fp[0].replace('oof_v', 'test_v'), index_col=0).loc[Xt.index]
    for c in op.columns: oof[c] = op[c].values; pred[c] = tp[c].values
else:
    print('OOF anteriores não encontrados')
for c in oof: print(c, 'CV AUC', round(roc_auc_score(y, oof[c]), 5))
pd.DataFrame(oof).to_csv('oof_v7.csv', index=False); pd.DataFrame(pred, index=Xt.index).to_csv('test_v7.csv')
# %% blend 1: hill-climb de ranks
M = list(oof)
r = lambda a: rankdata(a) / len(a)
R = {n: r(oof[n]) for n in M}; Rt = {n: r(pred[n]) for n in M}
w = {n: 0 for n in M}; b0 = max(M, key=lambda n: roc_auc_score(y, R[n])); w[b0] = 1
cur = roc_auc_score(y, R[b0])
for _ in range(80):
    s, n = max((roc_auc_score(y, (sum(w[q] * R[q] for q in M) + R[n]) / (sum(w.values()) + 1)), n) for n in M)
    if s <= cur + 1e-6: break
    cur = s; w[n] += 1
hc_auc = cur; hc = sum(w[n] * Rt[n] for n in M) / sum(w.values())
print('hill-climb pesos', w, '| CV AUC', round(hc_auc, 5), flush=True)

# %% blend 2: stack logístico em logits (avaliado com as mesmas dobras)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = np.column_stack([lg(oof[n]) for n in M]); St = np.column_stack([lg(pred[n]) for n in M])
best_c, st_auc = 1.0, 0
for Cc in (0.01, 0.1, 1.0):
    so = np.zeros(len(y))
    for tr, va in folds:
        so[va] = LogisticRegression(C=Cc, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])[:, 1]
    a = roc_auc_score(y, so); print('stack C', Cc, round(a, 5))
    if a > st_auc: best_c, st_auc = Cc, a
lr = LogisticRegression(C=best_c, max_iter=3000).fit(S, y)
print('stack coefs', dict(zip(M, np.round(lr.coef_[0], 3))), '| CV AUC', round(st_auc, 5), flush=True)

best = 'stack' if st_auc > hc_auc else 'hill'
final = lr.predict_proba(St)[:, 1] if best == 'stack' else hc
print('FINAL', best, '| CV AUC ensemble', round(max(st_auc, hc_auc), 5), flush=True)
pd.DataFrame({'id': Xt.index, 'satisfaction': final}).to_csv('submission.csv', index=False)
