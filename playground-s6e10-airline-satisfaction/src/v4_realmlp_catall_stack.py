# %% [markdown]
# Airline Satisfaction v4 — RealMLP (pytabkit) x2 seeds + CatBoost "allcat" + OOF da v3 (LGB/XGB/Cat), stack logístico vs hill-climb
# Mesmas dobras da v3: StratifiedKFold(5, shuffle=True, random_state=42)
# %%
import subprocess, sys
try:
    import pytabkit  # noqa
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytabkit', 'dill', 'msgpack'], check=False)
import glob, os, time, numpy as np, pandas as pd
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
orig_files = [f for f in files if os.path.dirname(f) != comp_dir and os.path.basename(f) in ('train.csv', 'test.csv')]
orig = pd.concat([pd.read_csv(f) for f in orig_files], ignore_index=True) if orig_files else pd.DataFrame()
orig = orig.drop(columns=[c for c in ['Unnamed: 0', 'id'] if c in orig])
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
print('features', X.shape, flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
oof, pred = {}, {}

# %% RealMLP
if HAS_MLP:
    for seed in (0, 1):
        n = f'mlp{seed}'; oof[n] = np.zeros(len(X)); pred[n] = np.zeros(len(Xt))
        for k, (tr, va) in enumerate(folds):
            t0 = time.time()
            m = RealMLP_TD_Classifier(device='cuda', random_state=seed * 10 + k, verbosity=0,
                                      n_epochs=4, batch_size=512, n_ens=4, hidden_sizes=[512, 256, 128],
                                      embedding_size=8, max_one_hot_cat_size=9, p_drop=0.05, lr=0.04,
                                      use_ls=False, val_metric_name='1-auc_ovr', use_early_stopping=False)
            m.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_col_names=ccols)
            oof[n][va] = m.predict_proba(X.iloc[va])[:, 1]
            pred[n] += m.predict_proba(Xt)[:, 1] / 5
            print(f'{n} fold {k}: {roc_auc_score(y[va], oof[n][va]):.5f} ({time.time()-t0:.0f}s)', flush=True)
        print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5), flush=True)

# %% CatBoost allcat (valores exatos como categoria + combinações)
n = 'cat_all'; oof[n] = np.zeros(len(X)); pred[n] = np.zeros(len(Xt))
cbX = X[ccols + [c for c in X.columns if c.startswith('orig_')]].copy(); cbXt = Xt[cbX.columns].copy()
for c in ccols: cbX[c] = cbX[c].astype(str); cbXt[c] = cbXt[c].astype(str)
for k, (tr, va) in enumerate(folds):
    t0 = time.time()
    cb = CatBoostClassifier(iterations=8000, learning_rate=0.06, depth=7, l2_leaf_reg=2, max_ctr_complexity=4,
                            one_hot_max_size=2, border_count=254, eval_metric='AUC', od_type='Iter', od_wait=300,
                            random_seed=42, verbose=0, task_type='GPU')
    cb.fit(cbX.iloc[tr], y[tr], cat_features=ccols, eval_set=(cbX.iloc[va], y[va]))
    oof[n][va] = cb.predict_proba(cbX.iloc[va])[:, 1]; pred[n] += cb.predict_proba(cbXt)[:, 1] / 5
    print(f'{n} fold {k}: {roc_auc_score(y[va], oof[n][va]):.5f} ({time.time()-t0:.0f}s)', flush=True)
print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5), flush=True)

# %% OOF da v3 (input do notebook v3), se disponível
f3 = glob.glob('/kaggle/input/**/oof_v3.csv', recursive=True)
if f3:
    o3 = pd.read_csv(f3[0]); t3 = pd.read_csv(f3[0].replace('oof_v3', 'test_v3'), index_col=0).loc[Xt.index]
    for c in o3.columns:
        oof['v3_' + c] = o3[c].values; pred['v3_' + c] = t3[c].values
        print('v3_' + c, 'CV AUC', round(roc_auc_score(y, oof['v3_' + c]), 5))
else:
    print('oof_v3 não encontrado')
pd.DataFrame(oof).to_csv('oof_v4.csv', index=False); pd.DataFrame(pred, index=Xt.index).to_csv('test_v4.csv')

# %% blend 1: hill-climb de ranks
M = list(oof)
r = lambda a: rankdata(a) / len(a)
R = {n: r(oof[n]) for n in M}; Rt = {n: r(pred[n]) for n in M}
w = {n: 0 for n in M}; b0 = max(M, key=lambda n: roc_auc_score(y, R[n])); w[b0] = 1
cur = roc_auc_score(y, R[b0])
for _ in range(50):
    s, n = max((roc_auc_score(y, (sum(w[q] * R[q] for q in M) + R[n]) / (sum(w.values()) + 1)), n) for n in M)
    if s <= cur + 1e-6: break
    cur = s; w[n] += 1
hc_auc = cur; hc = sum(w[n] * Rt[n] for n in M) / sum(w.values())
print('hill-climb pesos', w, '| CV AUC', round(hc_auc, 5), flush=True)

# %% blend 2: stack logístico em logits (avaliado com as mesmas dobras)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = np.column_stack([lg(oof[n]) for n in M]); St = np.column_stack([lg(pred[n]) for n in M])
so = np.zeros(len(y))
for tr, va in folds:
    so[va] = LogisticRegression(C=1.0, max_iter=2000).fit(S[tr], y[tr]).predict_proba(S[va])[:, 1]
st_auc = roc_auc_score(y, so)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(S, y)
print('stack coefs', dict(zip(M, np.round(lr.coef_[0], 3))), '| CV AUC', round(st_auc, 5), flush=True)

best = 'stack' if st_auc > hc_auc else 'hill'
final = lr.predict_proba(St)[:, 1] if best == 'stack' else hc
print('FINAL', best, '| CV AUC ensemble', round(max(st_auc, hc_auc), 5), flush=True)
pd.DataFrame({'id': Xt.index, 'satisfaction': final}).to_csv('submission.csv', index=False)
