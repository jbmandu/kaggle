# %% [markdown]
# Airline Satisfaction v6 — RealMLP (config da mlp0 da v4) bagging 5 seeds, com as linhas do dataset original somadas ao treino de cada dobra + OOF v5/v4/v3, stack
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

# %% features (mesmas da v4; linhas do original com TE leave-one-out para não vazar o próprio rótulo)
cols = list(train.columns)
cats = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
nums = [c for c in cols if c not in cats]
AD, DD = 'Arrival Delay in Minutes', 'Departure Delay in Minutes'
for d in (train, test, orig):
    if len(d): d[AD] = d[AD].fillna(d[DD])
ALL = pd.concat([train, test], ignore_index=True)
prior = yo.mean()

def build(df, is_orig=False):
    out = pd.DataFrame(index=df.index)
    for c in nums: out[c] = df[c].astype(float)
    for c in nums: out['cnt_' + c] = df[c].map(ALL[c].value_counts()).fillna(0).astype(float)
    for c in cols:
        g = pd.DataFrame({'k': orig[c].astype(str), 't': yo}).groupby('k')['t'].agg(['sum', 'count'])
        k = df[c].astype(str)
        s, n = k.map(g['sum']).fillna(0).values, k.map(g['count']).fillna(0).values
        if is_orig: s, n = s - yo, n - 1
        out['orig_' + c] = (s + prior * 20) / (n + 20)
    for c in cols: out['c_' + c] = df[c].astype(str)
    return out

X, Xt, Xo = build(train), build(test), build(orig, True)
ccols = [c for c in X.columns if c.startswith('c_')]
for c in ccols:
    cat_t = pd.CategoricalDtype(sorted(set(X[c]) | set(Xt[c]) | set(Xo[c])))
    X[c] = X[c].astype(cat_t); Xt[c] = Xt[c].astype(cat_t); Xo[c] = Xo[c].astype(cat_t)
print('features', X.shape, 'orig', Xo.shape, flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
oof, pred = {}, {}

# %% RealMLP bag: 5 seeds, treino = dobra + original inteiro; validação só no train da competição
SEEDS = (11, 12, 13, 14, 15)
n = 'mlp_bag_orig'; oof[n] = np.zeros(len(X)); pred[n] = np.zeros(len(Xt))
for k, (tr, va) in enumerate(folds):
    Xtr = pd.concat([X.iloc[tr], Xo], ignore_index=True); ytr = np.concatenate([y[tr], yo])
    for s in SEEDS:
        t0 = time.time()
        m = RealMLP_TD_Classifier(device='cuda', random_state=s * 10 + k, verbosity=0,
                                  n_epochs=4, batch_size=512, n_ens=4, hidden_sizes=[512, 256, 128],
                                  embedding_size=8, max_one_hot_cat_size=9, p_drop=0.05, lr=0.04,
                                  use_ls=False, val_metric_name='1-auc_ovr', use_early_stopping=False)
        m.fit(Xtr, ytr, X.iloc[va], y[va], cat_col_names=ccols)
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[n][va] += p / len(SEEDS); pred[n] += m.predict_proba(Xt)[:, 1] / (5 * len(SEEDS))
        print(f'fold {k} seed {s}: {roc_auc_score(y[va], p):.5f} ({time.time()-t0:.0f}s)', flush=True)
    print(f'{n} fold {k} bag: {roc_auc_score(y[va], oof[n][va]):.5f}', flush=True)
print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5), flush=True)

# %% OOF anteriores (v5 já inclui v4 e v3)
f5 = glob.glob('/kaggle/input/**/oof_v5.csv', recursive=True)
if f5:
    o5 = pd.read_csv(f5[0]); t5 = pd.read_csv(f5[0].replace('oof_v5', 'test_v5'), index_col=0).loc[Xt.index]
    for c in o5.columns: oof[c] = o5[c].values; pred[c] = t5[c].values
else:
    print('oof_v5 não encontrado')
for c in oof: print(c, 'CV AUC', round(roc_auc_score(y, oof[c]), 5))
pd.DataFrame(oof).to_csv('oof_v6.csv', index=False); pd.DataFrame(pred, index=Xt.index).to_csv('test_v6.csv')

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
