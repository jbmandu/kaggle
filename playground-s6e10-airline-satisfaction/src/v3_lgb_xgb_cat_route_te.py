# %% [markdown]
# Airline Satisfaction v3 — encodings de rota (Flight Distance) + LGB tunado + XGBoost + CatBoost, pesos por hill-climbing no OOF
# %%
import glob, os, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
comp_dir = os.path.dirname([f for f in files if 's6e10' in f and f.endswith('train.csv')][0])
train = pd.read_csv(f'{comp_dir}/train.csv', index_col='id')
test  = pd.read_csv(f'{comp_dir}/test.csv',  index_col='id')
orig_files = [f for f in files if os.path.dirname(f) != comp_dir and os.path.basename(f) in ('train.csv','test.csv')]
orig = pd.concat([pd.read_csv(f) for f in orig_files], ignore_index=True) if orig_files else pd.DataFrame()
orig = orig.drop(columns=[c for c in ['Unnamed: 0','id'] if c in orig])
T = 'satisfaction'
to01 = lambda s: s.astype(str).str.lower().isin(['true','1','satisfied']).astype(int)
y = to01(train.pop(T)).values
orig = orig.reindex(columns=list(train.columns)+[T]).dropna(subset=[T]) if len(orig) else orig
yo = to01(orig.pop(T)).values if len(orig) else None
print(train.shape, test.shape, orig.shape)
try:
    import torch; GPU = torch.cuda.is_available()
except Exception: GPU = False
print('GPU', GPU)

# %%
cats = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
nums = [c for c in train.columns if c not in cats]
svc = [c for c in nums if train[c].between(0,5).all() and train[c].nunique() <= 6]
ALL = pd.concat([train, test, orig], ignore_index=True)

def fe(df):
    df = df.copy()
    df['svc_mean'] = df[svc].mean(axis=1); df['svc_min'] = df[svc].min(axis=1)
    df['svc_std'] = df[svc].std(axis=1);   df['svc_zero'] = (df[svc] == 0).sum(axis=1)
    df['delay_total'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'].fillna(0)
    df['delay_diff']  = df['Arrival Delay in Minutes'] - df['Departure Delay in Minutes']
    for c in ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        df['cnt_' + c] = df[c].map(ALL[c].value_counts()).fillna(0)
    df['route_class'] = df['Flight Distance'].astype(str) + '_' + df['Class'].astype(str)
    df['seg'] = df['Class'].astype(str) + '_' + df['Type of Travel'].astype(str) + '_' + df['Customer Type'].astype(str)
    df['wifi_seg'] = df['seg'] + '_' + df['Inflight wifi service'].astype(str)
    if len(orig):  # taxa de satisfação de cada valor no original
        prior = yo.mean()
        for c in train.columns:
            g = pd.DataFrame({'k': orig[c].astype(str), 't': yo}).groupby('k')['t'].agg(['mean','count'])
            df['orig_' + c] = df[c].astype(str).map((g['mean']*g['count'] + prior*20)/(g['count']+20)).astype(float).fillna(prior)
    for c in cats: df[c] = df[c].astype(str)
    return df

X, Xt = fe(train), fe(test); Xo = fe(orig)[X.columns] if len(orig) else None
TE_COLS = ['Flight Distance', 'route_class', 'Age', 'wifi_seg']
cats2 = cats + ['seg']

def te_fit(keys, target, a=10):
    p = target.mean(); g = pd.DataFrame({'k': keys, 't': target}).groupby('k')['t'].agg(['sum','count'])
    return (g['sum'] + p*a)/(g['count'] + a), p

def add_te(Xtr, ytr, others, seed):
    """TE fora-da-dobra para o treino, ajuste completo para validação/teste."""
    Xtr = Xtr.copy(); others = [o.copy() for o in others]
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for c in TE_COLS:
        k = Xtr[c].astype(str).values; col = np.zeros(len(Xtr))
        for a, b in inner.split(k, ytr):
            m, p = te_fit(k[a], ytr[a]); col[b] = pd.Series(k[b]).map(m).fillna(p).values
        Xtr['te_' + c] = col
        m, p = te_fit(k, ytr)
        for o in others: o['te_' + c] = o[c].astype(str).map(m).fillna(p).values
    drop = ['route_class', 'wifi_seg']
    return Xtr.drop(columns=drop), [o.drop(columns=drop) for o in others]

def as_cat(df, ref):
    df = df.copy()
    for c in cats2: df[c] = pd.Categorical(df[c], categories=sorted(ref[c].unique()))
    return df

# %%
skf = StratifiedKFold(5, shuffle=True, random_state=42)
M = ['lgb', 'xgb', 'cat']
oof = {m: np.zeros(len(X)) for m in M}; pred = {m: np.zeros(len(Xt)) for m in M}
for k, (tr, va) in enumerate(skf.split(X, y)):
    Xtr, ytr = X.iloc[tr], y[tr]
    if Xo is not None: Xtr, ytr = pd.concat([Xtr, Xo]), np.r_[ytr, yo]
    Xtr, (Xva, Xte) = add_te(Xtr, ytr, [X.iloc[va], Xt], seed=k)
    yva = y[va]
    A, B, C = as_cat(Xtr, X), as_cat(Xva, X), as_cat(Xte, X)

    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.03, num_leaves=255, max_depth=-1, min_child_samples=80,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0, reg_alpha=0.5,
                           min_split_gain=0.0, cat_smooth=20, max_bin=511, n_estimators=12000, verbose=-1)
    m.fit(A, ytr, eval_set=[(B, yva)], eval_metric='auc', callbacks=[lgb.early_stopping(400, verbose=False)])
    oof['lgb'][va] = m.predict_proba(B)[:,1]; pred['lgb'] += m.predict_proba(C)[:,1]/5

    x = xgb.XGBClassifier(n_estimators=12000, learning_rate=0.02, max_depth=8, min_child_weight=5, subsample=0.8,
                          colsample_bytree=0.5, reg_lambda=3.0, max_bin=512, enable_categorical=True, tree_method='hist',
                          device='cuda' if GPU else 'cpu', eval_metric='auc', early_stopping_rounds=400)
    x.fit(A, ytr, eval_set=[(B, yva)], verbose=False)
    oof['xgb'][va] = x.predict_proba(B)[:,1]; pred['xgb'] += x.predict_proba(C)[:,1]/5

    cb = CatBoostClassifier(iterations=10000 if GPU else 3000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                            border_count=254, eval_metric='AUC', od_type='Iter', od_wait=400, random_seed=42, verbose=0,
                            task_type='GPU' if GPU else 'CPU')
    cfeat = cats2 + ['Flight Distance']
    P = lambda d: d.assign(**{'Flight Distance': d['Flight Distance'].astype(str)})
    cb.fit(P(Xtr), ytr, cat_features=cfeat, eval_set=(P(Xva), yva))
    oof['cat'][va] = cb.predict_proba(P(Xva))[:,1]; pred['cat'] += cb.predict_proba(P(Xte))[:,1]/5
    print(f'fold {k}: ' + ' | '.join(f'{n} {roc_auc_score(yva, oof[n][va]):.5f}' for n in M), flush=True)

for n in M: print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5))
pd.DataFrame(oof).to_csv('oof_v3.csv', index=False); pd.DataFrame(pred, index=Xt.index).to_csv('test_v3.csv')

# %% hill-climbing de pesos sobre ranks
r = lambda a: rankdata(a)/len(a)
R = {n: r(oof[n]) for n in M}; Rt = {n: r(pred[n]) for n in M}
w = {n: 0 for n in M}; best_n = max(M, key=lambda n: roc_auc_score(y, R[n])); w[best_n] = 1
cur = roc_auc_score(y, R[best_n])
for _ in range(30):
    cand = [(roc_auc_score(y, (sum(w[q]*R[q] for q in M) + R[n]) / (sum(w.values()) + 1)), n) for n in M]
    s, n = max(cand)
    if s <= cur + 1e-6: break
    cur = s; w[n] += 1
print('pesos', w, '| CV AUC ensemble', round(cur, 5))
blend = sum(w[n]*Rt[n] for n in M) / sum(w.values())
pd.DataFrame({'id': Xt.index, 'satisfaction': blend}).to_csv('submission.csv', index=False)
