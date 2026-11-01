# %% [markdown]
# # Airline Satisfaction v2 — dados originais + LightGBM + CatBoost
# Inputs: competição `playground-series-s6e10` + dataset `teejmahal20/airline-passenger-satisfaction`.
# 
# Melhorias sobre a v1: (1) features de taxa de satisfação por valor calculadas no dataset original; (2) linhas originais adicionadas só aos folds de treino; (3) ensemble LightGBM + CatBoost com peso escolhido no OOF.

# %%
import glob, os, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

files = glob.glob('/kaggle/input/**/*.csv', recursive=True) or glob.glob('*.csv') + glob.glob('orig/*.csv')
comp_train = [f for f in files if 's6e10' in f and f.endswith('train.csv')] or ['train.csv']
comp_dir = os.path.dirname(comp_train[0])
train = pd.read_csv(os.path.join(comp_dir, 'train.csv'), index_col='id')
test  = pd.read_csv(os.path.join(comp_dir, 'test.csv'),  index_col='id')
orig_files = [f for f in files if os.path.dirname(f) != comp_dir and os.path.basename(f) in ('train.csv','test.csv')]
orig = pd.concat([pd.read_csv(f) for f in orig_files], ignore_index=True) if orig_files else pd.DataFrame()
orig = orig.drop(columns=[c for c in ['Unnamed: 0','id'] if c in orig])
print(train.shape, test.shape, orig.shape)
print('colunas só no original:', [c for c in orig.columns if c not in train.columns and c != 'satisfaction'])
print('colunas faltando no original:', [c for c in train.columns if c not in orig.columns and c != 'satisfaction'])

T = 'satisfaction'
to01 = lambda s: s.astype(str).str.lower().isin(['true','1','satisfied']).astype(int)
y = to01(train.pop(T))
if len(orig): orig[T] = to01(orig[T])
print('target mean', y.mean().round(3), '| orig', orig[T].mean().round(3) if len(orig) else '-')

# %%
cats = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
nums = [c for c in train.columns if c not in cats]
svc = [c for c in nums if train[c].between(0,5).all() and train[c].nunique() <= 6]

def fe(df):
    df = df.copy()
    df['svc_mean'] = df[svc].mean(axis=1); df['svc_min'] = df[svc].min(axis=1)
    df['svc_std'] = df[svc].std(axis=1);   df['svc_zero'] = (df[svc] == 0).sum(axis=1)
    if 'Arrival Delay in Minutes' in df:
        df['delay_total'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'].fillna(0)
    # taxa de satisfação de cada valor no dataset original
    if len(orig):
        prior = orig[T].mean()
        for c in [c for c in train.columns if c in orig]:
            m = orig.groupby(orig[c].astype(str))[T].agg(['mean','count'])
            rate = (m['mean']*m['count'] + prior*20) / (m['count'] + 20)
            df['orig_' + c] = df[c].astype(str).map(rate).astype(float).fillna(prior)
    for c in cats: df[c] = df[c].astype(str)
    return df

X, Xt = fe(train), fe(test)
Xo = fe(orig.reindex(columns=train.columns))[X.columns] if len(orig) else None
yo = orig[T].values if len(orig) else None
print(X.shape)

# %%
def lgb_frame(df):
    df = df.copy()
    for c in cats: df[c] = pd.Categorical(df[c], categories=sorted(X[c].unique()))
    return df

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {'lgb': np.zeros(len(X)), 'cat': np.zeros(len(X))}
pred = {'lgb': np.zeros(len(Xt)), 'cat': np.zeros(len(Xt))}
XL, XtL = lgb_frame(X), lgb_frame(Xt); XoL = lgb_frame(Xo) if Xo is not None else None
try:
    import torch; GPU = torch.cuda.is_available()
except Exception: GPU = False
print('GPU:', GPU)

for k,(tr,va) in enumerate(skf.split(X,y)):
    Xtr, ytr = XL.iloc[tr], y.iloc[tr].values
    if XoL is not None: Xtr, ytr = pd.concat([Xtr, XoL]), np.r_[ytr, yo]
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.03, num_leaves=127, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                           n_estimators=8000, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(XL.iloc[va], y.iloc[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(300, verbose=False)])
    oof['lgb'][va] = m.predict_proba(XL.iloc[va])[:,1]; pred['lgb'] += m.predict_proba(XtL)[:,1]/5

    Ctr, cy = X.iloc[tr], y.iloc[tr].values
    if Xo is not None: Ctr, cy = pd.concat([Ctr, Xo]), np.r_[cy, yo]
    c = CatBoostClassifier(iterations=6000 if GPU else 2500, learning_rate=0.06, depth=7, l2_leaf_reg=3,
                           eval_metric='AUC', od_type='Iter', od_wait=300, random_seed=42, verbose=0,
                           task_type='GPU' if GPU else 'CPU', thread_count=-1)
    c.fit(Ctr, cy, cat_features=cats, eval_set=(X.iloc[va], y.iloc[va]))
    oof['cat'][va] = c.predict_proba(X.iloc[va])[:,1]; pred['cat'] += c.predict_proba(Xt)[:,1]/5
    print(f"fold {k}: LGB {roc_auc_score(y.iloc[va], oof['lgb'][va]):.5f} | CAT {roc_auc_score(y.iloc[va], oof['cat'][va]):.5f}")

for n in oof: print(n, 'CV AUC', round(roc_auc_score(y, oof[n]), 5))

# %%
from scipy.stats import rankdata
r = lambda a: rankdata(a)/len(a)
best = max(np.linspace(0,1,21), key=lambda w: roc_auc_score(y, w*r(oof['lgb']) + (1-w)*r(oof['cat'])))
print('peso LGB', best, '| CV AUC ensemble', round(roc_auc_score(y, best*r(oof['lgb']) + (1-best)*r(oof['cat'])), 5))
sub = pd.DataFrame({'id': Xt.index, 'satisfaction': best*r(pred['lgb']) + (1-best)*r(pred['cat'])})
sub.to_csv('submission.csv', index=False); sub.head()
