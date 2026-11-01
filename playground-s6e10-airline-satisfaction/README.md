# Playground Series S6E10 — Predicting Airline Satisfaction

Competição: https://www.kaggle.com/competitions/playground-series-s6e10
Métrica: ROC AUC · treino 699.635 linhas, teste 299.844 (dados sintéticos baseados no dataset Airline Passenger Satisfaction).

**Resultado final:** 1378º de 3748 times · LB privado **0,96157** · LB público 0,96082 (submissão v5).

## Evolução

| Versão | Abordagem | CV AUC | LB público | Notebook Kaggle |
| --- | --- | --- | --- | --- |
| v1 | LightGBM 5-fold + agregados das notas de serviço | 0,95888 | 0,95846 | [link](https://www.kaggle.com/code/juliomandu/airline-satisfaction-lightgbm-baseline) |
| v2 | LightGBM + CatBoost + dataset original no treino | 0,95981 | 0,95936 | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v2-lgbm-catboost-original) |
| v3 | Target encoding de rota (distância) e rota × classe; LGB + XGB + CatBoost (GPU), hill-climb | 0,96051 | 0,96008 | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v3-lgb-xgb-cat-route-te) |
| v4 | RealMLP (pytabkit) ×2 + CatBoost com todas as colunas categóricas + OOFs da v3, stack | 0,96114 | 0,96078 | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v4-realmlp-catall-stack) |
| v5 | Features kNN para o original + RealMLP ×2 + LightGBM max_bin 4095 + stack logístico | **0,96121** | **0,96082** | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v5-knn-realmlp-lgb-stack) |
| v6 | Bagging de RealMLP (5 seeds) + linhas do original no treino | 0,9612 | não submetida | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v6-realmlp-bag-orig) |
| v7 | Original correto (arseniyshutko 129k) + match exato de linhas + kNN | 0,96121 | não submetida | [link](https://www.kaggle.com/code/juliomandu/airline-sat-v7-shutko-orig-match) |

## Lições

- RealMLP foi o melhor modelo único (~0,96065 de CV), acima dos GBDTs (~0,9604).
- LightGBM melhora com `max_bin` alto (4095) e o CatBoost tratando tudo como categórico.
- O stack saturou em ~0,9612 de CV: empilhar modelos parecidos não trouxe ganho a partir da v5.
- O dataset que inspirou a competição é o `arseniyshutko/binary-aviation-satisfaction-129k`; as linhas sintéticas não copiam o original, então match exato não ajudou.
- Os topos do LB usaram stacks com OOFs públicas (TabPFN, muitos folds) — caminho não seguido aqui.

## Estrutura

- `src/` — scripts `.py` de cada versão
- `notebooks/` — notebooks equivalentes (rodar no Kaggle com o input da competição; v4+ usam GPU T4 e as saídas das versões anteriores)
