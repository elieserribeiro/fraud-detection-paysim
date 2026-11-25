# Detecção e Prevenção à Fraude Bancária (PaySim)

MVP de prevenção de fraudes: modelos supervisionados (Regressão Logística, Random Forest, XGBoost) e
não supervisionados (Isolation Forest, Autoencoder), score de risco com três decisões
(aprovação automática, autenticação adicional, investigação), API e painel Streamlit.

As decisões do projeto estão em [`docs/decisoes.md`](docs/decisoes.md).

## Estrutura

```
configs/        configuração central (caminhos, split, custos)
data/           raw (CSV do Kaggle) · interim (Parquet) · processed
docs/           decisões, metodologia, resultados
models/         modelos treinados (ignorados no git)
notebooks/      execução no Colab e análises (EDA)
reports/        figuras e tabelas de resultados
src/fraud/      código reutilizável (dados, features, split, modelos, avaliação)
app/            painel Streamlit e API (próximas fases)
tests/          testes automatizados
```

## Como começar (VS Code, local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Como rodar no Colab

1. Coloque este repositório no Google Drive (ex.: `MyDrive/fraud-detection-paysim`) e o CSV do
   PaySim em `data/raw/` (baixe do Kaggle: *PaySim synthetic financial datasets*).
2. Abra `notebooks/00_colab_setup.ipynb`, escolha runtime GPU e execute. Ele monta o Drive,
   instala dependências, converte o CSV em Parquet e valida split e features.
3. Os modelos treinados serão salvos em `models/` (no Drive) para o painel usar.

## Roadmap

- [x] Fase 1 — Fundação: estrutura, config, CSV→Parquet, split temporal, features (com/sem saldos)
- [x] Fase 2 — EDA e baseline (`notebooks/01_eda_baseline.ipynb`)
- [x] Fase 3 — Supervisionados (LogReg, RF, XGBoost) — `notebooks/02_supervisionados.ipynb`
- [x] Fase 4 — Anomalia (Isolation Forest, Autoencoder) — `notebooks/03_anomalia.ipynb`
- [x] Fase 5 — Comparação, ensemble e impacto financeiro — `notebooks/04_final_ensemble_impacto.ipynb`
- [x] Fase 6 — Thresholds por custo esperado — `notebooks/05_thresholds_3_niveis.ipynb`
- [ ] Fase 7 — Scoring, API e painel Streamlit (+ SHAP) — `notebooks/06_exportar_painel.ipynb`, `app/`, `api/`
- [ ] Fase 8 — Documentação técnica

## Painel e API

O painel e a API leem tudo de `app/data`, gerada pelo notebook `06_exportar_painel.ipynb` (modelo XGBoost em JSON,
amostra de demonstração do teste, exemplos, cortes e resultados). Não precisam do CSV original.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py          # painel em http://localhost:8501

pip install fastapi uvicorn
uvicorn api.main:app --reload               # API em http://127.0.0.1:8000/docs
```

Abas do painel: **Resumo** (números-chave), **Testar uma transação** (exemplos prontos, formulário e explicação SHAP),
**Política e impacto** (cortes e custos ajustáveis), **Lote (CSV)**, **Modelos** (comparação dos 5 modelos e do ensemble)
e **Sobre** (metodologia e limitações).
