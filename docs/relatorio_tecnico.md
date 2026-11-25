# Relatório técnico — Detecção e prevenção de fraude bancária (PaySim)

**Painel:** https://antifraude-paysim.streamlit.app · **Código:** https://github.com/elieserribeiro/fraud-detection-paysim
**Registro detalhado de decisões:** [`decisoes.md`](decisoes.md)

---

## 1. Objetivo

Construir um MVP que atribua um **score de risco** a cada transação e a encaminhe para um de três níveis:
**aprovação automática**, **autenticação adicional** ou **investigação**. O critério de sucesso é financeiro:
capturar o maior valor possível em fraudes mantendo sob controle os falsos positivos e o custo operacional.

O projeto é de portfólio, com dados sintéticos, e não tem intenção de uso em produção. A seção 8 discute o que
seria necessário para uma implantação real.

## 2. Dados

**PaySim** (Kaggle): simulação de transações de dinheiro móvel calibrada a partir de dados reais.

| Item | Valor |
|---|---|
| Transações | 6.362.620 |
| Fraudes | 8.213 (0,129%) |
| Valor fraudado | ~R$ 12,1 bi |
| Período | 743 *steps* (1 *step* = 1 hora, ~31 dias) |
| Tipos com fraude | só `TRANSFER` (4.097) e `CASH_OUT` (4.116) |

### Achados da análise exploratória

- **Padrão dominante: tomada de conta seguida de esvaziamento.** 97,8% das fraudes têm valor igual ao saldo inteiro
  da conta de origem; entre as legítimas, 0,0001%.
- **Artefatos do simulador** que viram atalhos para o modelo:
  - o saldo pós-transação "fecha" a conta em 99,5% das fraudes (e em só 9% das legítimas);
  - fraude limitada a R$ 10 milhões (287 casos exatamente no teto);
  - 16 fraudes com valor zero;
  - fraude em ritmo constante (~10 por hora), enquanto o volume de transações legítimas cai depois do *step* ~405;
  - muitas transações legítimas com saldo de origem zerado.
- **`isFlaggedFraud`** (a regra do próprio simulador) marca só 16 fraudes (0,2%). Ficou fora das features.

## 3. Metodologia

### 3.1 Divisão temporal

Treino, validação e teste em **ordem cronológica**, cortados por número de transações em fronteiras de *step*,
para simular o uso real (treinar no passado, prever o futuro) sem misturar horas entre conjuntos.

| Conjunto | Transações | Fraudes | Taxa | *Steps* |
|---|---|---|---|---|
| Treino | 3.820.599 | 3.193 | 0,084% | 1–281 |
| Validação | 1.293.285 | 770 | 0,060% | 282–355 |
| Teste | 1.248.736 | 4.250 | 0,340% | 356–743 |

O corte por *step* puro foi descartado: deixava o teste com 1,3% de fraude, muito longe da taxa real.
O teste foi usado **uma única vez**, na avaliação final; toda escolha (cortes, pesos de ensemble) foi feita na validação.

### 3.2 Cenários de features

Para medir o quanto cada modelo depende dos atalhos do simulador, as features foram organizadas em quatro cenários:

| Cenário | Conteúdo | Papel |
|---|---|---|
| A · `com_saldos` | tudo, inclusive saldos **após** a transação | teto irreal (vazamento) |
| **B · `sem_saldos_pos`** | só o que se sabe **antes** de aprovar: tipo, valor, hora, dia, saldos anteriores e sinais derivados | **principal** |
| C · `sem_esvaziamento` | B sem `amount_to_old_org`, `orig_balance_zero` e `amount_equals_old_org` | teste de estresse |
| D · `sem_esvaziamento_sem_tempo` | C sem hora e dia | estresse adicional (só validação) |

### 3.3 Desbalanceamento

**Pesos de classe** (`class_weight='balanced'`, `scale_pos_weight`), mantendo a distribuição real dos dados.
Durante o desenvolvimento, o treino usou todas as fraudes e 20% das legítimas para economizar processamento;
a rodada final usou 100% das legítimas.

### 3.4 Modelos

| Modelo | Tipo | Configuração principal |
|---|---|---|
| Regressão Logística | supervisionado | padronização + `class_weight='balanced'` |
| Random Forest | supervisionado | 150 árvores, profundidade 14, ~383 mil linhas por árvore, `balanced_subsample` |
| XGBoost | supervisionado | 400 árvores, profundidade 6, taxa 0,1, `subsample`/`colsample` 0,8, `scale_pos_weight` |
| Isolation Forest | anomalia | 200 árvores, 1.024 amostras por árvore |
| Autoencoder | anomalia | Keras 32-16-código-16-32, score = erro de reconstrução |

Os modelos de anomalia foram treinados **só com transações legítimas** de `TRANSFER` e `CASH_OUT`; os demais tipos
recebem o menor score. As entradas foram transformadas com log com sinal, padronizadas e **cortadas em ±5 desvios**:
sem o corte, colunas binárias raras (como `amount_equals_old_org`, com desvio 0,0012) geravam valores de até 578 desvios
nas fraudes e dominavam a reconstrução.

### 3.5 Métricas

- **PR-AUC** (área sob a curva precisão × recall): a métrica principal, adequada a classes muito desbalanceadas.
- **Recall em valor**: fração do **valor** fraudado capturada, que é o que importa para o banco.
- **Taxa de falsos positivos** e **carga operacional** (% das transações que exigem ação).
- Os falsos positivos são **reponderados** para a taxa de fraude real (0,129%), porque validação e teste têm taxas diferentes.

### 3.6 Ensemble

Os scores dos modelos foram levados a uma escala comum que preserva a cauda das fraudes, `-log10(1 - ECDF_validação)`, e
combinados de duas formas: **stacking** (Regressão Logística sobre os scores) e **média ponderada** (grade de pesos em
passos de 0,25). O ajuste foi feito só na validação, avaliado fora da amostra em dois blocos temporais e medido no teste.
Uma variante só com os supervisionados isolou a contribuição dos modelos de anomalia.

### 3.7 Decisão em três níveis por custo esperado

Sobre o score `s` do XGBoost, com cortes `t_auth ≤ t_inv`:

| Nível | Condição | Custo operacional | Captura assumida |
|---|---|---|---|
| Aprovar | `s < t_auth` | R$ 0 | 0% |
| Autenticar | `t_auth ≤ s < t_inv` | R$ 2 por transação | 70% do valor da fraude |
| Investigar | `s ≥ t_inv` | R$ 50 por transação | 95% do valor da fraude |

**Custo total = custo operacional + perda residual** (o valor das fraudes não barradas). Como o custo se separa em uma
parte que depende só de `t_auth` e outra que depende só de `t_inv`, o **ótimo exato** sobre todos os cortes possíveis sai
em tempo linear depois de ordenar os scores. O resultado foi conferido contra busca exaustiva nos testes automatizados.
Como os custos são premissas, a análise foi repetida em uma grade com a mesma razão 25:1: (50; 2), (500; 20),
(5.000; 200) e (50.000; 2.000).

### 3.8 Regra de precaução

Modelos de árvore não extrapolam: fora da faixa vista no treino, repetem o comportamento dos maiores valores conhecidos.
Transações com valor ou saldos acima do máximo do PaySim (valor ~R$ 92,4 mi, saldo de origem ~R$ 59,6 mi, saldo de
destino ~R$ 356 mi) vão para **investigação por regra**, independentemente do score. "Valor maior que o saldo" fica só
como aviso: no PaySim isso é comum em transações legítimas, e uma regra marcaria muitas delas.

## 4. Resultados

### 4.1 Os cinco modelos no teste

| Modelo | PR-AUC B | PR-AUC C | Recall em valor B | Recall em valor C | FPR C |
|---|---|---|---|---|---|
| **XGBoost** | **1,000** | **0,920** | 99,99% | **98,2%** | **0,02%** |
| Random Forest | 1,000 | 0,817 | 100,0% | 97,0% | 0,03% |
| Regressão Logística | 0,9998 | 0,468 | 99,99% | 88,6% | 0,04% |
| Isolation Forest | 0,220 | 0,306 | 51,6% | 50,3% | 0,11% |
| Autoencoder | 0,978 | 0,026 | 84,5% | 8,8% | 0,07% |

Recall em valor e FPR no corte de melhor F1. A PR-AUC do teste não é comparável à da validação, porque o teste tem
mais fraude (0,34% contra 0,06%).

**Leitura:**
- No **cenário B**, quase todos chegam perto de 1,0: o sinal "valor = saldo inteiro" resolve o problema. No XGBoost,
  essa única feature responde por 85% da importância.
- No **cenário C**, os modelos se separam. As árvores, e o XGBoost em especial, mantêm bom desempenho; a Regressão
  Logística cai pela metade.
- **Os modelos de anomalia não competem.** O Autoencoder só vai bem no cenário B, porque o atalho também é uma
  anomalia; sem ele, fica perto do acaso. Sem rótulos, eles não sabem *qual* anomalia importa.
- Tempos de treino em CPU com 3,8 milhões de linhas: Regressão Logística ~16 s, Random Forest ~3 min, XGBoost ~4 min,
  Isolation Forest ~20 s, Autoencoder ~1 min.

### 4.2 Ensemble (cenário C, teste)

| Combinação | PR-AUC | Recall em valor | FPR |
|---|---|---|---|
| XGBoost sozinho | **0,920** | **98,2%** | **0,02%** |
| Média ponderada (5 modelos) | 0,920 | 98,2% | 0,02% |
| Stacking (5 modelos) | 0,889 | 88,5% | 0,07% |
| Stacking (só supervisionados) | 0,887 | 87,2% | 0,07% |

A média ponderada colocou **peso 1,0 no XGBoost** e zero nos demais; o stacking ficou pior, possivelmente por ter sido
ajustado com só 770 fraudes na validação. Incluir os modelos de anomalia mudou a PR-AUC do stacking em 0,001.
**Conclusão: XGBoost sozinho**, mais simples e melhor.

### 4.3 Decisão em três níveis (XGBoost, cortes da validação, medidos no teste)

**Cenário B, custo padrão:** os dois cortes praticamente coincidem (score ≥ 99,96%). São investigadas 4.247
transações, **todas fraudes**: precisão de 100%, 0,13% das transações. Escapam 3 fraudes (0,01% do valor).
O nível de autenticação não é usado, porque não há zona cinzenta. Benefício líquido de R$ 6,35 bi sobre R$ 6,68 bi fraudados.

**Cenário C, custo padrão:**

| Nível | Transações reais | Fraudes | Precisão no teste | Valor das fraudes |
|---|---|---|---|---|
| Aprovar | 97,23% | 9 | — | R$ 0,19 mi |
| Autenticar | 1,87% | 84 | 0,36% | R$ 4,7 mi |
| Investigar | 0,90% | 4.157 | 30,2% | R$ 6.678 mi |

Valor poupado: **94,98%** (o teto é 95%, a captura assumida da investigação).

**Valor do terceiro nível e robustez (cenário C):**

| Custos (investigar / autenticar) | % autenticar | % investigar | Ganho de 3 níveis sobre 2 | Arrependimento |
|---|---|---|---|---|
| R$ 50 / R$ 2 | 1,87% | 0,90% | R$ 2,0 mi | 0,00% |
| R$ 500 / R$ 20 | 1,29% | 0,44% | R$ 5,6 mi | 0,00% |
| R$ 5.000 / R$ 200 | 0,73% | 0,17% | R$ 27,5 mi | 0,02% |
| R$ 50.000 / R$ 2.000 | 0,34% | 0,10% | R$ 80,4 mi | 0,02% |

- O terceiro nível vale mais quando investigar é caro: a autenticação filtra casos de risco intermediário a um custo baixo.
- O **arrependimento** (quanto se perde por escolher os cortes na validação, comparado ao melhor corte possível no
  próprio teste) fica em até 0,02%: os cortes do XGBoost se transferem bem entre períodos. Na comparação de dois
  níveis (cenário B), o Random Forest não teve a mesma estabilidade: pegou 99,2% das fraudes em quantidade, mas só 91,9% do valor.

### 4.4 Uma observação sobre os custos

O valor médio de uma fraude no PaySim (~R$ 1,5 mi) é muito maior que qualquer custo de investigação da grade. Por isso o
ótimo econômico sempre manda para análise quase todas as fraudes, e o dinheiro poupado quase não diferencia os modelos.
O que os separa é a **carga operacional**: a R$ 50 por investigação, no cenário C de dois níveis, o XGBoost gera 16.336
alertas, contra 132.913 da Regressão Logística, para capturar praticamente o mesmo valor.

## 5. Painel e API

- **Painel Streamlit** (link no topo), com seis abas: resumo, teste de uma transação com exemplos prontos e explicação
  SHAP, política e impacto com cortes e custos ajustáveis, lote em CSV, comparação de modelos, e metodologia e limitações.
- **Os dados do painel** (`app/data`, 2,4 MB): modelo XGBoost em JSON nativo (portável entre versões) e uma amostra do teste
  com todas as fraudes e 30 mil legítimas, com pesos que recompõem os números reais.
- **API FastAPI**: `GET /health`, `POST /score` (score, nível, regra aplicada e os 5 principais motivos) e `POST /score/lote`.
- **Código único** (`src/fraud`) para treino, painel e API; **39 testes automatizados**, entre eles a verificação do
  otimizador de cortes contra busca exaustiva. O notebook 06 confere que o modelo salvo para o painel reproduz os scores da avaliação.

## 6. Limitações

1. **Dados sintéticos com atalhos.** O desempenho no cenário B reflete artefatos do simulador; o cenário C é a melhor
   estimativa do que se poderia esperar com fraudadores que não esvaziam a conta, e ainda assim é otimista.
2. **Fragilidade a mudança de comportamento.** O modelo do painel (cenário B) depende fortemente do sinal de esvaziamento.
   Uma fraude parcial, como o exemplo "Fraude difícil", passa com risco de ~1%.
3. **Sem histórico por cliente.** O PaySim não traz o comportamento anterior de cada conta (horários usuais, valores
   típicos, destinatários conhecidos). Por isso a hora da transação, sozinha, pesa pouco.
4. **Premissas de custo e captura** (investigação R$ 50 / 95%, autenticação R$ 2 / 70%) não vêm de dados reais.
5. **Falsos positivos reponderados.** Os números de carga operacional dependem da hipótese de que a taxa real de
   fraude é 0,129%, a do PaySim.
6. **Modelos de anomalia fracos** nos cenários difíceis: ficaram como comparação e não entram na decisão.
7. **Amostra do painel.** As métricas do painel são recompostas a partir de uma amostra do teste, e não do teste inteiro.

## 7. Conclusões

- O **XGBoost supervisionado** é o melhor equilíbrio entre detecção, falsos positivos e custo de treino, e o ensemble não
  trouxe ganho.
- A **decisão por custo esperado** liga o modelo ao dinheiro e mostra quando o nível intermediário vale a pena.
- A principal lição é **metodológica**: sem os testes de estresse (cenários C e D), o projeto pareceria perfeito.
  Medir a dependência de atalhos é o que torna os resultados confiáveis.

## 8. Recomendações para uma implantação real

1. **Dados reais e histórico do cliente.** Features de comportamento: média e desvio do valor por cliente, horário
   usual, novo destinatário, velocidade de transações na última hora, dispositivo e localização.
2. **Rotulagem com atraso.** Fraudes são confirmadas dias ou semanas depois (contestação, chargeback). O treino e a
   avaliação devem respeitar essa defasagem.
3. **Custos reais.** Calibrar custo de investigação, custo de atrito ao cliente (inclusive a perda de clientes por
   bloqueios indevidos) e taxas de captura com dados da operação, e recalcular os cortes periodicamente.
4. **Implantação gradual.** Rodar primeiro em **modo sombra** (o modelo pontua, mas não decide), depois como
   **desafiante** do sistema atual, comparando resultados antes de assumir.
5. **Monitoramento.** Acompanhar a deriva das features e do score (por exemplo, PSI), a taxa de alertas por nível e a
   precisão das investigações, com alarmes e retreino programado.
6. **Ciclo de retorno.** O resultado de cada investigação e autenticação volta como rótulo para o próximo treino.
7. **Regras e modelo juntos.** Manter regras de negócio (saldo insuficiente, listas de bloqueio, limites) antes do
   modelo, como a regra de precaução deste MVP.
8. **Governança e explicabilidade.** Registrar score, motivos (SHAP) e decisão de cada transação para auditoria e para
   responder a clientes e reguladores, respeitando a LGPD.
9. **Latência.** O XGBoost pontua em milissegundos; em produção, a API precisaria de cache do modelo, testes de carga e
   alta disponibilidade.

## 9. Reprodutibilidade

| Notebook | Conteúdo | Ambiente | Tempo aproximado |
|---|---|---|---|
| `00_colab_setup` | monta o Drive e converte o CSV em Parquet | CPU | minutos |
| `01_eda_baseline` | análise exploratória e baseline | CPU | minutos |
| `02_supervisionados` | 3 modelos × 4 cenários na validação | GPU opcional | ~10 min |
| `03_anomalia` | Isolation Forest e Autoencoder | CPU | minutos |
| `04_final_ensemble_impacto` | retreino final, teste, ensemble e impacto financeiro | CPU | ~17 min de treino |
| `05_thresholds_3_niveis` | cortes de três níveis | CPU | segundos |
| `06_exportar_painel` | gera `app/data` para o painel | CPU | 1–2 min |

Semente fixa (`seed: 42`) e configuração central em `configs/config.yaml`.
