# Decisões do projeto

Registro das decisões tomadas na fase de planejamento (24/11/2026).

| # | Tema | Decisão | Justificativa |
|---|------|---------|---------------|
| 1 | Painel final | **Streamlit** com formulário manual + upload de CSV | Reaproveita os modelos treinados em Python; cobre avaliação unitária e em lote |
| 2 | Validação | **Split temporal por `step`** (treino / validação / teste) | Simula uso real (prever o futuro); evita métricas infladas |
| 3 | Saldos | **Testar com e sem** os saldos pós-transação | Quantifica o vazamento do simulador PaySim; resultado mais honesto |
| 4 | Runtime Colab | GPU (L4/T4/A100) para tudo; cuML opcional para RF/LogReg | Prioridade é ganhar tempo; sklearn puro não usa GPU |
| 5 | Desbalanceamento | **Pesos de classe** (`class_weight`, `scale_pos_weight`) | Mantém a distribuição real e probabilidades mais fiéis; barato em 6M linhas |
| 6 | Custos | **Parâmetros editáveis no painel** com valores padrão | Perda = `amount` da fraude não detectada; custos de investigação/atrito ajustáveis |
| 7 | Thresholds | **Minimização do custo esperado** (dois cortes, validados em conjunto separado) | Liga a decisão ao dinheiro e reotimiza se os custos mudarem |
| 8 | Entrada de dados | Formulário manual + upload de CSV | Cobre teste unitário e lote; simulação em tempo real fica para fase posterior |
| 9 | Combinação de modelos | **Comparar isolados e depois testar ensemble** (supervisionado + anomalia) | Responde à meta de achar o modelo/combinação com melhor equilíbrio |
| 10 | Dataset | Baixado do Kaggle pelo usuário, salvo no Google Drive, convertido para **Parquet** | Evita token do Kaggle; leitura mais rápida e menos memória |
| 11 | Explicabilidade | **SHAP** para o modelo de árvores | Justifica cada decisão; diferencial em contexto bancário |
| 12 | Estrutura de código | **Projeto no VS Code**, pastas organizadas, módulos `src/fraud` + app Streamlit | Treino e painel usam exatamente o mesmo código |
| 13 | Split | **Por linhas** (corte por nº de transações, em fronteiras de step): treino 3,82 mi / val 1,29 mi / teste 1,25 mi | Conjuntos equilibrados, teste com 4.250 fraudes, taxas mais próximas da real (0,13%) |
| 14 | Cenários de features | **A** com saldos · **B** sem saldos pós (principal) · **C** B sem sinal de esvaziamento · **D** C sem hora/dia | Baseline saturado (PR-AUC ~0,999); C e D diferenciam os modelos |
| 16 | Modelos de anomalia: dados | **Só transações legítimas** no treino (rótulos só para filtrar) | Padrão do Autoencoder; scores mais limpos no Isolation Forest |
| 17 | Modelos de anomalia: escopo | **Só TRANSFER e CASH_OUT**; demais tipos recebem o menor score (aprovação por regra) | Fraude só existe nesses dois tipos; comparação na mesma validação completa |
| 18 | Padronização nos modelos de anomalia | Valores padronizados **cortados em ±5** | Evita que colunas binárias raras (`amount_equals_old_org`, desvio 0,0012, z=578 nas fraudes) distorçam toda a reconstrução. Efeito medido: Isolation Forest melhorou (C: 0,076 → 0,129); Autoencoder no B ficou em 0,99 (0,9913 → 0,9924), ou seja, detecta o atalho "valor = saldo inteiro", que quase não existe nas legítimas |
| 15 | Treino de desenvolvimento | Todas as fraudes + 20% das legítimas (`train.legit_frac`); rodada final com 100% | Economiza unidades do Colab; avaliação sempre na validação completa |

## Achados da EDA (24/11/2026)
- 8.213 fraudes (0,129%), só em `TRANSFER` (4.097) e `CASH_OUT` (4.116); ~12,1 bi em valor fraudado.
- Padrão: tomada de conta + esvaziamento (97,8% das fraudes têm valor = saldo inteiro; legítimas: 0,0001%).
- Vazamentos do simulador: saldo de origem "fecha" em 99,5% das fraudes (e em 9% das legítimas); fraude com teto de 10 mi (287 casos); 16 fraudes com valor 0; fraude a ritmo constante (~10/h) enquanto o volume cai após o step ~405.
- `isFlaggedFraud` marca só 16 transações (0,2% das fraudes): fora das features.
- Baseline (Regressão Logística): PR-AUC ~0,999 já sem saldos pós-transação → cenário fácil está saturado.

## Resultados da Fase 3 (validação, split por linhas, treino com 20% das legítimas)
PR-AUC: cenários A/B ~1,0 para RF e XGBoost (saturado). Cenário C: LogReg 0,21 · RF 0,78 · XGBoost 0,83. Cenário D: 0,20 · 0,77 · 0,81.
Recall em valor (melhor F1): C — XGBoost 96,5%, RF 95,8%, LogReg 82,1%. XGBoost treina em ~3 s (GPU) vs ~75 s do RF.
No cenário B o XGBoost depende 85% de `amount_equals_old_org` (atalho do simulador): sistema frágil a mudança de comportamento do fraudador.

## Resultados da Fase 4 (validação, após o corte ±5)
Isolation Forest PR-AUC: B 0,069 · C 0,129 · D 0,034. Autoencoder: B 0,992 · C 0,010 · D 0,013.
O Autoencoder (sem rótulo) só detecta bem quando o atalho do simulador está presente; sem ele, fica próximo do acaso e gera ~0,2% de falsos positivos. Ambos ficam bem abaixo dos supervisionados nos cenários difíceis (XGBoost C: 0,83).

## Decisões da Fase 5
19. **Avaliação final** nos cenários B (principal) e C (sem o atalho do esvaziamento), com retreino usando 100% das legítimas (`FRAC_FINAL`); o teste só entra aqui.
20. **Ensemble** ajustado só na validação: scores levados à escala `-log10(1 - ECDF_val)` (preserva a cauda das fraudes); stacking por Regressão Logística balanceada e média ponderada (grade de 0,25). Avaliação interna em 2 blocos temporais; ajuste final na validação inteira. Variante só com supervisionados isola a contribuição dos modelos de anomalia.
21. **Impacto financeiro** em dois níveis (sinalizar se score >= corte): perda evitada = 95% do valor das fraudes sinalizadas; custo = custo de investigação por alerta. Corte escolhido na validação e aplicado no teste. Falsos positivos reponderados para a taxa real de fraude (0,129%). Como o valor médio de fraude do PaySim (~1,47 mi) é muito maior que um custo de investigação realista, o ótimo tende a sinalizar quase tudo; por isso há uma grade de custos (50, 500, 5.000, 50.000). Três níveis (autenticação) ficam para a Fase 6.

## Decisões da Fase 6
22. **Três níveis** sobre o score do XGBoost: aprovar (s < t_auth), autenticar (t_auth <= s < t_inv), investigar (s >= t_inv). Custo esperado = custo operacional (R$ 2 por autenticação, R$ 50 por investigação) + perda residual (fraude aprovada perde 100% do valor; autenticada, 30%; investigada, 5%). Os dois cortes são o **ótimo exato** (o custo é separável, então a busca cobre todos os cortes possíveis), escolhidos na validação e medidos no teste.
23. **Robustez**: além do benefício líquido, medimos o "arrependimento" (diferença para o melhor corte possível no próprio teste) e a carga operacional (% de transações em cada nível). Grade de custos 25:1: (50, 2), (500, 20), (5.000, 200), (50.000, 2.000).
24. **Finalidade: portfólio para entrevistas**, sem intenção de produção. O painel deve ser usado por um entrevistador na hora: botões e controles interativos, boa usabilidade e insights rápidos (poucos cliques para ver valor).
25. **Modelo do painel: só o cenário B** (XGBoost salvo em `models/final`); o cenário C não será retreinado. As limitações (dependência do atalho do simulador) ficam explícitas no painel e na documentação.
26. **Publicação**: link público no **Streamlit Community Cloud** (repositório no GitHub). O painel não usa o CSV original: lê `app/data` (modelo XGBoost B em JSON nativo, portável entre versões; amostra do teste com todas as fraudes + 30 mil legítimas e pesos que recompõem a taxa real; exemplos; cortes; resultados).
27. **Usabilidade**: tela inicial = resumo com números-chave e cards por nível; botões de exemplos prontos (fraude clássica, de alto valor, difícil, pagamento comum, transferência grande, falso alarme) com explicação SHAP nativa do XGBoost; botão "esvaziar a conta" para mostrar o principal sinal; política ajustável na barra lateral (recomendada, por custos ou cortes manuais) e cenário B/C para a análise da carteira. A aba de transação sempre usa o modelo B.
28. **API** FastAPI (`/health`, `/score` com motivos, `/score/lote`) com os mesmos módulos do painel; serve para demonstrar integração, não é publicada.
29. **Regra de precaução** antes do modelo: valores acima do máximo do PaySim (valor ~R$ 92,4 mi, saldo de origem ~R$ 59,6 mi, saldo de destino ~R$ 356 mi) vão para investigação, porque o modelo não tem referência fora dessa faixa (árvores não extrapolam). "Valor maior que o saldo" fica só como aviso: no PaySim isso é comum em transações legítimas (saldo zerado), e uma regra marcaria muitas delas.
Limitação: só o modelo do cenário B foi salvo (`models/final`); os cortes do cenário C valem para o XGBoost C, que seria preciso retreinar e salvar se for usado no painel.

## Resultados da Fase 5 (teste, treino com 100% das legítimas)
PR-AUC no teste — cenário B: LogReg 0,9998 · RF 1,0 · XGBoost 1,0 · IF 0,22 · AE 0,978. Cenário C: LogReg 0,468 · RF 0,817 · **XGBoost 0,920** · IF 0,306 · AE 0,026. Recall em valor (melhor F1), C: XGBoost 98,2% · RF 97,0% · LogReg 88,6%. Obs.: o teste tem 0,34% de fraude (validação 0,06%), então a PR-AUC não é comparável entre os dois.
Ensemble (C): média ponderada colapsou no XGBoost (peso 1,0; 0,9200); stacking ficou pior (0,888, recall em valor ~88%, FPR 0,07% vs 0,02%). Isolation Forest e Autoencoder tiveram peso zero e não agregam (stack com 5 = 0,8885 vs só supervisionados = 0,8872).
**Modelo recomendado nesta fase: XGBoost sozinho.**
Impacto financeiro (teste: 4.250 fraudes, ~6,68 bi; perda evitada limitada a 95% pela premissa de captura): em B todos os modelos poupam ~95% com ~4.250 alertas. Em C o benefício líquido varia pouco entre modelos (~0,1% a custo 50), mas a carga de alertas difere: a custo 50, XGBoost 16.336 alertas (4.175 fraudes) vs LogReg 132.913; a custo 5.000, XGBoost 5.634 alertas (94,4% poupado); a custo 50.000, 3.930 alertas (93,6%). O valor médio de fraude (~1,57 mi) é muito maior que qualquer custo da grade, então o ótimo sempre sinaliza quase todas as fraudes: o dinheiro poupado quase não discrimina modelos, e o que os separa é a carga de alertas/precisão.
Para a Fase 6: incluir custo de atrito do falso positivo (possivelmente proporcional ao valor) e usar a precisão/carga de alertas como critério dos níveis.

## Resultados da Fase 6 (XGBoost, cortes da validação, medidos no teste)
**Cenário B:** praticamente não há zona cinzenta. Os dois cortes coincidem (~0,9996): investigam-se 4.247 transações, todas fraude (precisão 100%, 0,13% das transações reais), e escapam só 3 fraudes (0,01% do valor). O nível de autenticação não é usado e o ganho sobre dois níveis é zero (17,8 mil só no custo extremo).
**Cenário C, custo padrão (50 / 2):** t_auth = 0,0021 e t_inv = 0,0777. Aprovar: 97,2% das transações reais (9 fraudes escapam, 0,003% do valor). Autenticar: 1,9% (84 fraudes; precisão 0,36%). Investigar: 0,9% (4.157 fraudes; precisão 30% no teste, ~14% na taxa de fraude real). Valor poupado 94,98% (teto de 95%).
**Valor do terceiro nível:** ganho sobre só investigar de 2,0 mi (custo 50/2) a 80,4 mi (custo 50.000/2.000), ou seja, de 0,03% a 1,3% do benefício; cresce com o custo de investigação. Em valor, quase tudo é pego na investigação (99,9% do valor fraudado); a autenticação protege sobretudo em quantidade de fraudes (a 50.000/2.000: 1.086 fraudes autenticadas contra 230 aprovadas).
**Robustez:** o arrependimento (corte da validação vs melhor corte do teste) é de 0,00% a 0,02%: para o XGBoost os cortes se transferem bem, ao contrário do Random Forest na Fase 5.
**Carga operacional (C):** de 1,9% autenticadas e 0,9% investigadas (50/2) até 0,3% e 0,1% (50.000/2.000). Alta para um banco real porque o valor médio de fraude do PaySim (~1,5 mi) é muito maior que os custos assumidos.
Cortes salvos em `models/final/thresholds.json` (`null` = nível nunca usado).

## Premissas a lembrar
- Em dados PaySim, fraude só ocorre em `TRANSFER` e `CASH_OUT`.
- `isFlaggedFraud` é regra do simulador e não entra como feature.
- Valores de custo (investigação, atrito) são premissas, não dados reais.
- Metas quantitativas serão definidas após o baseline.

## Fases
1. **Fundação** — concluída
2. **EDA e baseline** — concluída (split por linhas; cenários A–D)
3. **Modelos supervisionados** — concluída
4. **Modelos de anomalia** — concluída
5. **Comparação final no teste, ensemble e impacto financeiro** — concluída (notebook 04)
6. **Thresholds por custo esperado** — concluída (notebook 05)
7. **Scoring + API e painel Streamlit** — concluída: https://antifraude-paysim.streamlit.app · código em https://github.com/elieserribeiro/fraud-detection-paysim
8. **Documentação técnica** — README de portfólio e `docs/relatorio_tecnico.md`
