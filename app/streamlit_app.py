"""Painel de detecção de fraude bancária (PaySim) — MVP de portfólio.

Rode com:  streamlit run app/streamlit_app.py
Os dados e o modelo vêm de ``app/data`` (gerados pelo notebook 06).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT.parent / "src"))

from fraud.features import build_features  # noqa: E402
from fraud.scoring import (REQUIRED_COLUMNS, VARIANT, contributions, load_model, logit,  # noqa: E402
                           score_transactions, top_reasons)
from fraud.thresholds import LEVELS, decision_levels, evaluate_levels, optimize_levels  # noqa: E402

st.set_page_config(page_title="Antifraude PaySim", page_icon="🛡️", layout="wide")

COLORS = {"aprovar": "#2e9e6b", "autenticar": "#e8a317", "investigar": "#d64545"}
NOMES = {"aprovar": "Aprovar", "autenticar": "Autenticar", "investigar": "Investigar"}
ICONES = {"aprovar": "✅", "autenticar": "🔐", "investigar": "🚨"}
DESC = {"aprovar": "Passa direto, sem atrito para o cliente.",
        "autenticar": "Pede uma confirmação extra (senha, biometria, código).",
        "investigar": "Segue para a equipe de investigação antes de concluir."}
MODELOS = {"logreg": "Regressão Logística", "random_forest": "Random Forest", "xgboost": "XGBoost",
           "isolation_forest": "Isolation Forest", "autoencoder": "Autoencoder"}
ENSEMBLES = {"media_ponderada_5": "Média ponderada (5 modelos)", "stack_5": "Stacking (5 modelos)",
             "stack_supervisionados": "Stacking (só supervisionados)"}
NOME_MODELO = {**MODELOS, **ENSEMBLES}
INV_OPCOES = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
AUTH_OPCOES = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
PROB_OPCOES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9996, 0.9999]


# ---------- utilidades ----------
def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def brl(x):
    a = abs(x)
    if a >= 1e9:
        return f"R$ {num(x / 1e9, 2)} bi"
    if a >= 1e6:
        return f"R$ {num(x / 1e6, 2)} mi"
    if a >= 1e3:
        return f"R$ {num(x / 1e3, 1)} mil"
    return f"R$ {num(x, 2)}"


def pct(x, dec=1):
    return f"{num(x, dec)}%"


def prob_txt(p):
    return pct(100 * p, 2 if p < 0.999 else 3)


# ---------- dados (em cache) ----------
@st.cache_resource
def get_model():
    return load_model(DATA / "xgboost_B.json")


@st.cache_data
def get_data():
    return {
        "amostra": pd.read_parquet(DATA / "demo_amostra.parquet"),
        "meta": json.loads((DATA / "meta.json").read_text(encoding="utf-8")),
        "thr": json.loads((DATA / "thresholds.json").read_text(encoding="utf-8")),
        "exemplos": json.loads((DATA / "exemplos.json").read_text(encoding="utf-8")),
        "modelos": pd.read_csv(DATA / "final_5_modelos.csv"),
        "ensemble": pd.read_csv(DATA / "final_ensemble.csv"),
        "impacto": pd.read_csv(DATA / "final_impacto_financeiro.csv"),
        "cortes": pd.read_csv(DATA / "final_thresholds_3_niveis.csv"),
    }


try:
    model = get_model()
    D = get_data()
except FileNotFoundError as e:
    st.error(f"Arquivo não encontrado: {e.filename}. Rode o notebook 06 e coloque a pasta `app/data` no projeto.")
    st.stop()

am, meta, thr = D["amostra"], D["meta"], D["thr"]
INV_CATCH = thr["premissas"]["investigation_catch_rate"]
AUTH_CATCH = thr["premissas"]["authentication_catch_rate"]
W = np.where(am["isFraud"].values == 1, 1.0, am["peso"].values * meta["peso_legitima_taxa_real"])
Y, AMT = am["isFraud"].values, am["amount"].values
PRESETS = thr["thresholds"]["B"]
PADRAO = next(k for k, v in PRESETS.items() if v["padrao"])

# Se nem a legítima mais suspeita passa do corte, o botão "Falso alarme" não faria sentido
_t_auth_padrao = PRESETS[PADRAO]["t_auth"]
for _e in D["exemplos"]:
    if _e["titulo"] == "Falso alarme" and _t_auth_padrao is not None and _e["score"] < _t_auth_padrao:
        _e["titulo"] = "Legítima mais suspeita"
        _e["descricao"] = ("A transação legítima com o maior score da amostra: mesmo ela fica abaixo dos cortes, "
                           "ou seja, nenhum falso alarme no cenário B.")


def preset_nome(k):
    v = PRESETS[k]
    return f"Investigar R$ {num(v['custo_investigacao'])} · Autenticar R$ {num(v['custo_autenticacao'], 0)}" + \
        (" (padrão)" if v["padrao"] else "")


def cap(v):
    return np.inf if v is None else float(v)


# ---------- barra lateral: política de decisão ----------
with st.sidebar:
    st.markdown("## 🛡️ Antifraude PaySim")
    st.caption("MVP de portfólio. Dados sintéticos (PaySim), sem uso em produção.")
    st.markdown("### Política de decisão")
    cen = st.radio("Dados da análise", ["B", "C"], horizontal=True,
                   format_func=lambda c: {"B": "B · realista", "C": "C · estresse"}[c],
                   help="B: informações disponíveis antes da transação (inclui o atalho 'esvaziar a conta'). "
                        "C: remove esse atalho e mostra o desempenho em um cenário mais difícil.")
    modo = st.radio("Como definir os cortes?", ["Recomendado", "Escolher custos", "Cortes manuais"],
                    help="Recomendado: cortes validados no notebook 05. Escolher custos: recalcula os cortes "
                         "ideais para os custos informados. Manuais: você define os dois cortes de risco.")
    if modo == "Recomendado":
        preset = st.selectbox("Cenário de custo", list(PRESETS), index=list(PRESETS).index(PADRAO),
                              format_func=preset_nome)
        inv_cost, auth_cost = PRESETS[preset]["custo_investigacao"], PRESETS[preset]["custo_autenticacao"]
    elif modo == "Escolher custos":
        inv_cost = st.select_slider("Custo por investigação (R$)", INV_OPCOES, value=50)
        auth_cost = st.select_slider("Custo por autenticação (R$)", AUTH_OPCOES, value=2)
        st.caption("Cortes recalculados nesta amostra de demonstração (não são os validados).")
    else:
        inv_cost, auth_cost = PRESETS[PADRAO]["custo_investigacao"], PRESETS[PADRAO]["custo_autenticacao"]
        m_a = st.select_slider("Autenticar a partir de (prob. de fraude)", PROB_OPCOES, value=0.01,
                               format_func=prob_txt)
        m_i = st.select_slider("Investigar a partir de (prob. de fraude)", PROB_OPCOES, value=0.9,
                               format_func=prob_txt)
        if m_i < m_a:
            st.warning("O corte de investigação não pode ser menor que o de autenticação; usando o mesmo valor.")
            m_i = m_a
    st.divider()
    st.caption("Captura assumida: investigação 95% · autenticação 70%.")


def policy_for(c):
    if modo == "Recomendado":
        v = thr["thresholds"][c][preset]
        return cap(v["t_auth"]), cap(v["t_inv"])
    if modo == "Escolher custos":
        o = optimize_levels(Y, AMT, am[f"score_{c}"].values, inv_cost, auth_cost, INV_CATCH, AUTH_CATCH, weights=W)
        return o["t_auth"], o["t_inv"]
    return float(m_a), float(m_i)


POL = {c: policy_for(c) for c in ("B", "C")}
t_auth, t_inv = POL[cen]
score_col = f"score_{cen}"


def evaluate(c, ta, ti):
    return evaluate_levels(Y, AMT, am[f"score_{c}"].values, ta, ti, inv_cost, auth_cost, INV_CATCH, AUTH_CATCH,
                           weights=W)


EV = evaluate(cen, t_auth, t_inv)
NIV = {r["nivel"]: r for r in EV["niveis"]}
TOTAL_FRAUDES = int(Y.sum())


def level_card(nivel):
    r = NIV[nivel]
    prec = r["fraudes"] / r["transacoes"] if r["transacoes"] else None
    st.markdown(
        f"""<div style="border-top:5px solid {COLORS[nivel]};background:#f4f6fa;border-radius:8px;padding:14px 16px;">
        <div style="font-size:1.05rem;font-weight:700;color:{COLORS[nivel]}">{ICONES[nivel]} {NOMES[nivel]}</div>
        <div style="font-size:1.7rem;font-weight:700;color:#1b2430">{pct(r['pct_transacoes_reais'], 2)}</div>
        <div style="color:#5b6573;font-size:.85rem;margin-bottom:6px">das transações reais</div>
        <div style="font-size:.9rem;color:#1b2430">🎯 <b>{num(r['fraudes'])}</b> fraudes · {brl(r['valor_fraude'])}</div>
        <div style="font-size:.8rem;color:#5b6573;margin-top:4px">{DESC[nivel]}</div></div>""",
        unsafe_allow_html=True)


def risk_meter(score, ta, ti, height=150):
    lo, hi = -10.5, 10.5
    lt, li = float(np.clip(logit(ta), lo, hi)), float(np.clip(logit(ti), lo, hi))
    lt = min(lt, li)
    fig = go.Figure()
    for x0, x1, nivel in [(lo, lt, "aprovar"), (lt, li, "autenticar"), (li, hi, "investigar")]:
        if x1 > x0:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=COLORS[nivel], opacity=0.28, line_width=0)
    x = float(np.clip(logit(score), lo + 0.6, hi - 0.6))
    fig.add_trace(go.Scatter(x=[x], y=[0.5], mode="markers", marker=dict(symbol="diamond", size=22, color="#1b2430",
                                                                          line=dict(color="white", width=2)),
                             hovertemplate=f"Risco: {prob_txt(score)}<extra></extra>", showlegend=False))
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    fig.update_xaxes(range=[lo, hi], tickvals=[float(logit(t)) for t in ticks],
                     ticktext=[prob_txt(t) for t in ticks], title="Probabilidade de fraude (escala logarítmica)")
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def fmt_valor(chave, v):
    if chave == "amount_log":
        return brl(np.expm1(v))
    if chave in ("oldbalanceOrg", "oldbalanceDest"):
        return brl(v)
    if chave.startswith("type_") or chave in ("orig_balance_zero", "dest_balance_zero", "amount_equals_old_org"):
        return "sim" if v >= 0.5 else "não"
    if chave == "amount_to_old_org":
        return f"{num(v, 2)}×"
    return num(v)


# ---------- cabeçalho ----------
st.markdown(
    """<div style="background:linear-gradient(120deg,#13294b,#1f6feb);border-radius:12px;padding:22px 26px;color:white;margin-bottom:10px">
    <div style="font-size:1.7rem;font-weight:800">🛡️ Detecção e prevenção de fraude bancária</div>
    <div style="opacity:.92;margin-top:4px">Cada transação recebe um score de risco e cai em um de três níveis:
    <b>aprovar</b>, <b>autenticar</b> ou <b>investigar</b>. Mexa nos controles da barra lateral e veja o impacto na hora.</div></div>""",
    unsafe_allow_html=True)

tab_res, tab_tx, tab_pol, tab_lote, tab_mod, tab_sobre = st.tabs(
    ["📊 Resumo", "🔍 Testar uma transação", "⚖️ Política e impacto", "📁 Lote (CSV)", "🤖 Modelos", "ℹ️ Sobre"])

# ================= RESUMO =================
with tab_res:
    st.caption(f"Teste de {num(meta['n_teste'])} transações, das quais {num(meta['n_fraudes_teste'])} são fraudes "
               f"({brl(meta['valor_fraudado_teste'])}). Números recompostos para a taxa de fraude real do PaySim "
               f"({pct(100 * meta['taxa_fraude_real'], 3)}). Cenário {cen}.")
    acao = NIV["autenticar"]["pct_transacoes_reais"] + NIV["investigar"]["pct_transacoes_reais"]
    barradas = NIV["autenticar"]["fraudes"] + NIV["investigar"]["fraudes"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor fraudado poupado", pct(EV["valor_poupado_%"], 1), help="Perda evitada ÷ valor total fraudado.")
    c2.metric("Fraudes barradas", f"{num(barradas)} de {num(TOTAL_FRAUDES)}",
              help="Fraudes que caíram em autenticação ou investigação.")
    c3.metric("Transações com ação", pct(acao, 2), help="Fatia das transações que não passam direto.")
    c4.metric("Benefício líquido", brl(EV["beneficio_liquido"]), help="Perda evitada menos custos operacionais.")
    st.markdown("#### Para onde vai cada transação")
    k1, k2, k3 = st.columns(3)
    with k1:
        level_card("aprovar")
    with k2:
        level_card("autenticar")
    with k3:
        level_card("investigar")

    md_ = D["modelos"]
    t = md_[md_["split"] == "test"].set_index(["cenario", "modelo"])
    st.markdown("#### O que o projeto mostra")
    i1, i2, i3 = st.columns(3)
    i1.info(f"**XGBoost lidera.** PR-AUC {num(t.loc[('B','xgboost'),'pr_auc'], 3)} no cenário B e "
            f"{num(t.loc[('C','xgboost'),'pr_auc'], 2)} mesmo sem o atalho do simulador (C), com "
            f"{pct(100 * t.loc[('C','xgboost'),'bestF1_value_recall'], 0)} do valor fraudado recuperado.")
    i2.warning(f"**Anomalia não basta.** Sem rótulos, Isolation Forest e Autoencoder chegam a PR-AUC "
               f"{num(t.loc[('C','isolation_forest'),'pr_auc'], 2)} e {num(t.loc[('C','autoencoder'),'pr_auc'], 2)} "
               f"no cenário C. O Autoencoder só brilha no B, onde o atalho existe.")
    i3.success("**Custo importa.** Com o valor médio de fraude muito acima do custo de investigar, quase toda "
               "fraude vale ser vista. O terceiro nível (autenticação) ajuda mais quando investigar fica caro.")

# ================= TESTAR UMA TRANSAÇÃO =================
with tab_tx:
    ss = st.session_state
    defaults = {"f_type": "TRANSFER", "f_amount": 250000.0, "f_old_org": 300000.0, "f_old_dest": 0.0,
                "f_day": 10, "f_hour": 14, "f_example": None}
    for k, v in defaults.items():
        ss.setdefault(k, v)

    def load_example(i):
        e = D["exemplos"][i]
        ss.f_type, ss.f_amount = e["type"], float(round(e["amount"], 2))
        ss.f_old_org, ss.f_old_dest = float(round(e["oldbalanceOrg"], 2)), float(round(e["oldbalanceDest"], 2))
        ss.f_day, ss.f_hour = int(min(e["step"] // 24, 30)), int(e["step"] % 24)
        ss.f_example = i

    def drain_account():
        ss.f_amount = float(ss.f_old_org)
        ss.f_example = None

    st.markdown("**1. Escolha um exemplo pronto** (ou edite os campos abaixo)")
    for rotulo, eh_fraude in [("🔴 Fraudes", True), ("🟢 Legítimas", False)]:
        idx = [i for i, e in enumerate(D["exemplos"]) if e["eh_fraude"] == eh_fraude]
        lab, *cols = st.columns([1.1] + [2] * max(len(idx), 1), vertical_alignment="center")
        lab.markdown(f"<span style='font-size:.85rem;color:#5b6573'>{rotulo}</span>", unsafe_allow_html=True)
        for col, i in zip(cols, idx):
            e = D["exemplos"][i]
            col.button(e["titulo"], key=f"ex{i}", on_click=load_example, args=(i,), help=e["descricao"],
                       width="stretch")

    form, result = st.columns([1, 1.25], gap="large")
    with form:
        st.markdown("**2. Dados da transação**")
        st.selectbox("Tipo", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], key="f_type",
                     format_func=lambda x: {"TRANSFER": "Transferência", "CASH_OUT": "Saque", "PAYMENT": "Pagamento",
                                            "CASH_IN": "Depósito", "DEBIT": "Débito"}[x])
        st.number_input("Valor (R$)", min_value=0.0, step=1000.0, format="%.2f", key="f_amount")
        st.number_input("Saldo da conta de origem (R$)", min_value=0.0, step=1000.0, format="%.2f", key="f_old_org")
        st.number_input("Saldo da conta de destino (R$)", min_value=0.0, step=1000.0, format="%.2f", key="f_old_dest")
        d1, d2 = st.columns(2)
        d1.slider("Dia", 0, 30, key="f_day")
        d2.slider("Hora", 0, 23, key="f_hour")
        st.button("💥 Esvaziar a conta (valor = saldo)", on_click=drain_account, width="stretch",
                  help="Veja o principal sinal de fraude do PaySim: transferir o saldo inteiro.")

    row = pd.DataFrame([{"step": ss.f_day * 24 + ss.f_hour, "type": ss.f_type, "amount": ss.f_amount,
                         "oldbalanceOrg": ss.f_old_org, "oldbalanceDest": ss.f_old_dest}])
    s = float(score_transactions(model, row)[0])
    ta_b, ti_b = POL["B"]
    nivel = LEVELS[int(decision_levels([s], ta_b, ti_b)[0])]
    X = build_features(row, VARIANT)
    reasons = top_reasons(contributions(model, row).iloc[0], X.iloc[0], k=6)

    with result:
        st.markdown("**3. Decisão do modelo**")
        st.markdown(
            f"""<div style="background:{COLORS[nivel]};color:white;border-radius:10px;padding:14px 18px;">
            <div style="font-size:1.6rem;font-weight:800">{ICONES[nivel]} {NOMES[nivel]}</div>
            <div style="opacity:.95">Risco de fraude: <b>{prob_txt(s)}</b> · {DESC[nivel]}</div></div>""",
            unsafe_allow_html=True)
        st.plotly_chart(risk_meter(s, ta_b, ti_b), width="stretch", config={"displayModeBar": False})
        ex = ss.f_example
        if ex is not None:
            e = D["exemplos"][ex]
            st.caption(f"Exemplo: **{e['titulo']}**. Rótulo real: {'fraude' if e['eh_fraude'] else 'transação legítima'}.")
        st.markdown("**Por que essa decisão?** (valores SHAP: o que empurrou o risco para cima ou para baixo)")
        r = reasons.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=r["contribuicao"], y=[f"{a} = {fmt_valor(k, v)}" for a, k, v in zip(r["atributo"], r["chave"], r["valor"])],
            orientation="h", marker_color=["#d64545" if c > 0 else "#2e9e6b" for c in r["contribuicao"]],
            hovertemplate="%{y}<br>contribuição: %{x:.2f}<extra></extra>"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=5, b=30), xaxis_title="← reduz o risco | aumenta o risco →",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        top = reasons.iloc[0]
        st.caption(f"Principal fator: **{top['atributo']}**, que {'aumentou' if top['contribuicao'] > 0 else 'reduziu'} o risco. "
                   f"Esta aba sempre usa o modelo do cenário B, com os cortes da barra lateral para o cenário B.")

# ================= POLÍTICA E IMPACTO =================
with tab_pol:
    st.markdown("#### Como o modelo separa fraudes de transações normais")
    sc, w_ = am[score_col].values, W
    bins = np.linspace(-12, 12, 49)
    lg = logit(sc)
    hf, _ = np.histogram(lg[Y == 1], bins=bins)
    hl, _ = np.histogram(lg[Y == 0], bins=bins, weights=w_[Y == 0])
    mid = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    fig.add_bar(x=mid, y=np.where(hl > 0, hl, np.nan), name="Legítimas (taxa real)", marker_color="#8fa3bf")
    fig.add_bar(x=mid, y=np.where(hf > 0, hf, np.nan), name="Fraudes", marker_color="#d64545")
    for t_, nome, cor in [(t_auth, "autenticar", COLORS["autenticar"]), (t_inv, "investigar", COLORS["investigar"])]:
        if np.isfinite(t_):
            fig.add_vline(x=float(logit(t_)), line_dash="dash", line_color=cor, annotation_text=NOMES[nome],
                          annotation_position="top")
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    fig.update_xaxes(tickvals=[float(logit(t)) for t in ticks], ticktext=[prob_txt(t) for t in ticks],
                     title="Probabilidade de fraude dada pelo modelo")
    ymax = max(float(np.nanmax(hl)), float(hf.max()), 1.0)
    fig.update_yaxes(type="log", title="Transações (escala log)", range=[-0.3, np.log10(ymax) + 0.3])
    fig.update_layout(barmode="overlay", height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Cenário {cen}. Quanto menos as barras vermelhas e cinzas se sobrepõem, mais fácil é decidir. "
               "As linhas tracejadas são os cortes da política atual.")

    st.markdown("#### Resultado por nível")
    tab = pd.DataFrame(EV["niveis"])
    tab["precisao"] = np.where(tab["transacoes"] > 0, 100 * tab["fraudes"] / tab["transacoes"].replace(0, 1), np.nan)
    show = pd.DataFrame({
        "Nível": [f"{ICONES[n]} {NOMES[n]}" for n in tab["nivel"]],
        "% das transações reais": [pct(v, 2) for v in tab["pct_transacoes_reais"]],
        "Fraudes": tab["fraudes"].map(num),
        "Valor das fraudes": tab["valor_fraude"].map(brl),
        "Custo operacional": tab["custo_operacional"].map(brl),
        "Perda residual": tab["perda_residual"].map(brl),
    })
    st.dataframe(show, hide_index=True, width="stretch")

    ev2 = evaluate(cen, *([optimize_levels(Y, AMT, am[score_col].values, inv_cost, auth_cost, INV_CATCH, AUTH_CATCH,
                                           weights=W)["t_two"]] * 2))
    g = EV["beneficio_liquido"] - ev2["beneficio_liquido"]
    a, b, c = st.columns(3)
    a.metric("Benefício com 3 níveis", brl(EV["beneficio_liquido"]))
    b.metric("Só investigar (2 níveis)", brl(ev2["beneficio_liquido"]),
             help="Melhor resultado possível sem o nível de autenticação, nesta amostra.")
    c.metric("Perda residual (fraudes não barradas)", brl(EV["perda_residual"]))
    st.caption(f"Ganho do nível de autenticação nesta política: **{brl(g)}**. Comparação feita na amostra de "
               "demonstração (teste); os cortes 'Recomendado' foram escolhidos na validação.")

    st.markdown("#### Como a carga de trabalho muda com o custo (cortes validados)")
    ct = D["cortes"]
    ct = ct[ct["cenario"] == cen].copy()
    ct["rotulo"] = ct.apply(lambda r: f"{num(r['custo_invest'])}/{num(r['custo_auth'])}", axis=1)
    fig = go.Figure()
    fig.add_bar(x=ct["rotulo"], y=ct["%_autenticar"], name="Autenticar", marker_color=COLORS["autenticar"])
    fig.add_bar(x=ct["rotulo"], y=ct["%_investigar"], name="Investigar", marker_color=COLORS["investigar"])
    fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Custo de investigação / autenticação (R$)", yaxis_title="% das transações reais",
                      yaxis_type="log", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

# ================= LOTE =================
with tab_lote:
    st.markdown("Suba um CSV com várias transações e veja o nível de decisão de cada uma.")
    st.caption(f"Colunas obrigatórias: {', '.join(REQUIRED_COLUMNS)}. A coluna opcional `isFraud` permite conferir os acertos. "
               "Usa o modelo do cenário B.")
    cA, cB = st.columns([1, 1])
    amostra_ex = pd.concat([am[am["isFraud"] == 1].sample(15, random_state=1) if (am["isFraud"] == 1).sum() >= 15
                            else am[am["isFraud"] == 1],
                            am[am["isFraud"] == 0].sample(185, random_state=1)]).sample(frac=1, random_state=2)
    amostra_ex = amostra_ex[REQUIRED_COLUMNS + ["isFraud"]].reset_index(drop=True)
    cA.download_button("⬇️ Baixar CSV de exemplo (200 transações)", amostra_ex.to_csv(index=False).encode("utf-8"),
                       "transacoes_exemplo.csv", "text/csv", width="stretch")
    usar_ex = cB.button("▶️ Usar o exemplo agora", width="stretch")
    up = st.file_uploader("Ou envie seu CSV", type="csv")
    lote = None
    if up is not None:
        try:
            lote = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Não consegui ler o arquivo: {e}")
    elif usar_ex or st.session_state.get("lote_exemplo"):
        st.session_state["lote_exemplo"] = True
        lote = amostra_ex.copy()
    if lote is not None:
        try:
            lote = lote.copy()
            lote["score"] = score_transactions(model, lote)
            ta_b, ti_b = POL["B"]
            lote["nivel"] = [NOMES[LEVELS[i]] for i in decision_levels(lote["score"].values, ta_b, ti_b)]
            cont = lote["nivel"].value_counts().reindex([NOMES[n] for n in LEVELS], fill_value=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Transações", num(len(lote)))
            for mm, n in zip((m2, m3, m4), LEVELS):
                mm.metric(f"{ICONES[n]} {NOMES[n]}", num(int(cont[NOMES[n]])))
            if "isFraud" in lote.columns:
                f = lote[lote["isFraud"] == 1]
                st.success(f"Das {num(len(f))} fraudes do arquivo, {num(int((f['nivel'] != 'Aprovar').sum()))} "
                           f"caíram em autenticação ou investigação.")
            vis = lote.sort_values("score", ascending=False).head(500).copy()
            vis["score"] = vis["score"].map(prob_txt)
            for c_ in ("amount", "oldbalanceOrg", "oldbalanceDest"):
                vis[c_] = vis[c_].map(lambda v: num(v, 2))
            vis = vis.rename(columns={"step": "Hora (step)", "type": "Tipo", "amount": "Valor (R$)",
                                      "oldbalanceOrg": "Saldo origem", "oldbalanceDest": "Saldo destino",
                                      "isFraud": "Fraude real", "score": "Risco", "nivel": "Decisão"})
            st.dataframe(vis, hide_index=True, width="stretch")
            st.download_button("⬇️ Baixar resultado completo", lote.to_csv(index=False).encode("utf-8"),
                               "resultado_scores.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))

# ================= MODELOS =================
with tab_mod:
    esc = st.radio("Cenário", ["B", "C"], horizontal=True, index=1,
                   format_func=lambda c: {"B": "B · realista (com o atalho do simulador)",
                                          "C": "C · estresse (sem o atalho)"}[c])
    md_ = D["modelos"]
    t = md_[(md_["split"] == "test") & (md_["cenario"] == esc)].copy()
    t["nome"] = t["modelo"].map(MODELOS)
    t = t.set_index("modelo").loc[list(MODELOS)]
    g1, g2 = st.columns(2)
    for col, campo, titulo, fmt in [(g1, "pr_auc", "PR-AUC no teste", lambda v: num(v, 3)),
                                    (g2, "bestF1_value_recall", "Recall em valor (melhor F1)", lambda v: pct(100 * v, 1))]:
        fig = go.Figure(go.Bar(x=t["nome"], y=t[campo], text=[fmt(v) for v in t[campo]], textposition="outside",
                               marker_color=["#1f6feb" if m == "xgboost" else "#9fb3d1" for m in t.index]))
        fig.update_layout(title=titulo, height=330, margin=dict(l=10, r=10, t=45, b=10),
                          yaxis=dict(range=[0, 1.12]), plot_bgcolor="rgba(0,0,0,0)")
        col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if esc == "B":
        st.info("No cenário B quase todos chegam perto de 1,0 porque o PaySim entrega o sinal 'valor = saldo inteiro'. "
                "Por isso o cenário C existe: ele remove esse atalho para ver quem realmente generaliza.")
    else:
        st.info("Sem o atalho, os modelos supervisionados de árvore se destacam e os de anomalia ficam perto do acaso.")

    st.markdown("#### Combinar modelos ajuda?")
    en = D["ensemble"]
    en = en[en["cenario"] == esc].copy()
    en["Modelo"] = en["modelo"].map(lambda m: NOME_MODELO.get(m, m))
    en = en.sort_values("pr_auc_teste", ascending=False)
    st.dataframe(pd.DataFrame({"Modelo": en["Modelo"], "PR-AUC (teste)": en["pr_auc_teste"].map(lambda v: num(v, 3)),
                               "Recall em valor": en["recall_valor_teste"].map(lambda v: pct(100 * v, 1)),
                               "Taxa de falsos positivos": en["fpr_teste"].map(lambda v: pct(100 * v, 3))}),
                 hide_index=True, width="stretch")
    st.caption("Resultado: stacking e média ponderada não superaram o XGBoost sozinho, e os modelos de anomalia "
               "receberam peso zero. Por isso o painel usa só o XGBoost.")

    st.markdown("#### Dinheiro poupado conforme o custo de investigar")
    im = D["impacto"]
    im = im[im["cenario"] == esc]
    fig = go.Figure()
    for m in ["xgboost", "random_forest", "logreg", "stack_5", "stack_supervisionados"]:
        d = im[im["modelo"] == m].sort_values("custo_por_alerta")
        if len(d):
            fig.add_scatter(x=d["custo_por_alerta"], y=d["valor_poupado_%"], mode="lines+markers",
                            name=NOME_MODELO.get(m, m))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log",
                      xaxis_title="Custo de investigação por alerta (R$)", yaxis_title="% do valor fraudado poupado",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

# ================= SOBRE =================
with tab_sobre:
    st.markdown(f"""
### Sobre o projeto
Sistema de monitoramento de transações que **atribui um score de risco** e decide entre **aprovar**, **pedir autenticação**
ou **investigar**, usando o dataset sintético PaySim (6,36 milhões de transações, 0,129% de fraude).

**Metodologia.** Split temporal por linhas (treino, validação e teste em ordem cronológica); pesos de classe para o desbalanceamento;
cinco modelos (Regressão Logística, Random Forest, XGBoost, Isolation Forest e Autoencoder); métricas PR-AUC, recall em valor
e taxa de falsos positivos; cortes de risco escolhidos pelo **menor custo esperado** (perda das fraudes + custo operacional).

**Como ler os cenários.** B usa só o que se sabe antes de aprovar. C remove os sinais de "esvaziar a conta", um atalho do simulador.

### Limitações (importante)
- **Dados sintéticos.** O PaySim tem atalhos que um banco real não teria; o cenário B parece quase perfeito por causa disso.
- **Fragilidade a mudança de padrão.** O modelo do painel (cenário B) depende fortemente de "valor = saldo inteiro"; uma fraude que não esvazie a conta tende a passar.
- **Sem histórico do cliente.** O PaySim não traz comportamento anterior de cada conta, o que ajudaria muito na vida real.
- **Custos e taxas de captura são premissas**, não dados reais (investigação R$ 50 / 95%, autenticação R$ 2 / 70%).
- **Modelos de anomalia fracos** nos cenários difíceis; ficam como comparação, não como parte da decisão.
- **Projeto de portfólio**, sem intenção de uso em produção.

### Como os números são recompostos
A amostra de demonstração tem todas as {num(meta['n_fraudes_teste'])} fraudes do teste e {num(meta['n_amostra'] - meta['n_fraudes_teste'])}
transações legítimas; cada legítima recebe um peso para recompor a taxa de fraude real.
""")
