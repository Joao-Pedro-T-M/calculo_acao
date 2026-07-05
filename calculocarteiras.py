# -*- coding: utf-8 -*-
import io
import zipfile
import warnings
import html
import re
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from result_card_component import load_styles, render_asset_card

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Carteiras, CAPM e Fronteira Eficiente",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 0) CONFIGURAÇÕES
# ============================================================
BASE_DIR = Path("outputs")
BASE_DIR.mkdir(parents=True, exist_ok=True)

MARKET_LABEL = "IBOV"
MARKET_SYMBOL = "^BVSP"
MAX_RESULTS = 30

st.title("Análise de carteiras, CAPM e fronteira eficiente")
st.caption(
    "Pesquise os ativos no Yahoo Finance, selecione quantos quiser, informe os pesos e gere as tabelas, gráficos e o Excel automaticamente."
)

from pathlib import Path

css_path = Path("result_card_component/frontend/style.css")
st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ============================================================
# 1) FUNÇÕES AUXILIARES
# ============================================================
def format_pct(x: float) -> str:
    return f"{x:.2%}"


def normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
    total = float(sum(weight_map.values()))
    if total <= 0:
        raise ValueError("A soma dos pesos precisa ser maior que zero.")
    return {k: v / total for k, v in weight_map.items()}


def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("ME").last().dropna(how="all")


def safe_download_prices(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    raw = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Não foi possível baixar os dados do Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise RuntimeError("A coluna Close não foi encontrada no retorno do yfinance.")
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        else:
            prices = raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices.sort_index()


def search_yfinance_assets(query: str, max_results: int = 30) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    try:
        result = yf.Search(query=query.strip(), max_results=max_results)
        quotes = getattr(result, "quotes", []) or []
    except Exception as e:
        st.error(f"Erro ao buscar ativos no Yahoo Finance: {e}")
        return pd.DataFrame()

    rows = []
    for q in quotes:
        if not isinstance(q, dict):
            continue
        symbol = q.get("symbol")
        if not symbol:
            continue
        rows.append({
            "symbol": symbol,
            "shortname": q.get("shortname", "") or "",
            "longname": q.get("longname", "") or "",
            "exchange": q.get("exchange", "") or "",
            "quoteType": q.get("quoteType", "") or "",
            "typeDisp": q.get("typeDisp", "") or "",
            "currency": q.get("currency", "") or "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


def make_portfolios(ativos: list[str], user_weights: dict[str, float]) -> dict[str, dict[str, float]]:
    n = len(ativos)
    equal = {a: 1 / n for a in ativos}

    portfolio_user = {a: float(user_weights.get(a, 0.0)) for a in ativos}
    portfolio_user = normalize_weights(portfolio_user)

    carteira_1 = equal

    carteira_3 = {a: 0.0 for a in ativos}
    if n == 1:
        carteira_3[ativos[0]] = 1.0
    else:
        carteira_3[ativos[0]] = 0.50
        restante = 0.50 / (n - 1)
        for a in ativos[1:]:
            carteira_3[a] = restante
    carteira_3 = normalize_weights(carteira_3)

    return {
        "Carteira_1": carteira_1,
        "Carteira_2": portfolio_user,
        "Carteira_3": carteira_3,
    }


def portfolio_metrics(retornos_mensais: pd.DataFrame, ativos: list[str], pesos: np.ndarray, matriz_cov: pd.DataFrame):
    retorno_carteira = retornos_mensais[ativos].mul(pesos, axis=1).sum(axis=1)
    retorno_esperado_mensal = float(retorno_carteira.mean())
    risco_mensal = float(retorno_carteira.std())
    risco_mensal_cov = float(np.sqrt(np.dot(pesos.T, np.dot(matriz_cov.values, pesos))))
    return retorno_carteira, retorno_esperado_mensal, risco_mensal, risco_mensal_cov


def save_matplotlib_figure(fig, filename: str) -> Path:
    path = BASE_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def display_matplotlib(fig, filename: str):
    save_matplotlib_figure(fig, filename)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def build_zip(paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)
    buffer.seek(0)
    return buffer.read()


def init_weight_inputs(selected_assets: list[str]):
    default = round(100 / len(selected_assets), 2) if selected_assets else 0
    for asset in selected_assets:
        key = f"peso_{asset}"
        if key not in st.session_state:
            st.session_state[key] = default


def filter_valid_assets(retornos: pd.DataFrame, ativos: list[str]) -> list[str]:
    validos = []
    for ativo in ativos:
        serie = pd.to_numeric(retornos[ativo], errors="coerce")
        if serie.dropna().shape[0] >= 2:
            validos.append(ativo)
    return validos


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def reset_confirmation():
    st.session_state.pesos_confirmados = False
    st.session_state.pop("user_weights", None)


def normalize_symbol_list(values):
    cleaned = []
    for v in values:
        if not isinstance(v, str):
            continue
        sym = v.split(" | ", 1)[0].strip()
        if sym and sym not in cleaned:
            cleaned.append(sym)
    return cleaned


def toggle_symbol(symbol: str, label: str = ""):
    symbol = symbol.strip()
    if not symbol:
        return

    if symbol in st.session_state.selected_symbols:
        st.session_state.selected_symbols = [s for s in st.session_state.selected_symbols if s != symbol]
        st.session_state.selected_labels.pop(symbol, None)
        peso_key = f"peso_{symbol}"
        if peso_key in st.session_state:
            del st.session_state[peso_key]
    else:
        st.session_state.selected_symbols.append(symbol)
        if label:
            st.session_state.selected_labels[symbol] = label

    reset_confirmation()
    st.rerun()


def cleanup_weight_keys(selected_assets: list[str]):
    valid_keys = {f"peso_{asset}" for asset in selected_assets}
    for key in list(st.session_state.keys()):
        if key.startswith("peso_") and key not in valid_keys:
            del st.session_state[key]


# ============================================================
# 2) SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Configuração da análise")
    start_date = st.date_input("Data inicial", value=date(2016, 1, 1))
    end_date = st.date_input("Data final", value=date.today())

    st.markdown("---")
    st.subheader("Ativos selecionados")
    st.markdown(
        f"<div class='selected-header'>Selecionados: {len(st.session_state.get('selected_symbols', []))}</div>",
        unsafe_allow_html=True
    )

    if st.session_state.get("selected_symbols"):
        for sym in st.session_state.get("selected_symbols", []):
            label = st.session_state.get("selected_labels", {}).get(sym, sym)
            st.markdown(
                f"""
                <div class="sidebar-selected-card">
                    <div class="selected-symbol">{html.escape(sym)}</div>
                    <div class="selected-label">{html.escape(label)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("Nenhum ativo selecionado ainda.")

    st.markdown("---")
# =========================================================
# 3) BUSCA + SELEÇÃO EM UM ÚNICO CARD
# =========================================================
if start_date >= end_date:
    st.error("A data inicial precisa ser anterior à data final.")
    st.stop()

if "search_df" not in st.session_state:
    st.session_state.search_df = pd.DataFrame()
if "last_search_term" not in st.session_state:
    st.session_state.last_search_term = ""
if "pesos_confirmados" not in st.session_state:
    st.session_state.pesos_confirmados = False
if "selected_symbols" not in st.session_state:
    st.session_state.selected_symbols = []
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = {}

st.subheader("Buscar e selecionar ativos")

with st.form(key="search_form", clear_on_submit=False):
    search_term = st.text_input(
        "Digite uma letra, nome ou código e pressione Enter",
        placeholder="Ex.: B, PETR, VALE, ITUB, WEGE...",
        value=st.session_state.get("last_search_term", ""),
    )
    submitted = st.form_submit_button("Buscar")

if submitted and search_term.strip():
    st.session_state.last_search_term = search_term.strip()
    with st.spinner("Buscando ativos..."):
        st.session_state.search_df = search_yfinance_assets(search_term.strip(), max_results=MAX_RESULTS)

search_df = st.session_state.search_df

if not search_df.empty:
    st.success(f"🔍 {len(search_df)} resultados encontrados para **'{st.session_state.last_search_term}'**")
    search_df = search_df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    search_df["label"] = search_df.apply(
        lambda r: f"{r['symbol']} | {r['shortname'] or r['longname']} | {r['exchange']} | {r['typeDisp']}",
        axis=1
    )

    st.markdown("### Resultados da busca")

    cards_per_row = 4

    for i in range(0, len(search_df), cards_per_row):
        row_df = search_df.iloc[i:i + cards_per_row]
        cols = st.columns(cards_per_row, gap="small")

        for col, (_, row) in zip(cols, row_df.iterrows()):
            with col:
                sym = str(row["symbol"])
                name = str(row["shortname"] or row["longname"] or "Sem nome")
                exchange = str(row["exchange"] or "—")
                typedisp = str(row["typeDisp"] or "—")
                currency = str(row["currency"] or "—")
                label = str(row["label"])
                selected = sym in st.session_state.selected_symbols

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="asset-card">
                            <div>
                                <div class="asset-topline">
                                    <div class="asset-symbol">{html.escape(sym)}</div>
                                    <span class="pill">{html.escape(typedisp or '—')}</span>
                                </div>
                                <div class="asset-name">{html.escape(name)}</div>
                                <div class="asset-meta">{html.escape(exchange)} • {html.escape(currency)}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button(
                        "Remover" if selected else "Selecionar",
                        key=f"toggle_{safe_key(sym)}",
                        use_container_width=True,
                        type="primary" if selected else "secondary"
                    ):
                        toggle_symbol(sym, label)

elif search_df.empty and st.session_state.last_search_term:
    st.warning(f"Nenhum resultado encontrado para **'{st.session_state.last_search_term}'**")

st.markdown("### Ativos selecionados")

if st.session_state.selected_symbols:
    st.markdown(
        f"<div class='selected-header'>Selecionados: {len(st.session_state.selected_symbols)}</div>",
        unsafe_allow_html=True
    )

    selected_items = []
    for sym in st.session_state.selected_symbols:
        label = st.session_state.selected_labels.get(sym, sym)
        parts = [p.strip() for p in str(label).split(" | ")]
        selected_name = parts[1] if len(parts) > 1 and parts[1] else sym
        selected_exchange = parts[2] if len(parts) > 2 and parts[2] else "—"
        selected_typedisp = parts[3] if len(parts) > 3 and parts[3] else "—"
        selected_items.append((sym, selected_name, selected_exchange, selected_typedisp, label))

    cards_per_row = 4

    for i in range(0, len(selected_items), cards_per_row):
        row_items = selected_items[i:i + cards_per_row]
        cols = st.columns(cards_per_row, gap="small")

        for col, item in zip(cols, row_items):
            with col:
                sym, name, exchange, typedisp, label = item
                selected = True

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="asset-card">
                            <div>
                                <div class="asset-topline">
                                    <div class="asset-symbol">{html.escape(sym)}</div>
                                    <span class="pill">{html.escape(typedisp or '—')}</span>
                                </div>
                                <div class="asset-name">{html.escape(name)}</div>
                                <div class="asset-meta">{html.escape(exchange)} • —</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button(
                        "Remover",
                        key=f"rm_{safe_key(sym)}",
                        use_container_width=True,
                        type="primary",
                    ):
                        toggle_symbol(sym)

else:
    st.info("Nenhum ativo selecionado ainda. Clique no card do resultado para selecionar.")

selected_display_names = st.session_state.selected_symbols

if not selected_display_names:
    st.stop()
# ============================================================
# 4) PESOS DINÂMICOS (com confirmação manual)
# ============================================================
cleanup_weight_keys(selected_display_names)
init_weight_inputs(selected_display_names)

st.markdown("### Pesos da carteira escolhida")
st.caption("Ajuste os pesos de cada ativo. A soma deve ser exatamente 100%.")

with st.container(border=True):
    cards_per_row = 2 if len(selected_display_names) <= 4 else 3
    cols = st.columns(cards_per_row, gap="small")

    for i, asset in enumerate(selected_display_names):
        label = st.session_state.selected_labels.get(asset, asset)
        parts = [p.strip() for p in str(label).split(" | ")]
        selected_name = parts[1] if len(parts) > 1 and parts[1] else asset
        selected_exchange = parts[2] if len(parts) > 2 and parts[2] else "—"
        selected_typedisp = parts[3] if len(parts) > 3 and parts[3] else "—"

        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="weight-item">
                    <div class="asset-topline">
                        <div>
                            <div class="weight-name">{html.escape(selected_name)}</div>
                            <div class="weight-meta">
                                {html.escape(selected_exchange)} • {html.escape(selected_typedisp)}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.number_input(
                f"{asset} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(f"peso_{asset}", round(100 / len(selected_display_names), 2))),
                step=0.1,
                key=f"peso_{asset}",
                label_visibility="collapsed",
            )

    user_weight_values = {asset: float(st.session_state[f"peso_{asset}"]) for asset in selected_display_names}
    total = sum(user_weight_values.values())
    is_valid = abs(total - 100.0) < 0.01

    st.markdown(
        f"""
        <div class="sum-badge">
            <div class="sum-badge-left">
                <span class="sum-badge-icon">Σ</span>
                <span class="sum-badge-text">Soma atual</span>
            </div>
            <div class="sum-badge-value">{total:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="confirmar-pesos-btn">', unsafe_allow_html=True)

    confirm = st.button(
        "Confirmar pesos",
        key="confirmar_pesos",
        disabled=not is_valid,
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

if confirm:
    user_weights = {k: v / 100.0 for k, v in user_weight_values.items()}
    st.session_state.user_weights = user_weights
    st.session_state.pesos_confirmados = True
    st.success("Pesos confirmados! Prosseguindo com a análise...")
else:
    st.session_state.pesos_confirmados = False
    if total > 0 and not is_valid:
        st.warning("A soma dos pesos deve ser exatamente 100% para prosseguir.")
    elif total == 0:
        st.error("A soma dos pesos não pode ser zero.")

if not st.session_state.get("pesos_confirmados", False):
    st.stop()

user_weights = st.session_state.user_weights

st.write("**Ativos selecionados:**", ", ".join(selected_display_names))
st.write("**Pesos confirmados:**", {k: f"{v:.2%}" for k, v in user_weights.items()})

# ============================================================
# 5) BAIXAR DADOS E PROCESSAR
# ============================================================
download_symbols = list(dict.fromkeys(selected_display_names + [MARKET_SYMBOL]))

with st.spinner("Baixando preços do Yahoo Finance..."):
    precos = safe_download_prices(download_symbols, str(start_date), str(end_date))

rename_map = {sym: sym for sym in selected_display_names}
rename_map[MARKET_SYMBOL] = MARKET_LABEL

precos = precos.rename(columns=rename_map)
precos = precos.sort_index()

ativos = [c for c in selected_display_names if c in precos.columns]
if len(ativos) == 0:
    st.error("Nenhum dos ativos selecionados retornou dados válidos do Yahoo Finance.")
    st.stop()

if MARKET_LABEL not in precos.columns:
    st.error("Não foi possível baixar o índice IBOV (^BVSP) para usar como mercado.")
    st.stop()

# ============================================================
# 6) PRECOS E RETORNOS
# ============================================================
pasta_saida = BASE_DIR
pasta_saida.mkdir(parents=True, exist_ok=True)

with st.spinner("Convertendo para dados mensais e calculando retornos..."):
    precos_mensais = to_monthly_prices(precos)
    retornos_mensais = precos_mensais.pct_change().dropna()
    retornos_mensais = retornos_mensais.rename(columns={MARKET_LABEL: "Retorno_Mercado_IBOV"})

    ativos = [a for a in ativos if a in retornos_mensais.columns]
    ativos = filter_valid_assets(retornos_mensais, ativos)

if len(ativos) == 0:
    st.error("Após filtrar os dados válidos, nenhum ativo permaneceu disponível para análise.")
    st.stop()

if len(ativos) < len(selected_display_names):
    st.warning("Alguns ativos não tiveram dados suficientes no período escolhido e foram removidos da análise.")
# ============================================================
# 7) ANÁLISE INDIVIDUAL DOS ATIVOS
# ============================================================
with st.spinner("Calculando análise individual dos ativos..."):
    analise_ativos = []
    for ativo in ativos:
        retorno_medio = retornos_mensais[ativo].mean()
        volatilidade = retornos_mensais[ativo].std()

        analise_ativos.append({
            "Ativo": ativo,
            "Retorno_Medio_Historico_Mensal": retorno_medio,
            "Retorno_Medio_Historico_Anual": (1 + retorno_medio) ** 12 - 1,
            "Volatilidade_Mensal": volatilidade,
            "Volatilidade_Anual": volatilidade * np.sqrt(12),
            "Retorno_Medio_Positive": retorno_medio > 0,
        })

    df_analise_ativos = pd.DataFrame(analise_ativos)

# ============================================================
# 8) COVARIÂNCIA E CORRELAÇÃO
# ============================================================
with st.spinner("Calculando matrizes de covariância e correlação..."):
    matriz_cov = retornos_mensais[ativos].cov()
    matriz_corr = retornos_mensais[ativos].corr()

# ============================================================
# 9) CARTEIRAS
# ============================================================
with st.spinner("Criando carteiras e calculando métricas..."):
    carteiras = make_portfolios(ativos, user_weights)

    carteira_escolhida_nome = "Carteira_2"
    pesos_carteira_escolhida = np.array([carteiras[carteira_escolhida_nome].get(a, 0) for a in ativos])

    resultados_carteiras = []
    for nome_carteira, pesos_dict in carteiras.items():
        pesos_array = np.array([pesos_dict.get(a, 0) for a in ativos])

        retorno_carteira, retorno_esperado_mensal, risco_mensal, risco_mensal_cov = portfolio_metrics(
            retornos_mensais, ativos, pesos_array, matriz_cov
        )

        resultados_carteiras.append({
            "Carteira": nome_carteira,
            **{f"Peso_{a}": pesos_dict.get(a, 0) for a in ativos},
            "Retorno_Esperado_Mensal": retorno_esperado_mensal,
            "Retorno_Esperado_Anual": (1 + retorno_esperado_mensal) ** 12 - 1,
            "Risco_Mensal": risco_mensal,
            "Risco_Mensal_Cov": risco_mensal_cov,
            "Risco_Anual": risco_mensal_cov * np.sqrt(12),
        })

    df_carteiras = pd.DataFrame(resultados_carteiras)

    retorno_carteira_escolhida = retornos_mensais[ativos].mul(pesos_carteira_escolhida, axis=1).sum(axis=1)
    retorno_hist_carteira_mensal = retorno_carteira_escolhida.mean()
    risco_carteira_mensal = retorno_carteira_escolhida.std()
    risco_carteira_anual = risco_carteira_mensal * np.sqrt(12)

    df_carteira_escolhida = pd.DataFrame([{
        "Carteira": carteira_escolhida_nome,
        **{f"Peso_{a}": carteiras[carteira_escolhida_nome].get(a, 0) for a in ativos},
        "Retorno_Historico_Mensal": retorno_hist_carteira_mensal,
        "Retorno_Historico_Anual": (1 + retorno_hist_carteira_mensal) ** 12 - 1,
        "Risco_Mensal": risco_carteira_mensal,
        "Risco_Anual": risco_carteira_anual,
    }])

    descricao_pesos = ", ".join([f"{a}: {carteiras[carteira_escolhida_nome].get(a, 0):.0%}" for a in ativos])

    justificativa_carteira = (
        f"A {carteira_escolhida_nome} foi escolhida por apresentar equilíbrio entre risco e retorno. "
        f"Seu risco mensal foi {risco_carteira_mensal:.4f} e o retorno médio mensal foi {retorno_hist_carteira_mensal:.4f}."
    )

# ============================================================
# 10) BETA E CAPM
# ============================================================
with st.spinner("Calculando Beta e CAPM dos ativos e da carteira..."):
    taxa_livre_risco_anual = 0.1450
    taxa_livre_risco_mensal = (1 + taxa_livre_risco_anual) ** (1 / 12) - 1
    retorno_mercado = retornos_mensais["Retorno_Mercado_IBOV"]

    resultados_beta = []
    for ativo in ativos:
        df = pd.concat([retornos_mensais[ativo], retorno_mercado], axis=1).dropna()
        df.columns = ["ativo", "mercado"]

        excesso_ativo = df["ativo"] - taxa_livre_risco_mensal
        excesso_mercado = df["mercado"] - taxa_livre_risco_mensal

        beta = excesso_ativo.cov(excesso_mercado) / excesso_mercado.var()

        ret_hist_mensal = df["ativo"].mean()
        ret_capm_mensal = taxa_livre_risco_mensal + beta * (df["mercado"].mean() - taxa_livre_risco_mensal)
        alpha_mensal = ret_hist_mensal - ret_capm_mensal

        resultados_beta.append({
            "Ativo": ativo,
            "Beta": beta,
            "Retorno_Historico_Mensal": ret_hist_mensal,
            "Retorno_Historico_Anual": (1 + ret_hist_mensal) ** 12 - 1,
            "Retorno_CAPM_Mensal": ret_capm_mensal,
            "Retorno_CAPM_Anual": (1 + ret_capm_mensal) ** 12 - 1,
            "Alpha_Mensal": alpha_mensal,
            "Alpha_Anual": (1 + alpha_mensal) ** 12 - 1,
            "Retorno_Mercado_Medio_Mensal": df["mercado"].mean(),
        })

    df_beta_ativos = pd.DataFrame(resultados_beta)

    df_carteira = pd.concat([retorno_carteira_escolhida, retorno_mercado], axis=1).dropna()
    df_carteira.columns = ["carteira", "mercado"]

    excesso_carteira = df_carteira["carteira"] - taxa_livre_risco_mensal
    excesso_mercado_carteira = df_carteira["mercado"] - taxa_livre_risco_mensal

    beta_carteira = excesso_carteira.cov(excesso_mercado_carteira) / excesso_mercado_carteira.var()
    ret_hist_carteira_mensal = df_carteira["carteira"].mean()
    ret_capm_carteira_mensal = taxa_livre_risco_mensal + beta_carteira * (df_carteira["mercado"].mean() - taxa_livre_risco_mensal)
    alpha_carteira_mensal = ret_hist_carteira_mensal - ret_capm_carteira_mensal

    df_beta_carteira_resumo = pd.DataFrame([{
        "Carteira": f"{carteira_escolhida_nome} ({descricao_pesos})",
        "Beta": beta_carteira,
        "Alpha_Mensal": alpha_carteira_mensal,
        "Alpha_Anual": (1 + alpha_carteira_mensal) ** 12 - 1,
        "Retorno_Historico_Mensal": ret_hist_carteira_mensal,
        "Retorno_Historico_Anual": (1 + ret_hist_carteira_mensal) ** 12 - 1,
        "Retorno_CAPM_Mensal": ret_capm_carteira_mensal,
        "Retorno_CAPM_Anual": (1 + ret_capm_carteira_mensal) ** 12 - 1,
    }])

# ============================================================
# 11) FRONTEIRA EFICIENTE
# ============================================================
with st.spinner("Simulando fronteira eficiente (800 carteiras)..."):
    mu = retornos_mensais[ativos].mean().values
    cov = retornos_mensais[ativos].cov().values

    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
        st.error("Ainda existem valores ausentes nos dados usados para a fronteira eficiente.")
        st.stop()

    n_carteiras = 800
    rng = np.random.default_rng(42)
    pesos = rng.dirichlet(np.ones(len(ativos)), size=n_carteiras)

    retornos = pesos @ mu
    riscos = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))

    df_fronteira = pd.DataFrame(pesos, columns=[f"Peso_{a}" for a in ativos])
    df_fronteira["Retorno_Esperado_Mensal"] = retornos
    df_fronteira["Risco_Mensal"] = riscos

    df_fronteira = df_fronteira.replace([np.inf, -np.inf], np.nan).dropna(subset=["Risco_Mensal", "Retorno_Esperado_Mensal"])
    if df_fronteira.empty:
        st.error("Não foi possível montar a fronteira eficiente com os dados atuais.")
        st.stop()

    idx_gmv = df_fronteira["Risco_Mensal"].idxmin()
    carteira_gmv = df_fronteira.loc[idx_gmv].copy()

    df_fronteira = df_fronteira.sort_values(["Risco_Mensal", "Retorno_Esperado_Mensal"]).reset_index(drop=True)
    df_fronteira["Eficiente"] = False
    melhor_retorno = -np.inf

    for i, row in df_fronteira.iterrows():
        risco_val = row["Risco_Mensal"]
        retorno_val = row["Retorno_Esperado_Mensal"]
        if pd.notna(risco_val) and pd.notna(retorno_val) and retorno_val > melhor_retorno:
            df_fronteira.at[i, "Eficiente"] = True
            melhor_retorno = retorno_val

    df_eficiente = df_fronteira[df_fronteira["Eficiente"]].copy().sort_values(
        ["Risco_Mensal", "Retorno_Esperado_Mensal"]
    ).reset_index(drop=True)
    df_ineficiente = df_fronteira[~df_fronteira["Eficiente"]].copy()
    df_eficiente_ordenado = df_eficiente.sort_values("Risco_Mensal").reset_index(drop=True)

# ============================================================
# 12) RESUMOS INTERPRETATIVOS
# ============================================================
resumo_economico = pd.DataFrame({
    "Tópico": [
        "Beta",
        "Risco total",
        "CAPM",
        "Fronteira eficiente",
        "Carteira escolhida",
    ],
    "Interpretação": [
        "Beta mede a sensibilidade do ativo em relação ao mercado. Beta acima de 1 indica maior sensibilidade às variações do mercado; abaixo de 1 indica menor sensibilidade.",
        "O risco total é a volatilidade dos retornos. Ele mostra o quanto o retorno oscila ao redor da média.",
        "O CAPM estima o retorno exigido pelo risco sistemático. Se o retorno histórico for maior que o CAPM, o ativo entregou desempenho acima do esperado pelo modelo.",
        "A fronteira eficiente reúne as combinações que oferecem maior retorno para cada nível de risco. Pontos abaixo dela são dominados e, portanto, ineficientes.",
        f"A carteira escolhida ({carteira_escolhida_nome}) representa a combinação destacada entre os {len(ativos)} ativos analisados. Ela deve ser interpretada em conjunto com a fronteira eficiente e com o perfil de risco desejado.",
    ]
})

# ============================================================
# 13) INTERFACE
# ============================================================
st.success(f"Carteira carregada com {len(ativos)} ativos selecionados.")
st.write("**Pesos confirmados:**", {k: f"{v:.2%}" for k, v in user_weights.items()})

tab_resumo, tab_tabelas, tab_graficos, tab_arquivos = st.tabs(
    ["Resumo", "Tabelas", "Gráficos", "Arquivos"]
)

with tab_resumo:
    st.subheader("Carteira escolhida")
    st.write(f"**{carteira_escolhida_nome}**")
    st.write(descricao_pesos)
    st.dataframe(df_carteira_escolhida, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Beta da carteira", f"{beta_carteira:.4f}")
    c2.metric("Alpha mensal", f"{alpha_carteira_mensal:.4f}")
    c3.metric("Retorno CAPM mensal", f"{ret_capm_carteira_mensal:.4f}")

    st.subheader("Resumo econômico")
    st.dataframe(resumo_economico, use_container_width=True)

with tab_tabelas:
    st.subheader("Preços mensais")
    st.dataframe(precos_mensais, use_container_width=True)

    st.subheader("Retornos mensais")
    st.dataframe(retornos_mensais, use_container_width=True)

    st.subheader("Análise individual dos ativos")
    st.dataframe(df_analise_ativos, use_container_width=True)

    st.subheader("Covariância")
    st.dataframe(matriz_cov, use_container_width=True)

    st.subheader("Correlação")
    st.dataframe(matriz_corr, use_container_width=True)

    st.subheader("Carteiras")
    st.dataframe(df_carteiras, use_container_width=True)

    st.subheader("Fronteira eficiente")
    st.dataframe(df_fronteira, use_container_width=True)

    st.subheader("Beta dos ativos")
    st.dataframe(df_beta_ativos, use_container_width=True)

    st.subheader("Beta da carteira")
    st.dataframe(df_beta_carteira_resumo, use_container_width=True)

with tab_graficos:
    with st.spinner("Gerando gráfico 3D da fronteira..."):
        st.subheader("Fronteira eficiente 3D")
        ativo_destaque = ativos[0]
        col_z = f"Peso_{ativo_destaque}"

        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=df_ineficiente["Retorno_Esperado_Mensal"],
            y=df_ineficiente["Risco_Mensal"],
            z=df_ineficiente[col_z],
            mode="markers",
            marker=dict(size=3, opacity=0.20),
            name="Ineficientes",
        ))
        fig3d.add_trace(go.Scatter3d(
            x=df_eficiente["Retorno_Esperado_Mensal"],
            y=df_eficiente["Risco_Mensal"],
            z=df_eficiente[col_z],
            mode="markers",
            marker=dict(size=4, opacity=0.75),
            name="Eficientes",
        ))
        fig3d.add_trace(go.Scatter3d(
            x=df_eficiente["Retorno_Esperado_Mensal"],
            y=df_eficiente["Risco_Mensal"],
            z=df_eficiente[col_z],
            mode="lines",
            line=dict(width=5),
            name="Fronteira eficiente",
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[carteira_gmv["Retorno_Esperado_Mensal"]],
            y=[carteira_gmv["Risco_Mensal"]],
            z=[carteira_gmv[col_z]],
            mode="markers",
            marker=dict(size=11, symbol="x"),
            name="Mínima variância",
        ))
        fig3d.update_layout(
            title=f"Fronteira eficiente 3D com {len(ativos)} ativos",
            scene=dict(
                xaxis_title="Retorno esperado mensal",
                yaxis_title="Risco mensal",
                zaxis_title=f"Peso em {ativo_destaque}",
            ),
            height=700,
            legend=dict(x=1.02, y=1),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with st.spinner("Gerando gráfico 2D da fronteira..."):
        st.subheader("Fronteira eficiente (risco vs retorno)")
        fig = plt.figure(figsize=(9, 6))
        plt.scatter(df_ineficiente["Risco_Mensal"], df_ineficiente["Retorno_Esperado_Mensal"], alpha=0.35, label="Ineficientes")
        plt.scatter(df_eficiente["Risco_Mensal"], df_eficiente["Retorno_Esperado_Mensal"], alpha=0.85, label="Eficientes")
        plt.scatter(carteira_gmv["Risco_Mensal"], carteira_gmv["Retorno_Esperado_Mensal"], s=120, marker="x", label="Mínima variância")
        plt.scatter(
            df_carteira_escolhida["Risco_Mensal"].iloc[0],
            df_carteira_escolhida["Retorno_Historico_Mensal"].iloc[0],
            s=180,
            marker="D",
            label=f"Carteira escolhida ({carteira_escolhida_nome})",
        )
        plt.xlabel("Risco (desvio-padrão mensal)")
        plt.ylabel("Retorno esperado mensal")
        plt.title("Fronteira eficiente (risco vs retorno)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        display_matplotlib(fig, "fronteira_eficiente_2D.png")

    with st.spinner("Gerando gráfico com perfis de carteiras..."):
        st.subheader("Fronteira eficiente com perfis de carteiras")
        fig = plt.figure(figsize=(12, 6))
        plt.scatter(df_ineficiente["Risco_Mensal"], df_ineficiente["Retorno_Esperado_Mensal"], alpha=0.15, s=5, label="Ineficientes")
        plt.scatter(df_eficiente["Risco_Mensal"], df_eficiente["Retorno_Esperado_Mensal"], alpha=0.5, s=12, label="Eficientes")
        plt.plot(df_eficiente_ordenado["Risco_Mensal"], df_eficiente_ordenado["Retorno_Esperado_Mensal"], linewidth=4, linestyle="-", label="Fronteira eficiente")
        plt.scatter(carteira_gmv["Risco_Mensal"], carteira_gmv["Retorno_Esperado_Mensal"], s=180, marker="x", linewidth=2, label="Mínima variância")

        palette = ["#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#16a085", "#8e44ad"]
        for i, (_, row) in enumerate(df_carteiras.iterrows()):
            cor = palette[i % len(palette)]
            pesos_txt = ", ".join([f"{a}: {row.get(f'Peso_{a}', 0):.0%}" for a in ativos])
            plt.scatter(
                row["Risco_Mensal"],
                row["Retorno_Esperado_Mensal"],
                s=150,
                color=cor,
                edgecolors="white",
                linewidth=1.5,
                label=f'{row["Carteira"]} ({pesos_txt})',
                zorder=5,
            )

        plt.scatter(
            df_carteira_escolhida["Risco_Mensal"].iloc[0],
            df_carteira_escolhida["Retorno_Historico_Mensal"].iloc[0],
            s=300,
            color="gold",
            marker="D",
            edgecolors="darkorange",
            linewidth=2.5,
            label=f"⭐ Carteira Escolhida ({carteira_escolhida_nome})",
            zorder=10,
        )

        plt.xlabel("Risco (desvio-padrão mensal)")
        plt.ylabel("Retorno esperado mensal")
        plt.title("Fronteira eficiente com perfis de carteiras – legenda externa")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        display_matplotlib(fig, "fronteira_perfis_legenda_externa.png")

    with st.spinner("Gerando regressão linear (Beta)..."):
        st.subheader("Regressão linear da carteira contra o mercado (beta)")
        df_beta_carteira = pd.concat([retorno_carteira_escolhida, retorno_mercado], axis=1).dropna()
        df_beta_carteira.columns = ["carteira", "mercado"]

        x = df_beta_carteira["mercado"].values
        y = df_beta_carteira["carteira"].values
        beta_regressao, intercepto = np.polyfit(x, y, 1)
        reta = np.poly1d([beta_regressao, intercepto])
        x_linha = np.linspace(x.min(), x.max(), 200)
        y_linha = reta(x_linha)

        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.7, label="Observações")
        plt.plot(x_linha, y_linha, linewidth=2, label=f"Regressão linear (beta = {beta_regressao:.2f})")
        plt.title("Regressão linear da carteira contra o mercado (beta)")
        plt.xlabel("Retorno do mercado")
        plt.ylabel("Retorno da carteira")
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        display_matplotlib(fig, "regressao_carteira_contra_mercado.png")

    with st.spinner("Gerando gráfico de Beta dos ativos..."):
        st.subheader("Beta dos ativos e da carteira")
        betas = df_beta_ativos["Beta"].values
        nomes = df_beta_ativos["Ativo"].tolist() + ["Carteira"]
        valores_beta = list(betas) + [beta_carteira]

        fig = plt.figure(figsize=(8, 5))
        barras = plt.barh(nomes, valores_beta)
        plt.title("Beta dos ativos e da carteira")
        plt.xlabel("Beta")
        plt.ylabel("Ativo / Carteira")
        plt.grid(True, axis="x", alpha=0.3)
        for barra in barras:
            largura = barra.get_width()
            plt.text(largura, barra.get_y() + barra.get_height() / 2, f"{largura:.2f}", va="center", ha="left")
        plt.tight_layout()
        display_matplotlib(fig, "beta_ativos_e_carteira.png")

    with st.spinner("Gerando gráfico Histórico vs CAPM..."):
        st.subheader("Retorno Histórico x CAPM")
        x = np.arange(len(df_beta_ativos))
        largura = 0.35
        fig = plt.figure(figsize=(9, 5))
        plt.bar(x - largura / 2, df_beta_ativos["Retorno_Historico_Anual"], width=largura, label="Histórico")
        plt.bar(x + largura / 2, df_beta_ativos["Retorno_CAPM_Anual"], width=largura, label="CAPM")
        plt.xticks(x, df_beta_ativos["Ativo"])
        plt.title("Retorno Histórico x CAPM")
        plt.ylabel("Retorno anual")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        display_matplotlib(fig, "historico_vs_capm.png")

    with st.spinner("Gerando gráfico de retorno acumulado..."):
        st.subheader("Retorno acumulado dos ativos")
        acumulado = (1 + retornos_mensais[ativos]).cumprod()
        fig = plt.figure(figsize=(10, 6))
        for ativo in ativos:
            plt.plot(acumulado.index, acumulado[ativo], label=ativo)
        plt.title("Retorno acumulado dos ativos")
        plt.xlabel("Data")
        plt.ylabel("Valor acumulado")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        display_matplotlib(fig, "retorno_acumulado_ativos.png")

    with st.spinner("Gerando gráfico de risco dos ativos..."):
        st.subheader("Risco dos ativos versus carteira")
        riscos_ativos = [retornos_mensais[ativo].std() for ativo in ativos]
        nomes_risco = ativos + ["Carteira"]
        valores_risco = riscos_ativos + [df_carteira_escolhida["Risco_Mensal"].iloc[0]]

        fig = plt.figure(figsize=(8, 5))
        barras = plt.bar(nomes_risco, valores_risco)
        plt.title("Risco dos ativos versus carteira")
        plt.ylabel("Desvio-padrão mensal")
        plt.grid(True, axis="y", alpha=0.3)
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, altura, f"{altura:.4f}", ha="center", va="bottom")
        plt.tight_layout()
        display_matplotlib(fig, "risco_ativos_vs_carteira.png")

with tab_arquivos:
    st.subheader("Arquivos gerados")
    with st.spinner("Gerando arquivo Excel..."):
        excel_path = BASE_DIR / "dados_mensais_CAPM_Varios_ativos.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            precos_mensais.to_excel(writer, sheet_name="Precos_Mensais")
            retornos_mensais.to_excel(writer, sheet_name="Retornos_Mensais")
            df_analise_ativos.to_excel(writer, sheet_name="Analise_Ativos", index=False)
            matriz_cov.to_excel(writer, sheet_name="Covariancia")
            matriz_corr.to_excel(writer, sheet_name="Correlacao")
            df_carteiras.to_excel(writer, sheet_name="Carteiras", index=False)
            df_fronteira.to_excel(writer, sheet_name="Fronteira_Eficiente", index=False)
            df_eficiente.to_excel(writer, sheet_name="Fronteira_Pontos_Eficientes", index=False)
            df_ineficiente.to_excel(writer, sheet_name="Fronteira_Pontos_Ineficientes", index=False)
            df_beta_ativos.to_excel(writer, sheet_name="Beta_Ativos", index=False)
            df_beta_carteira_resumo.to_excel(writer, sheet_name="Beta_Carteira", index=False)
            df_carteira_escolhida.to_excel(writer, sheet_name="Carteira_Escolhida", index=False)
            resumo_economico.to_excel(writer, sheet_name="Resumo_Economico", index=False)

    st.success(f"Arquivo salvo em: {excel_path}")
    st.download_button(
        "Baixar Excel",
        data=excel_path.read_bytes(),
        file_name=excel_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.subheader("Saída principal")
st.write(f"Taxa livre de risco anual: **{taxa_livre_risco_anual:.2%}**")
st.write(f"Taxa livre de risco mensal: **{taxa_livre_risco_mensal:.4%}**")
st.write(f"Beta da carteira principal: **{beta_carteira:.4f}**")
st.write(f"Alpha mensal da carteira principal: **{alpha_carteira_mensal:.4f}**")
st.write(f"Retorno CAPM mensal da carteira principal: **{ret_capm_carteira_mensal:.4f}**")