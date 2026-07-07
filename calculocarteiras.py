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
import requests
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


def format_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise RuntimeError("A coluna Adj Close/Close não foi encontrada no retorno do yfinance.")
    else:
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
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


@st.cache_data(ttl=24 * 60 * 60)
def baixar_selic_mensal_bcb(data_inicio, data_fim) -> pd.Series:
    """
    Baixa a Selic diária (SGS 11) do Banco Central e converte para série mensal
    por capitalização composta.
    """
    data_inicio = pd.Timestamp(data_inicio)
    data_fim = pd.Timestamp(data_fim)

    partes = []
    inicio_atual = data_inicio

    while inicio_atual <= data_fim:
        fim_bloco = min(
            inicio_atual + pd.DateOffset(years=10) - pd.Timedelta(days=1),
            data_fim,
        )

        url = (
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
            f"?formato=json&dataInicial={inicio_atual.strftime('%d/%m/%Y')}"
            f"&dataFinal={fim_bloco.strftime('%d/%m/%Y')}"
        )

        resposta = requests.get(url, timeout=30)
        resposta.raise_for_status()

        selic = pd.DataFrame(resposta.json())
        if not selic.empty:
            selic["data"] = pd.to_datetime(selic["data"], dayfirst=True)
            selic["valor"] = pd.to_numeric(selic["valor"], errors="coerce") / 100.0
            partes.append(selic[["data", "valor"]])

        inicio_atual = fim_bloco + pd.Timedelta(days=1)

    if not partes:
        return pd.Series(dtype=float, name="Taxa_Livre_Risco_Mensal")

    selic = pd.concat(partes, ignore_index=True)
    selic = selic.drop_duplicates(subset="data").sort_values("data")

    selic_diaria = selic.set_index("data")["valor"]

    # Converte a Selic diária em taxa mensal composta
    selic_mensal = (1 + selic_diaria).resample("ME").prod() - 1
    selic_mensal.name = "Taxa_Livre_Risco_Mensal"
    return selic_mensal


def obter_taxa_livre_risco_mensal(indice_mensal: pd.DatetimeIndex) -> pd.Series:
    """Obtém a taxa livre de risco mensal alinhada ao índice mensal dos retornos."""
    indice_mensal = pd.DatetimeIndex(indice_mensal).sort_values()

    try:
        serie = baixar_selic_mensal_bcb(indice_mensal.min(), indice_mensal.max())
        if serie.empty:
            raise RuntimeError("A API não retornou dados de Selic no período.")
        return serie.reindex(indice_mensal).ffill().bfill()
    except Exception as e:
        st.warning(
            f"Não foi possível carregar a Selic automática da API do Banco Central ({e}). "
            "Será usada a taxa fixa de contingência."
        )
        taxa_livre_risco_anual = 0.1450
        taxa_livre_risco_mensal = (1 + taxa_livre_risco_anual) ** (1 / 12) - 1
        return pd.Series(
            taxa_livre_risco_mensal,
            index=indice_mensal,
            name="Taxa_Livre_Risco_Mensal",
        )


def _pct_signature(weight_map: dict[str, float], ativos: list[str], casas: int = 2) -> tuple[float, ...]:
    """
    Assinatura da carteira em porcentagem com 2 casas decimais.
    Usa ordenação dos pesos para ignorar pequenas diferenças de posição/float.
    """
    pesos = [round(float(weight_map.get(a, 0.0)) * 100, casas) for a in ativos]
    return tuple(sorted(pesos))


def _equal_pct_weights(ativos: list[str]) -> dict[str, float]:
    """
    Carteira igualitária com 2 casas decimais e ajuste do último ativo
    para fechar exatamente 100%.
    Ex.: 3 ativos -> 33.33, 33.33, 33.34
    """
    n = len(ativos)
    if n == 0:
        return {}

    base = round(100 / n, 2)
    pesos = {a: base for a in ativos}
    ajuste = round(100 - sum(pesos.values()), 2)
    pesos[ativos[-1]] = round(pesos[ativos[-1]] + ajuste, 2)
    return pesos
def make_portfolios(
    ativos: list[str],
    user_weights: dict[str, float],
    extra_portfolios: list[dict] | None = None
) -> dict[str, dict[str, float]]:
    if not ativos:
        raise ValueError("É preciso informar ao menos um ativo.")

    carteiras_pct: dict[str, dict[str, float]] = {}

    # ============================================================
    # CARTEIRA PRINCIPAL: AGORA USA OS PESOS DO USUÁRIO
    # ============================================================
    # Verifica se o usuário forneceu pesos válidos
    if user_weights and any(user_weights.values()):
        total = sum(user_weights.values())
        # Converte de fração (ex: 0.5) para porcentagem (ex: 50.0)
        pesos_pct = {a: (user_weights.get(a, 0.0) / total) * 100 for a in ativos}
        # Ajusta o último ativo para garantir que a soma seja exatamente 100%
        # (evita 99.9999 ou 100.0001)
        soma = sum(pesos_pct.values())
        if abs(soma - 100.0) > 0.0001:
            pesos_pct[ativos[-1]] += (100.0 - soma)
        # Arredonda para 2 casas
        carteiras_pct["Carteira_1"] = {a: round(p, 2) for a, p in pesos_pct.items()}
    else:
        # Fallback: se o usuário não forneceu pesos, usa a carteira igualitária
        carteiras_pct["Carteira_1"] = _equal_pct_weights(ativos)

    # ============================================================
    # CARTEIRAS EXTRAS (mantido igual ao original)
    # ============================================================
    extra_portfolios = extra_portfolios or []
    for spec in extra_portfolios:
        nome = str(spec.get("nome", "")).strip()
        if not nome:
            continue

        pesos_raw = spec.get("pesos", {}) or {}
        pesos_pct = {a: round(float(pesos_raw.get(a, 0.0)) * 100, 2) for a in ativos}

        if sum(pesos_pct.values()) <= 0:
            continue

        carteiras_pct[nome] = pesos_pct

    # Converte de % para fração para os cálculos
    return {
        nome: {a: pct / 100.0 for a, pct in pesos_pct.items()}
        for nome, pesos_pct in carteiras_pct.items()
    }
def portfolio_metrics(retornos_mensais: pd.DataFrame, ativos: list[str], pesos: np.ndarray, matriz_cov: pd.DataFrame):
    retorno_carteira = retornos_mensais[ativos].mul(pesos, axis=1).sum(axis=1)
    retorno_esperado_mensal = float(retorno_carteira.mean())
    risco_mensal = float(retorno_carteira.std())
    risco_mensal_cov = float(np.sqrt(np.dot(pesos.T, np.dot(matriz_cov.values, pesos))))
    return retorno_carteira, retorno_esperado_mensal, risco_mensal, risco_mensal_cov


def project_portfolio_values(initial_value: float, monthly_return: float, steps: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=float)
    return float(initial_value) * np.power(1.0 + float(monthly_return), steps)


def estimate_monthly_return_bounds(series: pd.Series, z_score: float = 1.96):
    """
    Estima retorno mensal base, conservador e otimista usando log-retornos.
    Isso é mais estável que usar a média aritmética simples.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > -0.9999]

    if s.empty:
        return None

    log_r = np.log1p(s)
    mu_log = float(log_r.mean())
    base = float(np.expm1(mu_log))

    if len(log_r) > 1:
        se_log = float(log_r.std(ddof=1)) / np.sqrt(len(log_r))
        low = float(np.expm1(mu_log - z_score * se_log))
        high = float(np.expm1(mu_log + z_score * se_log))
    else:
        low = base
        high = base

    return {
        "base": base,
        "conservador": low,
        "otimista": high,
        "amostra": int(len(s)),
    }


def simulate_portfolio_paths(
    return_series: pd.Series,
    initial_value: float,
    horizon: int,
    n_simulations: int = 5000,
    method: str = "Bootstrap histórico",
    seed: int = 42,
) -> np.ndarray:
    """Gera trajetórias simuladas para o valor da carteira."""
    s = pd.to_numeric(return_series, errors="coerce").dropna()
    s = s[s > -0.9999]

    if len(s) < 2:
        raise ValueError("Histórico insuficiente para simulação.")
    if horizon < 1:
        raise ValueError("O horizonte precisa ser maior que zero.")
    if n_simulations < 1:
        raise ValueError("A quantidade de simulações precisa ser maior que zero.")

    rng = np.random.default_rng(seed)

    if method == "Bootstrap histórico":
        sampled_returns = rng.choice(s.to_numpy(), size=(n_simulations, horizon), replace=True)
    else:
        mu = float(s.mean())
        sigma = float(s.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(abs(mu) * 0.10, 0.01)
        sampled_returns = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon))
        sampled_returns = np.clip(sampled_returns, -0.9999, None)

    paths = float(initial_value) * np.cumprod(1.0 + sampled_returns, axis=1)
    paths = np.concatenate([np.full((n_simulations, 1), float(initial_value)), paths], axis=1)
    return paths


def summarize_simulation_paths(
    paths: np.ndarray,
    initial_value: float,
    benchmark_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resume as trajetórias simuladas em métricas de risco."""
    finals = np.asarray(paths[:, -1], dtype=float)
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_max - 1.0
    max_drawdown = drawdowns.min(axis=1)

    summary = {
        "chance_perda": float(np.mean(finals < initial_value)),
        "media_final": float(np.mean(finals)),
        "mediana_final": float(np.percentile(finals, 50)),
        "p05_final": float(np.percentile(finals, 5)),
        "p10_final": float(np.percentile(finals, 10)),
        "p90_final": float(np.percentile(finals, 90)),
        "p95_final": float(np.percentile(finals, 95)),
        "max_drawdown_medio": float(np.mean(max_drawdown)),
        "max_drawdown_p05": float(np.percentile(max_drawdown, 5)),
        "max_drawdown_p50": float(np.percentile(max_drawdown, 50)),
    }

    if benchmark_values:
        for nome, valor in benchmark_values.items():
            if valor is None or not np.isfinite(valor):
                summary[f"chance_superar_{nome}"] = np.nan
            else:
                summary[f"chance_superar_{nome}"] = float(np.mean(finals > float(valor)))

    return summary

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
    valid_base_keys = {f"peso_{asset}" for asset in selected_assets}

    for key in list(st.session_state.keys()):
        if key.startswith("peso_extra_"):
            continue
        if key.startswith("peso_") and key not in valid_base_keys:
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

if "last_start_date" not in st.session_state:
    st.session_state.last_start_date = start_date
if "last_end_date" not in st.session_state:
    st.session_state.last_end_date = end_date

if (
    start_date != st.session_state.last_start_date
    or end_date != st.session_state.last_end_date
):
    st.session_state.last_start_date = start_date
    st.session_state.last_end_date = end_date
    st.session_state.pesos_confirmados = False
    st.session_state.pop("user_weights_confirmed", None)
    st.session_state.pop("extra_portfolios_confirmed", None)

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
# =========================================================
# 4) PESOS DINÂMICOS (com carteira base + extras abaixo)
# =========================================================
cleanup_weight_keys(selected_display_names)
init_weight_inputs(selected_display_names)

if "portfolio_specs" not in st.session_state:
    st.session_state.portfolio_specs = []

if "portfolio_counter" not in st.session_state:
    st.session_state.portfolio_counter = 0

if "pesos_confirmados" not in st.session_state:
    st.session_state.pesos_confirmados = False

if "user_weights_confirmed" not in st.session_state:
    st.session_state.user_weights_confirmed = {}

if "extra_portfolios_confirmed" not in st.session_state:
    st.session_state.extra_portfolios_confirmed = []

if "valor_investido" not in st.session_state:
    st.session_state.valor_investido = 1000.0

if "n_pontos_fronteira" not in st.session_state:
    st.session_state.n_pontos_fronteira = 800


def _default_weight_map() -> dict[str, float]:
    n = len(selected_display_names)
    if n == 0:
        return {}
    base = round(100 / n, 2)
    weights = {asset: base for asset in selected_display_names}
    ajuste = 100 - sum(weights.values())
    weights[selected_display_names[-1]] += ajuste
    return weights


def reset_confirmation():
    st.session_state.pesos_confirmados = False


def add_extra_portfolio():
    st.session_state.portfolio_counter += 1
    pid = st.session_state.portfolio_counter

    st.session_state.portfolio_specs.append({
        "id": pid,
        "nome": f"Carteira_extra_{pid}",
    })

    default_map = _default_weight_map()
    for asset in selected_display_names:
        key = f"peso_extra_{pid}_{asset}"
        if key not in st.session_state:
            st.session_state[key] = default_map.get(asset, 0.0)

    st.session_state[f"nome_carteira_{pid}"] = f"Carteira_extra_{pid}"
    reset_confirmation()
    st.rerun()


def remove_extra_portfolio(portfolio_id: int):
    st.session_state.portfolio_specs = [
        p for p in st.session_state.portfolio_specs if p["id"] != portfolio_id
    ]

    for asset in selected_display_names:
        st.session_state.pop(f"peso_extra_{portfolio_id}_{asset}", None)

    st.session_state.pop(f"nome_carteira_{portfolio_id}", None)
    reset_confirmation()
    st.rerun()


def _render_asset_card(asset: str, cols, idx: int, key_prefix: str = "peso"):
    label = st.session_state.selected_labels.get(asset, asset)
    parts = [p.strip() for p in str(label).split(" | ")]

    selected_name = parts[1] if len(parts) > 1 and parts[1] else asset
    selected_exchange = parts[2] if len(parts) > 2 and parts[2] else "—"
    selected_typedisp = parts[3] if len(parts) > 3 and parts[3] else "—"

    with cols[idx % len(cols)]:
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

        default_value = _default_weight_map().get(
            asset, round(100 / len(selected_display_names), 2)
        )
        state_key = key_prefix

        st.session_state.setdefault(state_key, default_value)

        st.number_input(
            f"{state_key} (%)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key=state_key,
            label_visibility="collapsed",
            on_change=reset_confirmation,
        )



st.markdown("### Planejamento do investimento")
col_inv, col_pts = st.columns(2)
with col_inv:
    st.number_input(
        "Quanto vai investir (R$)",
        min_value=0.0,
        step=100.0,
        format="%.2f",
        key="valor_investido",
        help="Valor total que será distribuído entre os ativos da carteira confirmada.",
    )
with col_pts:
    st.number_input(
        "Quantidade de pontos da Fronteira Eficiente",
        min_value=10,
        max_value=100000,
        step=100,
        value=int(st.session_state.get("n_pontos_fronteira", 800)),
        key="n_pontos_fronteira",
        help="Aumente este número para gerar mais carteiras simuladas na fronteira eficiente.",
    )

st.markdown("### Pesos da carteira escolhida")
st.caption("Edite livremente. A validação só acontece ao clicar em Confirmar pesos.")

with st.container(border=True):
    cards_per_row = 2 if len(selected_display_names) <= 4 else 3
    cols = st.columns(cards_per_row, gap="small")

    for i, asset in enumerate(selected_display_names):
        _render_asset_card(asset, cols, i, key_prefix=f"peso_{asset}")

    user_weight_values = {
        asset: float(st.session_state[f"peso_{asset}"])
        for asset in selected_display_names
    }
    total = sum(user_weight_values.values())
    is_valid = abs(total - 100.0) < 0.01


    st.markdown(
        f"""
        <div class="portfolio-summary-bar" style="align-items: flex-start;">
            <div class="portfolio-summary-left" style="align-items: flex-start;">
                <div class="portfolio-summary-icon">🧩Σ</div>
                <div style="display:flex; flex-direction:column; line-height:1.15;">
                    <div class="portfolio-summary-text" style="white-space: normal; line-height: 1.15;">
                        Carteira principal
                    </div>
                    <div class="portfolio-summary-sub" style="white-space: normal; margin-top: 4px;">
                        Carteira base selecionada
                    </div>
                </div>
            </div>
            <div class="portfolio-summary-value">{total:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pré-validação das carteiras extras
    all_extra_valid = True
    for portfolio in st.session_state.portfolio_specs:
        pid = portfolio["id"]
        soma_extra_preview = sum(
            float(st.session_state.get(f"peso_extra_{pid}_{asset}", 0.0))
            for asset in selected_display_names
        )
        extra_valid_preview = abs(soma_extra_preview - 100.0) < 0.01
        all_extra_valid = all_extra_valid and extra_valid_preview

    all_valid = is_valid and all_extra_valid and len(selected_display_names) > 0

    confirm = st.button(
        "Confirmar pesos",
        key="confirmar_pesos",
        disabled=not all_valid,
        use_container_width=True,
    )
    st.button(
        "➕ Adicionar carteira",
        key="add_extra_portfolio_btn",
        on_click=add_extra_portfolio,
        use_container_width=True,
    )
    if not is_valid and total > 0:
        st.warning("A soma dos pesos da carteira base deve ser exatamente 100% para confirmar.")
    elif total == 0:
        st.error("A soma dos pesos não pode ser zero.")

    st.markdown("---")

    extra_portfolios_preview = []

    for portfolio in st.session_state.portfolio_specs:
        pid = portfolio["id"]

        nome_key = f"nome_carteira_{pid}"
        st.session_state.setdefault(nome_key, portfolio.get("nome", f"Carteira_extra_{pid}"))

        st.text_input(
            f"Nome da carteira {pid}",
            key=nome_key,
            on_change=reset_confirmation,
        )

        cols = st.columns(cards_per_row, gap="small")

        for i, asset in enumerate(selected_display_names):
            label = st.session_state.selected_labels.get(asset, asset)
            parts = [p.strip() for p in str(label).split(" | ")]

            selected_name = parts[1] if len(parts) > 1 and parts[1] else asset
            selected_exchange = parts[2] if len(parts) > 2 and parts[2] else "—"
            selected_typedisp = parts[3] if len(parts) > 3 and parts[3] else "—"

            peso_key = f"peso_extra_{pid}_{asset}"
            default_value = _default_weight_map().get(
                asset, round(100 / len(selected_display_names), 2)
            )
            st.session_state.setdefault(peso_key, default_value)

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
                    f"{pid}_{asset} (%)",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=peso_key,
                    label_visibility="collapsed",
                    on_change=reset_confirmation,
                )

        soma_extra = sum(
            float(st.session_state[f"peso_extra_{pid}_{asset}"])
            for asset in selected_display_names
        )
        extra_valid = abs(soma_extra - 100.0) < 0.01
        all_extra_valid = all_extra_valid and extra_valid
   
        st.markdown(
            f"""
            <div class="portfolio-summary-bar">
                <div class="portfolio-summary-left">
                    <div class="portfolio-summary-icon">📁</div>
                    <div>
                        <div class="portfolio-summary-text">{html.escape(st.session_state[nome_key])}</div>
                        <div class="portfolio-summary-sub">Clique abaixo para ver os pesos</div>
                    </div>
                </div>
                <div class="portfolio-summary-value">{soma_extra:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Abrir carteira extra", expanded=False):
            st.caption(f"Resumo de {st.session_state[nome_key]}")

            st.markdown('<div class="extra-list-box">', unsafe_allow_html=True)
            for asset in selected_display_names:
                peso_pct = float(st.session_state[f"peso_extra_{pid}_{asset}"])
                label = st.session_state.selected_labels.get(asset, asset)
                parts = [p.strip() for p in str(label).split(" | ")]
                selected_name = parts[1] if len(parts) > 1 and parts[1] else asset
                st.markdown(
                    f"""
                    <div class="extra-card">
                        <div class="extra-card-head">
                            <div>
                                <div class="extra-card-title">{html.escape(selected_name)}</div>
                                <div class="extra-card-meta">{html.escape(asset)}</div>
                            </div>
                            <div class="portfolio-summary-value">{peso_pct:.2f}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

         
        if extra_valid:
            st.success("Esta carteira está pronta.")
        elif soma_extra > 0:
            st.warning("Esta carteira ainda não soma 100%.")

        extra_portfolios_preview.append({
            "nome": st.session_state[nome_key].strip() or f"Carteira_extra_{pid}",
            "soma_pct": soma_extra / 100.0,
            "pesos": {
                a: float(st.session_state[f"peso_extra_{pid}_{a}"]) / 100.0
                for a in selected_display_names
            },
        })
        st.button(
                        "🗑️ Remover carteira extra",
                        key=f"remover_carteira_extra_{pid}",
                        use_container_width=True,
                        on_click=remove_extra_portfolio,
                        args=(pid,),
                        type="secondary",
                    )
    if confirm and all_valid:
        st.session_state.user_weights_confirmed = {
            k: v / 100.0 for k, v in user_weight_values.items()
        }
        st.session_state.extra_portfolios_confirmed = [
            {
                "nome": p["nome"],
                "soma_pct": p["soma_pct"],
                "pesos": dict(p["pesos"]),
            }
            for p in extra_portfolios_preview
        ]
        st.session_state.pesos_confirmados = True
        st.success("Todos os pesos foram confirmados.")

    elif confirm and not all_valid:
        st.session_state.pesos_confirmados = False
        st.warning("Para confirmar, todas as carteiras precisam somar exatamente 100%.")

    if not st.session_state.get("pesos_confirmados", False):
        st.stop()

user_weights = st.session_state.get("user_weights_confirmed", {})
extra_portfolios = st.session_state.get("extra_portfolios_confirmed", [])
valor_investido = float(st.session_state.get("valor_investido", 0.0))

st.markdown("### Resumo final")

if valor_investido > 0 and user_weights:
    st.markdown("**Alocação estimada em R$ da carteira base:**")
    st.write({k: format_brl(valor_investido * float(v)) for k, v in user_weights.items()})

st.markdown(
    f"""
    <div class="portfolio-summary-bar">
        <div class="portfolio-summary-left">
            <div class="portfolio-summary-icon">✅</div>
            <div>
                <div class="portfolio-summary-text">Carteira principal confirmada</div>
                <div class="portfolio-summary-sub">{len(selected_display_names)} ativos selecionados</div>
            </div>
        </div>
        <div class="portfolio-summary-value">{sum(user_weights.values()):.2%}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(f"Carteiras extras confirmadas ({len(extra_portfolios)})", expanded=False):
    if not extra_portfolios:
        st.info("Nenhuma carteira extra confirmada.")
    else:
        for p in extra_portfolios:
            nome = p.get("nome", "Carteira extra")
            pesos = p.get("pesos", {})
            soma_pct = p.get("soma_pct", sum(pesos.values()))

            st.markdown(
                f"""
                <div class="portfolio-summary-bar">
                    <div class="portfolio-summary-left">
                        <div class="portfolio-summary-icon">📌</div>
                        <div>
                            <div class="portfolio-summary-text">{html.escape(nome)}</div>
                            <div class="portfolio-summary-sub">{len(pesos)} ativos • clique para ver os detalhes abaixo</div>
                        </div>
                    </div>
                    <div class="portfolio-summary-value">{soma_pct:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Ver pesos de {nome}", expanded=False):
                for ativo, peso in pesos.items():
                    st.write(f"**{ativo}**: {peso:.2%}")

st.write("**Ativos selecionados:**", ", ".join(selected_display_names))
st.write(
    "**Pesos confirmados da carteira base:**",
    {k: f"{v:.2%}" for k, v in user_weights.items()}
)
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
    carteiras = make_portfolios(
        ativos,
        user_weights,
        extra_portfolios
    )

    resultados_carteiras = []
    retornos_carteiras_series = {}

    for nome_carteira, pesos_dict in carteiras.items():
        pesos_array = np.array([pesos_dict.get(a, 0) for a in ativos], dtype=float)

        retorno_carteira = retornos_mensais[ativos].mul(pesos_array, axis=1).sum(axis=1)
        retornos_carteiras_series[nome_carteira] = retorno_carteira

        retorno_esperado_mensal = float(retorno_carteira.mean())
        risco_mensal = float(retorno_carteira.std())
        risco_mensal_cov = float(np.sqrt(np.dot(pesos_array.T, np.dot(matriz_cov.values, pesos_array))))

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

# ============================================================
# 10) BETA E CAPM
# ============================================================
with st.spinner("Calculando Beta e CAPM dos ativos e das carteiras..."):
    taxa_livre_risco_mensal = obter_taxa_livre_risco_mensal(retornos_mensais.index)
    taxa_livre_risco_mensal_media = float(taxa_livre_risco_mensal.mean())
    taxa_livre_risco_anual = float((1 + taxa_livre_risco_mensal_media) ** 12 - 1)
    retorno_mercado = retornos_mensais["Retorno_Mercado_IBOV"]

    # --------------------------------------------------------
    # Beta e CAPM dos ativos
    # --------------------------------------------------------
    resultados_beta_ativos = []
    for ativo in ativos:
        df = pd.concat([retornos_mensais[ativo], retorno_mercado, taxa_livre_risco_mensal], axis=1).dropna()
        df.columns = ["ativo", "mercado", "rf"]

        excesso_ativo = df["ativo"] - df["rf"]
        excesso_mercado = df["mercado"] - df["rf"]

        beta = excesso_ativo.cov(excesso_mercado) / excesso_mercado.var()

        ret_hist_mensal = float(df["ativo"].mean())
        ret_capm_mensal = float(df["rf"].mean() + beta * (df["mercado"].mean() - df["rf"].mean()))
        alpha_mensal = ret_hist_mensal - ret_capm_mensal

        resultados_beta_ativos.append({
            "Ativo": ativo,
            "Beta": beta,
            "Retorno_Historico_Mensal": ret_hist_mensal,
            "Retorno_Historico_Anual": (1 + ret_hist_mensal) ** 12 - 1,
            "Retorno_CAPM_Mensal": ret_capm_mensal,
            "Retorno_CAPM_Anual": (1 + ret_capm_mensal) ** 12 - 1,
            "Alpha_Mensal": alpha_mensal,
            "Alpha_Anual": (1 + alpha_mensal) ** 12 - 1,
            "Retorno_Mercado_Medio_Mensal": float(df["mercado"].mean()),
        })

    df_beta_ativos = pd.DataFrame(resultados_beta_ativos)

    # --------------------------------------------------------
    # Remove carteiras duplicadas antes de calcular Beta/CAPM
    # --------------------------------------------------------
    peso_cols = [f"Peso_{a}" for a in ativos]

    df_carteiras_beta_base = df_carteiras.copy()
    df_carteiras_beta_base["_assinatura_pesos"] = (
        df_carteiras_beta_base[peso_cols]
        .round(8)
        .astype(str)
        .agg("|".join, axis=1)
    )

    df_carteiras_beta_base = (
        df_carteiras_beta_base
        .drop_duplicates(subset=["_assinatura_pesos"], keep="first")
        .drop(columns=["_assinatura_pesos"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Beta e CAPM das carteiras únicas
    # --------------------------------------------------------
    resultados_beta_carteiras = []

    for nome_carteira in df_carteiras_beta_base["Carteira"].tolist():
        if nome_carteira not in retornos_carteiras_series:
            continue

        retorno_carteira = retornos_carteiras_series[nome_carteira]
        df_carteira = pd.concat([retorno_carteira, retorno_mercado, taxa_livre_risco_mensal], axis=1).dropna()
        df_carteira.columns = ["carteira", "mercado", "rf"]

        excesso_carteira = df_carteira["carteira"] - df_carteira["rf"]
        excesso_mercado_carteira = df_carteira["mercado"] - df_carteira["rf"]

        beta_carteira = excesso_carteira.cov(excesso_mercado_carteira) / excesso_mercado_carteira.var()
        ret_hist_carteira_mensal = float(df_carteira["carteira"].mean())
        ret_capm_carteira_mensal = float(df_carteira["rf"].mean() + beta_carteira * (df_carteira["mercado"].mean() - df_carteira["rf"].mean()))
        alpha_carteira_mensal = ret_hist_carteira_mensal - ret_capm_carteira_mensal

        resultados_beta_carteiras.append({
            "Carteira": nome_carteira,
            "Beta": beta_carteira,
            "Alpha_Mensal": alpha_carteira_mensal,
            "Alpha_Anual": (1 + alpha_carteira_mensal) ** 12 - 1,
            "Retorno_Historico_Mensal": ret_hist_carteira_mensal,
            "Retorno_Historico_Anual": (1 + ret_hist_carteira_mensal) ** 12 - 1,
            "Retorno_CAPM_Mensal": ret_capm_carteira_mensal,
            "Retorno_CAPM_Anual": (1 + ret_capm_carteira_mensal) ** 12 - 1,
            "Retorno_Mercado_Medio_Mensal": float(df_carteira["mercado"].mean()),
        })

    df_beta_carteiras = pd.DataFrame(resultados_beta_carteiras)

    # Resumo final sem duplicata
    df_portfolios_resumo = df_carteiras_beta_base.merge(
        df_beta_carteiras,
        on="Carteira",
        how="left"
    )

# ============================================================
# 11) FRONTEIRA EFICIENTE
# ============================================================
n_carteiras = int(st.session_state.get("n_pontos_fronteira", 800))
n_carteiras = max(10, n_carteiras)
with st.spinner(f"Simulando fronteira eficiente ({n_carteiras} carteiras)..."):
    mu = retornos_mensais[ativos].mean().values
    cov = retornos_mensais[ativos].cov().values

    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
        st.error("Ainda existem valores ausentes nos dados usados para a fronteira eficiente.")
        st.stop()

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
    ],
    "Interpretação": [
        "Beta mede a sensibilidade do ativo em relação ao mercado. Beta acima de 1 indica maior sensibilidade às variações do mercado; abaixo de 1 indica menor sensibilidade.",
        "O risco total é a volatilidade dos retornos. Ele mostra o quanto o retorno oscila ao redor da média.",
        "O CAPM estima o retorno exigido pelo risco sistemático. Se o retorno histórico for maior que o CAPM, o ativo entregou desempenho acima do esperado pelo modelo.",
        "A fronteira eficiente reúne as combinações que oferecem maior retorno para cada nível de risco. Pontos abaixo dela são dominados e, portanto, ineficientes.",
    ]
})

# ============================================================
# 13) INTERFACE
# ============================================================
st.success(f"Carteiras carregadas com {len(carteiras)} carteiras confirmadas e {len(ativos)} ativos selecionados.")
st.write("**Pesos confirmados da carteira base:**", {k: f"{v:.2%}" for k, v in user_weights.items()})

# Agora 5 abas: Resumo, Análise Detalhada, Projeção, Simulação de Risco, Arquivos
tab_resumo, tab_detalhes, tab_projecao, tab_risco_simulado, tab_arquivos = st.tabs(
    ["Resumo", "Análise Detalhada", "Projeção", "Simulação de Risco", "Arquivos"]
)

df_simulacao_risco = pd.DataFrame()
# ============================================================
# ABA RESUMO
# ============================================================
with tab_resumo:
    st.subheader("📊 Resumo das carteiras confirmadas")

    st.caption(
        "Painel consolidado das carteiras confirmadas, contendo indicadores de risco, retorno e desempenho esperado."
    )

    df_portfolios_resumo_exibicao = df_portfolios_resumo.copy()

    # Mantém a carteira principal no resumo e só cria um nome amigável para exibição
    if "Carteira" in df_portfolios_resumo_exibicao.columns:
        df_portfolios_resumo_exibicao["Carteira_Exibicao"] = df_portfolios_resumo_exibicao["Carteira"].replace({
            "Carteira_1": "Carteira principal"
        })
    else:
        df_portfolios_resumo_exibicao["Carteira_Exibicao"] = "Carteira principal"

    # =====================================================
    # Tabela consolidada
    # =====================================================

    st.subheader("📋 Resumo consolidado")

    display_cols = [
        "Carteira_Exibicao",
        "Retorno_Historico_Anual",
        "Retorno_Esperado_Anual",
        "Risco_Anual",
        "Beta",
        "Alpha_Mensal",
        "Retorno_CAPM_Anual",
    ]

    display_cols = [
        c for c in display_cols
        if c in df_portfolios_resumo_exibicao.columns
    ]

    st.dataframe(
        df_portfolios_resumo_exibicao[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    if not df_portfolios_resumo_exibicao.empty:

        # =====================================================
        # Indicadores gerais
        # =====================================================

        carteira_maior_retorno = df_portfolios_resumo_exibicao.loc[
            df_portfolios_resumo_exibicao["Retorno_Historico_Anual"].idxmax()
        ]

        carteira_menor_risco = df_portfolios_resumo_exibicao.loc[
            df_portfolios_resumo_exibicao["Risco_Anual"].idxmin()
        ]

        beta_medio = df_portfolios_resumo_exibicao["Beta"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Carteiras confirmadas",
            len(df_portfolios_resumo_exibicao),
        )

        col2.metric(
            "Menor risco",
            f"{carteira_menor_risco['Risco_Anual']:.2%}",
            carteira_menor_risco["Carteira_Exibicao"],
        )

        col3.metric(
            "Maior retorno",
            f"{carteira_maior_retorno['Retorno_Historico_Anual']:.2%}",
            carteira_maior_retorno["Carteira_Exibicao"],
        )

        col4.metric(
            "Beta médio",
            f"{beta_medio:.3f}",
        )

        st.markdown("---")

        # =====================================================
        # Taxa livre de risco
        # =====================================================

        st.subheader("🏦 Taxa livre de risco")

        esq, centro, dir = st.columns([1, 2, 1])

        with centro:
            c1, c2 = st.columns(2)

            c1.metric(
                "Anual",
                f"{taxa_livre_risco_anual:.2%}",
            )

            c2.metric(
                "Mensal",
                f"{float(taxa_livre_risco_mensal.mean()):.2%}",
            )

        st.markdown("---")

    # =====================================================
    # Resumo econômico
    # =====================================================

    st.subheader("📖 Resumo econômico")

    st.caption(
        "Indicadores macroeconômicos utilizados durante os cálculos."
    )

    st.dataframe(
        resumo_economico,
        use_container_width=True,
        hide_index=True,
    )
# ============================================================
# ABA ANÁLISE DETALHADA (mantida exatamente como antes)
# ============================================================
with tab_detalhes:
    st.header("Análise Detalhada")

    # ------------------------------------------------------------
    # 1. MAPAS DE CALOR (Correlação e Covariância) lado a lado
    # ------------------------------------------------------------
    st.subheader("Matrizes de Correlação e Covariância")
    col_cov, col_corr = st.columns(2)

    with col_corr:
        st.markdown("**Correlação (%)**")

        # Correlação em porcentagem apenas para exibição no gráfico
        corr_pct = (matriz_corr * 100).round(2)

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_pct.values,
                x=corr_pct.columns,
                y=corr_pct.index,
                colorscale="RdBu_r",
                zmin=-100,
                zmax=100,
                zmid=0,
                text=corr_pct.values,
                texttemplate="%{text:.1f}%",
                textfont={"size": 10, "color": "black"},
                colorbar=dict(title="Correlação (%)"),
                hovertemplate=(
                    "<b>%{y}</b> x <b>%{x}</b><br>"
                    "Correlação: %{z:.2f}%<extra></extra>"
                ),
                hoverongaps=False,
            )
        )

        fig_corr.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_cov:
        st.markdown("**Covariância**")

        cov_rounded = matriz_cov.round(4)

        fig_cov = go.Figure(
            data=go.Heatmap(
                z=cov_rounded.values,
                x=cov_rounded.columns,
                y=cov_rounded.index,
                colorscale="Viridis",
                text=cov_rounded.values,
                texttemplate="%{text:.4f}",
                textfont={"size": 8, "color": "white"},
                colorbar=dict(title="Covariância"),
                hovertemplate=(
                    "<b>%{y}</b> x <b>%{x}</b><br>"
                    "Covariância: %{z:.4f}<extra></extra>"
                ),
                hoverongaps=False,
            )
        )

        fig_cov.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # 2. ATIVOS E RETORNOS (Tabela + Gráfico de barras agrupadas)
    # ------------------------------------------------------------
    st.subheader("Análise Individual dos Ativos")
    col3, col4 = st.columns(2)

    with col3:
        st.dataframe(df_analise_ativos, use_container_width=True, height=300)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Posições das barras
        x = np.arange(len(df_analise_ativos["Ativo"]))
        width = 0.35  # largura das barras
        
        # Barras de Retorno Anual
        bars1 = ax.bar(x - width/2, df_analise_ativos["Retorno_Medio_Historico_Anual"], 
                    width, label="Retorno Anual", color="#2E86AB", edgecolor='black')
        # Barras de Volatilidade Anual
        bars2 = ax.bar(x + width/2, df_analise_ativos["Volatilidade_Anual"], 
                    width, label="Volatilidade Anual", color="#A23B72", edgecolor='black')
        
        # Linha de referência em zero
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # Rótulos e título
        ax.set_ylabel('Valor (anual)', fontsize=10)
        ax.set_title('Retorno vs Volatilidade por Ativo', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df_analise_ativos["Ativo"], rotation=0, fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Legenda posicionada abaixo do gráfico (centralizada)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)
        
        # Ajusta o layout para reservar espaço para a legenda inferior
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
    # ------------------------------------------------------------
    # 3. PREÇOS E RETORNOS MENSAIS (Tabela reduzida + gráfico)
    # ------------------------------------------------------------
    st.subheader("Preços e Retornos Mensais")
    col5, col6 = st.columns(2)
    with col5:
        st.dataframe(precos_mensais, use_container_width=True, height=250)
    with col6:
        acumulado = (1 + retornos_mensais[ativos]).cumprod()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for ativo in ativos:
            ax.plot(acumulado.index, acumulado[ativo], label=ativo, linewidth=1.5)
        ax.set_title("Retorno acumulado")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # 4. FRONTEIRA EFICIENTE E CARTEIRAS (Tabela + 2D + 3D)
    # ------------------------------------------------------------
    st.subheader("Fronteira Eficiente e Carteiras")

    # Remove carteiras duplicadas com base nos pesos
    peso_cols = [f"Peso_{a}" for a in ativos]

    df_carteiras_exibicao = df_carteiras.copy()
    df_carteiras_exibicao["_chave_pesos"] = (
        df_carteiras_exibicao[peso_cols]
        .round(2)
        .astype(str)
        .agg("|".join, axis=1)
    )

    df_carteiras_exibicao = (
        df_carteiras_exibicao
        .drop_duplicates(subset=["_chave_pesos"], keep="first")
        .drop(columns=["_chave_pesos"])
        .reset_index(drop=True)
    )

    # Primeira linha: tabela + gráfico 2D
    col7, col8 = st.columns(2)

    with col7:
        st.dataframe(
            df_carteiras_exibicao[["Carteira"] + peso_cols + ["Retorno_Esperado_Anual", "Risco_Anual"]],
            use_container_width=True,
            height=250
        )

    with col8:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df_ineficiente["Risco_Mensal"],
            df_ineficiente["Retorno_Esperado_Mensal"],
            alpha=0.2,
            s=10,
            label="Ineficientes"
        )

        ax.scatter(
            df_eficiente["Risco_Mensal"],
            df_eficiente["Retorno_Esperado_Mensal"],
            alpha=0.6,
            s=15,
            label="Eficientes"
        )

        # Mostra apenas carteiras únicas no gráfico
        for _, row in df_carteiras_exibicao.iterrows():
            ax.scatter(
                row["Risco_Mensal"],
                row["Retorno_Esperado_Mensal"],
                s=80,
                marker="D",
                label=row["Carteira"],
                edgecolors="white"
            )

        # --- Mínima variância ---
        ax.scatter(
            carteira_gmv["Risco_Mensal"],
            carteira_gmv["Retorno_Esperado_Mensal"],
            s=100,
            marker="x",
            color="green",
            linewidth=2,
            label="Mínima variância",
            zorder=5
        )

        ax.set_xlabel("Risco mensal")
        ax.set_ylabel("Retorno esperado mensal")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Segunda linha: gráfico 3D centralizado e maior
    with st.spinner("Gerando gráfico 3D da fronteira..."):
        ativo_destaque = ativos[0]
        col_z = f"Peso_{ativo_destaque}"

        fig3d = go.Figure()

        # Azul = Ineficientes
        fig3d.add_trace(go.Scatter3d(
            x=df_ineficiente["Retorno_Esperado_Mensal"],
            y=df_ineficiente["Risco_Mensal"],
            z=df_ineficiente[col_z],
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.20,
                color="#1f77b4"
            ),
            name="Ineficientes",
        ))

        # Laranja = Eficientes
        fig3d.add_trace(go.Scatter3d(
            x=df_eficiente["Retorno_Esperado_Mensal"],
            y=df_eficiente["Risco_Mensal"],
            z=df_eficiente[col_z],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.75,
                color="#ff7f0e"
            ),
            name="Eficientes",
        ))

        # Linha da fronteira eficiente em laranja
        fig3d.add_trace(go.Scatter3d(
            x=df_eficiente["Retorno_Esperado_Mensal"],
            y=df_eficiente["Risco_Mensal"],
            z=df_eficiente[col_z],
            mode="lines",
            line=dict(width=5, color="#ff7f0e"),
            name="Fronteira eficiente",
        ))

        # --- Carteiras confirmadas (pontos únicos) ---
        fig3d.add_trace(go.Scatter3d(
            x=df_carteiras_exibicao["Retorno_Esperado_Mensal"],
            y=df_carteiras_exibicao["Risco_Mensal"],
            z=df_carteiras_exibicao[col_z],
            mode="markers",
            marker=dict(
                size=7,
                symbol="diamond",
                color="#444444",
                opacity=0.95
            ),
            name="Carteiras",
            text=df_carteiras_exibicao["Carteira"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Retorno: %{x:.4f}<br>"
                "Risco: %{y:.4f}<br>"
                f"Peso em {ativo_destaque}: %{{z:.2%}}<extra></extra>"
            ),
        ))

        # --- Mínima variância ---
        fig3d.add_trace(go.Scatter3d(
            x=[carteira_gmv["Retorno_Esperado_Mensal"]],
            y=[carteira_gmv["Risco_Mensal"]],
            z=[carteira_gmv[col_z]],
            mode="markers",
            marker=dict(size=11, symbol="x", color="green"),
            name="Mínima variância",
        ))

        fig3d.update_layout(
            title=f"Fronteira eficiente 3D com {len(ativos)} ativos",
            scene=dict(
                xaxis_title="Retorno esperado mensal",
                yaxis_title="Risco mensal",
                zaxis_title=f"Peso em {ativo_destaque}",
                aspectmode="cube"
            ),
            height=650,
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(x=1.02, y=1),
        )

    col_left, col_mid, col_right = st.columns([0.5, 9, 0.5])
    with col_mid:
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
    # ------------------------------------------------------------
    # 5. BETA E CAPM (Tabela + gráficos lado a lado + regressão)
    # ------------------------------------------------------------
    st.subheader("Beta e CAPM")

    # Remove carteiras duplicadas apenas para exibição
    peso_cols = [f"Peso_{a}" for a in ativos]

    df_beta_carteiras_exibicao = df_beta_carteiras.copy()
    df_beta_carteiras_exibicao["_chave_pesos"] = (
        df_carteiras[peso_cols]
        .round(8)
        .astype(str)
        .agg("|".join, axis=1)
    )

    df_beta_carteiras_exibicao = (
        df_beta_carteiras_exibicao
        .drop_duplicates(subset=["_chave_pesos"], keep="first")
        .drop(columns=["_chave_pesos"])
        .reset_index(drop=True)
    )

    st.write("**Ativos**")
    st.dataframe(
        df_beta_ativos[["Ativo", "Beta", "Retorno_Historico_Anual", "Retorno_CAPM_Anual"]],
        use_container_width=True,
        height=250
    )

    st.write("**Carteiras**")
    st.dataframe(
        df_beta_carteiras_exibicao[["Carteira", "Beta", "Alpha_Mensal", "Retorno_Historico_Anual", "Retorno_CAPM_Anual"]],
        use_container_width=True,
        height=250
    )

    # Dois gráficos lado a lado (Beta e Alpha)
    col_graf1, col_graf2 = st.columns(2)

    # Junta ativos + carteiras para mostrar tudo no mesmo gráfico
    df_beta_geral = pd.concat([
        df_beta_ativos[["Ativo", "Beta", "Alpha_Anual"]].rename(columns={"Ativo": "Nome"}),
        df_beta_carteiras_exibicao[["Carteira", "Beta", "Alpha_Mensal"]].rename(
            columns={"Carteira": "Nome", "Alpha_Mensal": "Alpha_Anual"}
        )
    ], ignore_index=True)

    df_beta_geral["Tipo"] = [
        "Ativo"] * len(df_beta_ativos) + ["Carteira"] * len(df_beta_carteiras_exibicao)
    df_beta_geral["Rotulo"] = df_beta_geral["Nome"]

    with col_graf1:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        cores_beta = ["#6A0DAD" if t == "Ativo" else "#2E86AB" for t in df_beta_geral["Tipo"]]

        ax.bar(
            df_beta_geral["Rotulo"],
            df_beta_geral["Beta"],
            color=cores_beta,
            alpha=0.85,
            edgecolor="black"
        )
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="β = 1.0 (mercado)")
        ax.set_ylabel("Beta", fontsize=10)
        ax.set_title("Beta por Ativo e Carteira", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=1)
        plt.xticks(rotation=0)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)

    with col_graf2:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        cores_alpha = ["#FF8C00" if t == "Ativo" else "#1f77b4" for t in df_beta_geral["Tipo"]]

        ax.bar(
            df_beta_geral["Rotulo"],
            df_beta_geral["Alpha_Anual"],
            color=cores_alpha,
            alpha=0.85,
            edgecolor="black"
        )
        ax.axhline(y=0, color="green", linestyle="-", linewidth=1, label="α = 0 (referência)")
        ax.set_ylabel("Alpha Anual", fontsize=10)
        ax.set_title("Alpha Anual por Ativo e Carteira", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=1)
        plt.xticks(rotation=0)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)
    # --- Regressão da carteira vs mercado (largura total) ---
    st.subheader("Regressão da carteira vs mercado")

    carteiras_unicas = df_beta_carteiras_exibicao["Carteira"].tolist()

    carteira_regressao_nome = st.selectbox(
        "Escolha a carteira para a regressão",
        carteiras_unicas,
        index=0,
    )

    retorno_carteira_regressao = retornos_carteiras_series[carteira_regressao_nome]
    df_beta_carteira = pd.concat([retorno_carteira_regressao, retorno_mercado], axis=1).dropna()
    df_beta_carteira.columns = ["carteira", "mercado"]

    x = df_beta_carteira["mercado"].values
    y = df_beta_carteira["carteira"].values

    beta_regressao, intercepto = np.polyfit(x, y, 1)
    reta = np.poly1d([beta_regressao, intercepto])
    x_linha = np.linspace(x.min(), x.max(), 200)
    y_linha = reta(x_linha)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, alpha=0.7, label="Observações")
    ax.plot(x_linha, y_linha, linewidth=2, label=f"β = {beta_regressao:.2f}")
    ax.set_title(f"Regressão linear da {carteira_regressao_nome} contra o mercado")
    ax.set_xlabel("Retorno do mercado")
    ax.set_ylabel("Retorno da carteira")
    ax.axhline(0, linewidth=1, color="gray", linestyle="--")
    ax.axvline(0, linewidth=1, color="gray", linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)    
    # ------------------------------------------------------------
    # 6. OUTRAS ANÁLISES (Tabela de risco + gráfico de barras)
    # ------------------------------------------------------------
    st.subheader("Risco dos Ativos e das Carteiras")

    # Remove carteiras duplicadas apenas para exibição
    peso_cols = [f"Peso_{a}" for a in ativos]

    df_carteiras_exibicao = df_carteiras.copy()
    df_carteiras_exibicao["_chave_pesos"] = (
        df_carteiras_exibicao[peso_cols]
        .round(2)
        .astype(str)
        .agg("|".join, axis=1)
    )

    df_carteiras_exibicao = (
        df_carteiras_exibicao
        .drop_duplicates(subset=["_chave_pesos"], keep="first")
        .drop(columns=["_chave_pesos"])
        .reset_index(drop=True)
    )
    # Preparar dados para a tabela
    riscos_ativos = [retornos_mensais[ativo].std() for ativo in ativos]
    nomes_risco = ativos + df_carteiras_exibicao["Carteira"].tolist()
    valores_risco = riscos_ativos + df_carteiras_exibicao["Risco_Mensal"].tolist()

    df_risco = pd.DataFrame({
        "Ativo / Carteira": nomes_risco,
        "Desvio Padrão Mensal": valores_risco
    })

    col11, col12 = st.columns(2)

    with col11:
        st.dataframe(df_risco, use_container_width=True, height=250)

    with col12:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Define cores: azul para ativos, laranja para carteiras
        cores = ["skyblue"] * len(ativos) + ["orange"] * len(df_carteiras_exibicao)

        barras = ax.bar(nomes_risco, valores_risco, color=cores, edgecolor="navy")
        ax.set_title("Desvio-padrão mensal")
        ax.set_ylabel("Volatilidade")
        ax.grid(True, axis="y", alpha=0.3)

        for barra in barras:
            altura = barra.get_height()
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                altura,
                f"{altura:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        # Legenda manual
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="skyblue", label="Ativos"),
            Patch(facecolor="orange", label="Carteiras")
        ]
        ax.legend(handles=legend_elements, loc="upper left")

        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
with tab_projecao:
    st.subheader("📈 Projeção do investimento")

    st.caption(
        "Projeção estimada baseada nos retornos históricos da própria carteira, usando retorno geométrico mensal e faixa de confiança. "
        "A linha principal mostra o cenário base; as linhas pontilhadas indicam uma faixa conservadora e otimista. "
        "O gráfico usa datas reais para deixar claro quando a projeção começa e termina."
    )

    if valor_investido <= 0:
        st.warning("Informe um valor maior que zero em 'Quanto vai investir (R$)' para visualizar a projeção.")
    elif "Carteira" not in df_portfolios_resumo.columns:
        st.info("Não há carteiras confirmadas suficientes para projetar.")
    else:
        col_h1, col_h2, col_h3 = st.columns([1, 1, 1])

        with col_h1:
            unidade = st.radio(
                "Mostrar horizonte em",
                ["Meses", "Anos"],
                horizontal=True,
                key="projecao_unidade",
            )

        with col_h2:
            if unidade == "Meses":
                horizonte = st.number_input(
                    "Quantidade de meses",
                    min_value=1,
                    max_value=600,
                    value=12,
                    step=1,
                    key="projecao_horizonte_meses",
                )
            else:
                horizonte = st.number_input(
                    "Quantidade de anos",
                    min_value=1,
                    max_value=50,
                    value=3,
                    step=1,
                    key="projecao_horizonte_anos",
                )

        with col_h3:
            carteira_foco = None
            if "Carteira" in df_portfolios_resumo.columns and not df_portfolios_resumo.empty:
                if "Carteira_Exibicao" in df_portfolios_resumo.columns:
                    carteiras_disponiveis = df_portfolios_resumo["Carteira_Exibicao"].tolist()
                else:
                    carteiras_disponiveis = df_portfolios_resumo["Carteira"].tolist()

                carteiras_disponiveis = list(dict.fromkeys(carteiras_disponiveis))
                carteira_foco = st.selectbox(
                    "Carteira em destaque",
                    carteiras_disponiveis,
                    index=0 if carteiras_disponiveis else None,
                    key="projecao_carteira_foco",
                )

        horizonte = int(horizonte)
        total_passos = horizonte if unidade == "Meses" else horizonte * 12
        if total_passos < 1:
            st.stop()

        df_proj = df_portfolios_resumo.copy()

        if "Carteira" in df_proj.columns:
            df_proj["Carteira_Exibicao"] = df_proj["Carteira"].replace({
                "Carteira_1": "Carteira principal"
            })
            df_proj = df_proj.drop_duplicates(subset=["Carteira"], keep="first").reset_index(drop=True)
        else:
            df_proj["Carteira_Exibicao"] = "Carteira principal"

        # ----------------------------
        # Datas reais da projeção
        # ----------------------------
        ultima_data_hist = None
        try:
            if retornos_carteiras_series:
                datas_validas = []
                for serie in retornos_carteiras_series.values():
                    if serie is not None and not serie.dropna().empty:
                        datas_validas.append(serie.dropna().index.max())
                if datas_validas:
                    ultima_data_hist = max(datas_validas)
        except Exception:
            ultima_data_hist = None

        if ultima_data_hist is None or pd.isna(ultima_data_hist):
            ultima_data_hist = pd.Timestamp.today()

        data_inicio_proj = (pd.Timestamp(ultima_data_hist) + pd.offsets.MonthEnd(1)).normalize()
        datas_proj = pd.date_range(start=data_inicio_proj, periods=total_passos + 1, freq="ME")

        x_steps = np.arange(0, total_passos + 1, 1)
        x_label = "Data"

        projeccoes = []
        fig_proj = go.Figure()

        taxa_livre_risco_anual_local = float(taxa_livre_risco_anual)

        ibov_anual_estimado = np.nan
        ibov_mensal_media = np.nan
        if "Retorno_Mercado_IBOV" in retornos_mensais.columns:
            ibov_mensal_media = float(retornos_mensais["Retorno_Mercado_IBOV"].mean())
            ibov_anual_estimado = (1 + ibov_mensal_media) ** 12 - 1

        for _, row in df_proj.iterrows():
            carteira_nome = str(row.get("Carteira", "")).strip()
            carteira_exibicao = str(row.get("Carteira_Exibicao", carteira_nome or "Carteira"))

            serie_carteira = retornos_carteiras_series.get(carteira_nome)

            stats = None
            if serie_carteira is not None and not serie_carteira.dropna().empty:
                stats = estimate_monthly_return_bounds(serie_carteira)

            if stats is None:
                retorno_fallback = row.get("Retorno_Esperado_Mensal", np.nan)
                if pd.isna(retorno_fallback):
                    retorno_fallback = row.get("Retorno_Historico_Mensal", np.nan)
                if pd.isna(retorno_fallback):
                    continue

                retorno_base = float(retorno_fallback)
                retorno_conservador = retorno_base
                retorno_otimista = retorno_base
                origem = "Resumo da carteira"
                amostra = 0
            else:
                retorno_base = float(stats["base"])
                retorno_conservador = float(stats["conservador"])
                retorno_otimista = float(stats["otimista"])
                origem = "Retornos históricos"
                amostra = int(stats["amostra"])

            valor_base = project_portfolio_values(valor_investido, retorno_base, x_steps)
            valor_cons = project_portfolio_values(valor_investido, retorno_conservador, x_steps)
            valor_oti = project_portfolio_values(valor_investido, retorno_otimista, x_steps)

            ganho_base = valor_base - valor_investido

            idx_acima_base = next((i for i, v in enumerate(valor_base) if v >= valor_investido), None)
            idx_abaixo_base = next((i for i, v in enumerate(valor_base) if v < valor_investido), None)

            tabela_evolucao = pd.DataFrame({
                "Data": datas_proj,
                "Mes": x_steps,
                "Valor_Conservador": valor_cons,
                "Valor_Base": valor_base,
                "Valor_Otimista": valor_oti,
            })
            tabela_evolucao["Resultado_Base_vs_Inicial"] = np.where(
                tabela_evolucao["Valor_Base"] >= valor_investido, "Ganho", "Perda"
            )
            tabela_evolucao["Ganho_Base_R$"] = tabela_evolucao["Valor_Base"] - valor_investido
            tabela_evolucao["Ganho_Base_%"] = tabela_evolucao["Valor_Base"] / valor_investido - 1.0

            projeccoes.append({
                "Carteira": carteira_exibicao,
                "Origem": origem,
                "Amostra_Mensal": amostra,
                "Data_Inicio_Projecao": datas_proj[0].date(),
                "Data_Final_Projecao": datas_proj[-1].date(),
                "Horizonte": f"{horizonte} {unidade.lower()}",
                "Retorno_Mensal_Base": retorno_base,
                "Retorno_Mensal_Conservador": retorno_conservador,
                "Retorno_Mensal_Otimista": retorno_otimista,
                "Retorno_Anual_Estimado_Base": (1 + retorno_base) ** 12 - 1,
                "Valor_Inicial": float(valor_investido),
                "Valor_Final_Base": float(valor_base[-1]),
                "Valor_Final_Conservador": float(valor_cons[-1]),
                "Valor_Final_Otimista": float(valor_oti[-1]),
                "Ganho_Absoluto_Base": float(ganho_base[-1]),
                "Ganho_%_Base": float(valor_base[-1] / valor_investido - 1.0),
                "Primeiro_Mes_de_Ganho_Base": datas_proj[idx_acima_base].date() if idx_acima_base is not None else "—",
                "Primeiro_Mes_de_Perca_Base": datas_proj[idx_abaixo_base].date() if idx_abaixo_base is not None else "—",
                "_tabela_evolucao": tabela_evolucao,
            })

            fig_proj.add_trace(go.Scatter(
                x=datas_proj,
                y=valor_cons,
                mode="lines",
                line=dict(width=1, dash="dot"),
                opacity=0.40,
                name=f"{carteira_exibicao} (conservador)",
                hovertemplate=(
                    f"<b>{carteira_exibicao}</b><br>"
                    "Data: %{x|%d/%m/%Y}<br>"
                    "Valor projetado: R$ %{y:,.2f}<extra></extra>"
                ),
            ))

            fig_proj.add_trace(go.Scatter(
                x=datas_proj,
                y=valor_base,
                mode="lines+markers",
                line=dict(width=3),
                name=f"{carteira_exibicao} (base)",
                hovertemplate=(
                    f"<b>{carteira_exibicao}</b><br>"
                    "Data: %{x|%d/%m/%Y}<br>"
                    "Valor projetado: R$ %{y:,.2f}<extra></extra>"
                ),
            ))

            fig_proj.add_trace(go.Scatter(
                x=datas_proj,
                y=valor_oti,
                mode="lines",
                line=dict(width=1, dash="dot"),
                opacity=0.40,
                name=f"{carteira_exibicao} (otimista)",
                hovertemplate=(
                    f"<b>{carteira_exibicao}</b><br>"
                    "Data: %{x|%d/%m/%Y}<br>"
                    "Valor projetado: R$ %{y:,.2f}<extra></extra>"
                ),
            ))

        if not projeccoes:
            st.warning("Não foi possível montar a projeção com os dados disponíveis.")
        else:
            df_projecao = pd.DataFrame(projeccoes).sort_values("Valor_Final_Base", ascending=False).reset_index(drop=True)
            df_projecao_exib = df_projecao.copy()

            for col in [
                "Retorno_Mensal_Base",
                "Retorno_Mensal_Conservador",
                "Retorno_Mensal_Otimista",
                "Retorno_Anual_Estimado_Base",
                "Ganho_%_Base"
            ]:
                if col in df_projecao_exib.columns:
                    df_projecao_exib[col] = df_projecao_exib[col].map(
                        lambda v: f"{v:.2%}" if pd.notna(v) else "—"
                    )

            for col in [
                "Valor_Inicial",
                "Valor_Final_Base",
                "Valor_Final_Conservador",
                "Valor_Final_Otimista",
                "Ganho_Absoluto_Base"
            ]:
                if col in df_projecao_exib.columns:
                    df_projecao_exib[col] = df_projecao_exib[col].map(
                        lambda v: format_brl(v) if pd.notna(v) else "—"
                    )

            for col in [
                "Data_Inicio_Projecao",
                "Data_Final_Projecao",
                "Primeiro_Mes_de_Ganho_Base",
                "Primeiro_Mes_de_Perca_Base",
            ]:
                if col in df_projecao_exib.columns:
                    df_projecao_exib[col] = df_projecao_exib[col].map(
                        lambda v: v.strftime("%d/%m/%Y") if hasattr(v, "strftime") else str(v)
                    )

            # Tabela ocupando a largura toda
            st.dataframe(
                df_projecao_exib[[
                    "Carteira",
                    "Origem",
                    "Amostra_Mensal",
                    "Data_Inicio_Projecao",
                    "Data_Final_Projecao",
                    "Retorno_Mensal_Base",
                    "Retorno_Mensal_Conservador",
                    "Retorno_Mensal_Otimista",
                    "Valor_Inicial",
                    "Valor_Final_Base",
                    "Valor_Final_Conservador",
                    "Valor_Final_Otimista",
                    "Ganho_Absoluto_Base",
                    "Ganho_%_Base",
                    "Primeiro_Mes_de_Ganho_Base",
                    "Primeiro_Mes_de_Perca_Base",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            # Leitura rápida abaixo da tabela
            st.markdown("### Leitura rápida")

            if carteira_foco and "Carteira" in df_projecao.columns:
                foco_row = df_projecao[df_projecao["Carteira"] == carteira_foco]
                if foco_row.empty:
                    foco_row = df_projecao.iloc[[0]]
            else:
                foco_row = df_projecao.iloc[[0]]

            foco = foco_row.iloc[0]

            retorno_base_foco = float(foco["Retorno_Mensal_Base"])
            retorno_anual_foco = (1 + retorno_base_foco) ** 12 - 1
            valor_final_foco = float(foco["Valor_Final_Base"])
            ganho_foco = float(foco["Ganho_Absoluto_Base"])

            if retorno_anual_foco > taxa_livre_risco_anual_local:
                status = "Acima da Selic"
            elif not np.isnan(ibov_anual_estimado) and retorno_anual_foco > ibov_anual_estimado:
                status = "Acima do mercado"
            else:
                status = "Abaixo da Selic/mercado"

            resumo_col1, resumo_col2 = st.columns(2)

            with resumo_col1:
                st.metric("Carteira em destaque", foco["Carteira"])
                st.metric("Retorno base mensal", f"{retorno_base_foco:.2%}")
                st.metric("Retorno anual estimado", f"{retorno_anual_foco:.2%}")
                st.metric("Status", status)

            with resumo_col2:
                st.metric("Valor inicial", format_brl(valor_investido))
                st.metric("Valor final estimado", format_brl(valor_final_foco))
                st.metric("Lucro estimado", format_brl(ganho_foco))
                ibov_txt = f"{ibov_anual_estimado:.2%}" if pd.notna(ibov_anual_estimado) else "—"
                st.metric("IBOV anual estimado", ibov_txt)


            fig_proj.add_hline(
                y=valor_investido,
                line_width=2,
                line_dash="dash",
                line_color="#6B7280",
                annotation_text="Capital inicial",
                annotation_position="top left",
            )

            try:
                if carteira_foco:
                    fig_proj.add_annotation(
                        x=datas_proj[-1],
                        y=float(foco["Valor_Final_Base"]),
                        text=f"Foco: {carteira_foco}",
                        showarrow=True,
                        arrowhead=2,
                        ax=30,
                        ay=-30,
                    )
            except Exception:
                pass

            fig_proj.update_layout(
                title=f"Projeção do capital investido ao longo de {horizonte} {unidade.lower()}",
                xaxis_title=x_label,
                yaxis_title="Valor projetado (R$)",
                hovermode="x unified",
                legend_title_text="Carteiras",
                height=600,
                margin=dict(l=10, r=10, t=60, b=10),
                template="plotly_white",
            )

            fig_proj.update_xaxes(
                tickformat="%d/%m/%Y",
                showgrid=True,
            )
            fig_proj.update_yaxes(
                tickprefix="R$ ",
                separatethousands=True,
                showgrid=True,
            )

            st.plotly_chart(fig_proj, use_container_width=True)

            st.markdown("### Onde há ganho e onde há perda")

            carteira_base_sel = df_projecao.iloc[0]
            tabela_evolucao = carteira_base_sel["_tabela_evolucao"].copy()

            tabela_evolucao["Data"] = pd.to_datetime(tabela_evolucao["Data"]).dt.strftime("%d/%m/%Y")
            tabela_evolucao["Valor_Conservador"] = tabela_evolucao["Valor_Conservador"].map(format_brl)
            tabela_evolucao["Valor_Base"] = tabela_evolucao["Valor_Base"].map(format_brl)
            tabela_evolucao["Valor_Otimista"] = tabela_evolucao["Valor_Otimista"].map(format_brl)
            tabela_evolucao["Ganho_Base_R$"] = tabela_evolucao["Ganho_Base_R$"].map(format_brl)
            tabela_evolucao["Ganho_Base_%"] = tabela_evolucao["Ganho_Base_%"].map(lambda v: f"{v:.2%}")

            st.dataframe(
                tabela_evolucao[[
                    "Data",
                    "Mes",
                    "Valor_Conservador",
                    "Valor_Base",
                    "Valor_Otimista",
                    "Resultado_Base_vs_Inicial",
                    "Ganho_Base_R$",
                    "Ganho_Base_%",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Interpretação da projeção")

            col_i1, col_i2, col_i3 = st.columns(3)

            with col_i1:
                st.info(
                    "A linha pontilhada abaixo do capital inicial indica possível perda acumulada. "
                    "Se a linha base ficar acima da linha cinza, há ganho sobre o valor aplicado."
                )

            with col_i2:
                st.info(
                    "A projeção começa no primeiro fechamento mensal após o último dado histórico disponível, "
                    "e a data inicial/final aparece na tabela e no resumo."
                )

            with col_i3:
                st.info(
                    "A carteira em destaque pode ser trocada no seletor acima para comparar rapidamente cenários diferentes."
                )



    # --------------------------------------------------------
    # tab_risco_simulado
    # --------------------------------------------------------

with tab_risco_simulado:
    st.subheader("🧪 Simulação de risco da carteira")

    st.caption(
        "Aqui a leitura deixa de ser uma curva única e passa a mostrar faixas prováveis de resultado. "
        "A simulação usa os retornos mensais históricos da carteira escolhida, por bootstrap ou Monte Carlo, "
        "e calcula a chance de perda, a chance de superar Selic e IBOV, além da distribuição dos resultados finais."
    )

    if valor_investido <= 0:
        st.warning("Informe um valor maior que zero em 'Quanto vai investir (R$)' para simular o risco.")
    elif "Carteira" not in df_portfolios_resumo.columns:
        st.info("Não há carteiras confirmadas suficientes para simular.")
    else:
        col_r1, col_r2, col_r3, col_r4 = st.columns([1.2, 1.2, 1.3, 1.3])

        with col_r1:
            unidade_risco = st.radio(
                "Horizonte",
                ["Meses", "Anos"],
                horizontal=True,
                key="risco_unidade",
            )

        with col_r2:
            if unidade_risco == "Meses":
                horizonte_risco = st.number_input(
                    "Quantidade de meses",
                    min_value=1,
                    max_value=600,
                    value=12,
                    step=1,
                    key="risco_horizonte_meses",
                )
            else:
                horizonte_risco = st.number_input(
                    "Quantidade de anos",
                    min_value=1,
                    max_value=50,
                    value=3,
                    step=1,
                    key="risco_horizonte_anos",
                )

        with col_r3:
            metodo_simulacao = st.selectbox(
                "Método",
                ["Bootstrap histórico", "Monte Carlo normal"],
                index=0,
                key="risco_metodo",
            )

        with col_r4:
            n_simulacoes = st.number_input(
                "Simulações",
                min_value=500,
                max_value=50000,
                value=5000,
                step=500,
                key="risco_n_simulacoes",
            )

        horizonte_risco = int(horizonte_risco)
        total_passos_risco = horizonte_risco if unidade_risco == "Meses" else horizonte_risco * 12

        df_risco_base = df_portfolios_resumo.copy()
        if "Carteira" in df_risco_base.columns:
            df_risco_base["Carteira_Exibicao"] = df_risco_base["Carteira"].replace({
                "Carteira_1": "Carteira principal"
            })
            df_risco_base = df_risco_base.drop_duplicates(subset=["Carteira"], keep="first").reset_index(drop=True)
        else:
            df_risco_base["Carteira_Exibicao"] = "Carteira principal"

        carteiras_disponiveis_risco = df_risco_base["Carteira_Exibicao"].tolist()
        if not carteiras_disponiveis_risco:
            st.info("Não há carteiras disponíveis para a simulação de risco.")
        else:
            carteira_foco_risco = st.selectbox(
                "Carteira em destaque",
                list(dict.fromkeys(carteiras_disponiveis_risco)),
                index=0,
                key="risco_carteira_foco",
            )

            if carteira_foco_risco:
                linha_foco = df_risco_base[df_risco_base["Carteira_Exibicao"] == carteira_foco_risco]
                if linha_foco.empty:
                    linha_foco = df_risco_base.iloc[[0]]
            else:
                linha_foco = df_risco_base.iloc[[0]]

            foco_risco = linha_foco.iloc[0]
        carteira_nome_interno = str(foco_risco.get("Carteira", "")).strip()
        serie_carteira = retornos_carteiras_series.get(carteira_nome_interno)

        if serie_carteira is None or serie_carteira.dropna().empty:
            st.warning("Não foi possível localizar a série histórica da carteira escolhida para simulação.")
        else:
            df_hist_risco = pd.to_numeric(serie_carteira, errors="coerce").dropna()
            df_hist_risco = df_hist_risco[df_hist_risco > -0.9999]

            if len(df_hist_risco) < 2:
                st.warning("Histórico insuficiente para simular risco com segurança.")
            else:
                ultima_data_hist_risco = None
                try:
                    ultima_data_hist_risco = df_hist_risco.index.max()
                except Exception:
                    ultima_data_hist_risco = pd.Timestamp.today()

                if ultima_data_hist_risco is None or pd.isna(ultima_data_hist_risco):
                    ultima_data_hist_risco = pd.Timestamp.today()

                data_inicio_proj_risco = (pd.Timestamp(ultima_data_hist_risco) + pd.offsets.MonthEnd(1)).normalize()
                datas_risco = pd.date_range(start=data_inicio_proj_risco, periods=total_passos_risco + 1, freq="ME")

                try:
                    paths = simulate_portfolio_paths(
                        df_hist_risco,
                        valor_investido,
                        total_passos_risco,
                        n_simulations=int(n_simulacoes),
                        method=metodo_simulacao,
                        seed=42,
                    )
                except Exception as e:
                    st.error(f"Não foi possível rodar a simulação: {e}")
                else:
                    ibov_mensal_media_local = np.nan
                    if "Retorno_Mercado_IBOV" in retornos_mensais.columns:
                        ibov_mensal_media_local = float(retornos_mensais["Retorno_Mercado_IBOV"].mean())

                    selic_mensal_media_local = float(taxa_livre_risco_mensal.mean())

                    benchmark_values = {}
                    benchmark_curvas = {}

                    benchmark_values["selic"] = float(valor_investido) * (1 + selic_mensal_media_local) ** total_passos_risco
                    benchmark_curvas["Selic"] = valor_investido * np.power(1 + selic_mensal_media_local, np.arange(total_passos_risco + 1))

                    if pd.notna(ibov_mensal_media_local):
                        benchmark_values["ibov"] = float(valor_investido) * (1 + ibov_mensal_media_local) ** total_passos_risco
                        benchmark_curvas["IBOV"] = valor_investido * np.power(1 + ibov_mensal_media_local, np.arange(total_passos_risco + 1))

                    resumo_sim = summarize_simulation_paths(
                        paths,
                        valor_investido,
                        benchmark_values=benchmark_values,
                    )

                    running_max = np.maximum.accumulate(paths, axis=1)
                    drawdowns = paths / running_max - 1.0
                    max_drawdown = drawdowns.min(axis=1)

                    p05 = np.percentile(paths, 5, axis=0)
                    p10 = np.percentile(paths, 10, axis=0)
                    p50 = np.percentile(paths, 50, axis=0)
                    p90 = np.percentile(paths, 90, axis=0)
                    p95 = np.percentile(paths, 95, axis=0)

                    df_risco_resumo = pd.DataFrame([{
                        "Carteira": foco_risco["Carteira_Exibicao"],
                        "Carteira_Interna": carteira_nome_interno,
                        "Metodo": metodo_simulacao,
                        "Horizonte": f"{horizonte_risco} {unidade_risco.lower()}",
                        "Amostra_Mensal": int(len(df_hist_risco)),
                        "Simulacoes": int(n_simulacoes),
                        "Retorno_Mensal_Medio_Historico": float(df_hist_risco.mean()),
                        "Volatilidade_Mensal_Historica": float(df_hist_risco.std()),
                        "Chance_de_Perder_Capital": resumo_sim["chance_perda"],
                        "Chance_Superar_Selic": resumo_sim.get("chance_superar_selic", np.nan),
                        "Chance_Superar_IBOV": resumo_sim.get("chance_superar_ibov", np.nan),
                        "Valor_Final_Medio": resumo_sim["media_final"],
                        "Valor_Final_Mediano": resumo_sim["mediana_final"],
                        "Valor_Final_P05": resumo_sim["p05_final"],
                        "Valor_Final_P10": resumo_sim["p10_final"],
                        "Valor_Final_P90": resumo_sim["p90_final"],
                        "Valor_Final_P95": resumo_sim["p95_final"],
                        "Max_Drawdown_Medio": resumo_sim["max_drawdown_medio"],
                        "Max_Drawdown_P05": resumo_sim["max_drawdown_p05"],
                        "Max_Drawdown_Mediano": resumo_sim["max_drawdown_p50"],
                    }])

                    df_simulacao_risco = df_risco_resumo.copy()

                    colm1, colm2, colm3, colm4 = st.columns(4)

                    colm1.metric("Chance de perda", f"{resumo_sim['chance_perda']:.2%}")
                    chance_selic = resumo_sim.get("chance_superar_selic", np.nan)
                    chance_ibov = resumo_sim.get("chance_superar_ibov", np.nan)
                    colm2.metric("Chance de superar a Selic", f"{chance_selic:.2%}" if pd.notna(chance_selic) else "—")
                    colm3.metric("Chance de superar o IBOV", f"{chance_ibov:.2%}" if pd.notna(chance_ibov) else "—")
                    colm4.metric("Mediana final", format_brl(resumo_sim["mediana_final"]))

                    colm5, colm6 = st.columns(2)
                    with colm5:
                        st.metric("Pior cenário razoável (P5)", format_brl(resumo_sim["p05_final"]))
                        st.metric("Maior drawdown médio", f"{resumo_sim['max_drawdown_medio']:.2%}")
                    with colm6:
                        st.metric("Faixa provável (P10)", format_brl(resumo_sim["p10_final"]))
                        st.metric("Faixa provável (P90)", format_brl(resumo_sim["p90_final"]))

                    st.markdown("### Faixa provável de resultados")
                    fig_risco = go.Figure()

                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=p95,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=p05,
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        name="Faixa 5% a 95%",
                        hovertemplate="Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.2f}<extra></extra>",
                    ))

                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=p90,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=p10,
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        name="Faixa 10% a 90%",
                        hovertemplate="Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.2f}<extra></extra>",
                    ))

                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=p50,
                        mode="lines+markers",
                        line=dict(width=3),
                        name="Mediana da simulação",
                        hovertemplate="Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.2f}<extra></extra>",
                    ))

                    sample_size = min(20, paths.shape[0])
                    rng_local = np.random.default_rng(42)
                    sample_idx = rng_local.choice(paths.shape[0], size=sample_size, replace=False)
                    for idx in sample_idx:
                        fig_risco.add_trace(go.Scatter(
                            x=datas_risco,
                            y=paths[idx],
                            mode="lines",
                            line=dict(width=1),
                            opacity=0.12,
                            showlegend=False,
                            hoverinfo="skip",
                        ))

                    fig_risco.add_trace(go.Scatter(
                        x=datas_risco,
                        y=valor_investido * np.power(1 + selic_mensal_media_local, np.arange(total_passos_risco + 1)),
                        mode="lines",
                        line=dict(width=2, dash="dash"),
                        name="Selic média",
                        hovertemplate="Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.2f}<extra></extra>",
                    ))

                    if "IBOV" in benchmark_curvas:
                        fig_risco.add_trace(go.Scatter(
                            x=datas_risco,
                            y=benchmark_curvas["IBOV"],
                            mode="lines",
                            line=dict(width=2, dash="dot"),
                            name="IBOV médio",
                            hovertemplate="Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.2f}<extra></extra>",
                        ))

                    fig_risco.add_hline(
                        y=valor_investido,
                        line_width=2,
                        line_dash="dash",
                        line_color="#6B7280",
                        annotation_text="Capital inicial",
                        annotation_position="top left",
                    )

                    fig_risco.update_layout(
                        title=f"Distribuição provável para {foco_risco['Carteira_Exibicao']}",
                        xaxis_title="Data",
                        yaxis_title="Valor projetado (R$)",
                        hovermode="x unified",
                        legend_title_text="Cenários",
                        height=620,
                        margin=dict(l=10, r=10, t=60, b=10),
                        template="plotly_white",
                    )
                    fig_risco.update_xaxes(tickformat="%d/%m/%Y", showgrid=True)
                    fig_risco.update_yaxes(tickprefix="R$ ", separatethousands=True, showgrid=True)

                    st.plotly_chart(fig_risco, use_container_width=True)

                    st.markdown("### Resumo da simulação")
                    st.dataframe(
                        df_risco_resumo,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("### Leitura rápida")
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.info(
                            "A chance de perda mostra a fração de caminhos em que o valor final ficou abaixo do capital inicial."
                        )
                    with c2:
                        st.info(
                            "A faixa 10% a 90% mostra um intervalo mais realista do que uma única linha de projeção."
                        )
                    with c3:
                        st.info(
                            "Quanto mais larga a faixa, maior a incerteza; quanto maior a chance de superar Selic ou IBOV, melhor o cenário relativo."
                        )



    # --------------------------------------------------------
    # tab_arquivos
    # --------------------------------------------------------

with tab_arquivos:
    st.subheader("Arquivos gerados")
    with st.spinner("Gerando arquivo Excel..."):
        excel_path = BASE_DIR / "dados_mensais_CAPM_Varios_ativos.xlsx"

        # Versões sem duplicata para as abas de saída
        df_carteiras_excel = df_carteiras_exibicao.copy()
        df_beta_carteiras_excel = df_beta_carteiras.copy()

        # Remove duplicatas também da aba Beta_Carteiras
        df_beta_carteiras_excel["_chave_pesos"] = (
            df_carteiras[peso_cols]
            .round(8)
            .astype(str)
            .agg("|".join, axis=1)
        )
        df_beta_carteiras_excel = (
            df_beta_carteiras_excel
            .drop_duplicates(subset=["_chave_pesos"], keep="first")
            .drop(columns=["_chave_pesos"])
            .reset_index(drop=True)
        )

        # Resumo também sem duplicatas
        df_portfolios_resumo_excel = df_portfolios_resumo.copy()
        if "Carteira" in df_portfolios_resumo_excel.columns:
            pesos_resumo = [c for c in df_portfolios_resumo_excel.columns if c.startswith("Peso_")]
            if pesos_resumo:
                df_portfolios_resumo_excel["_chave_pesos"] = (
                    df_portfolios_resumo_excel[pesos_resumo]
                    .round(8)
                    .astype(str)
                    .agg("|".join, axis=1)
                )
                df_portfolios_resumo_excel = (
                    df_portfolios_resumo_excel
                    .drop_duplicates(subset=["_chave_pesos"], keep="first")
                    .drop(columns=["_chave_pesos"])
                    .reset_index(drop=True)
                )

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            precos_mensais.to_excel(writer, sheet_name="Precos_Mensais")
            retornos_mensais.to_excel(writer, sheet_name="Retornos_Mensais")
            df_analise_ativos.to_excel(writer, sheet_name="Analise_Ativos", index=False)
            matriz_cov.to_excel(writer, sheet_name="Covariancia")
            matriz_corr.to_excel(writer, sheet_name="Correlacao")
            df_carteiras_excel.to_excel(writer, sheet_name="Carteiras", index=False)
            df_fronteira.to_excel(writer, sheet_name="Fronteira_Eficiente", index=False)
            df_eficiente.to_excel(writer, sheet_name="Fronteira_Pontos_Eficientes", index=False)
            df_ineficiente.to_excel(writer, sheet_name="Fronteira_Pontos_Ineficientes", index=False)
            df_beta_ativos.to_excel(writer, sheet_name="Beta_Ativos", index=False)
            df_beta_carteiras_excel.to_excel(writer, sheet_name="Beta_Carteiras", index=False)
            df_portfolios_resumo_excel.to_excel(writer, sheet_name="Carteiras_Resumo", index=False)
            if not df_simulacao_risco.empty:
                df_simulacao_risco.to_excel(writer, sheet_name="Simulacao_Risco", index=False)
            resumo_economico.to_excel(writer, sheet_name="Resumo_Economico", index=False)

    st.success(f"Arquivo salvo em: {excel_path}")
    st.download_button(
        "Baixar Excel",
        data=excel_path.read_bytes(),
        file_name=excel_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )