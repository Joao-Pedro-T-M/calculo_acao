# Importar as bibliotecas
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
import numpy as np
import os
from bcb import sgs
from datetime import datetime
from google.colab import drive
drive.mount('/content/drive')
import warnings
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import scipy.stats as stats
import matplotlib.ticker as mtick
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from IPython.display import display, HTML
import requests
from mpl_toolkits.mplot3d import Axes3D  # para gráfico 3D
import streamlit as st
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1) Caminho da pasta no Drive
# ----------------------------------------------------
pasta_saida = '/content/drive/MyDrive/'
os.makedirs(pasta_saida, exist_ok=True)

# ----------------------------------------------------
# 2) Ativos escolhidos (agora com base flexível para mais empresas)
# ----------------------------------------------------
tickers = {
    'BRAP4': 'BRAP4.SA',
    'POMO4': 'POMO4.SA',
    'ITUB4': 'ITUB4.SA',
    'IBOV': '^BVSP'
}

# Se quiser trocar por 10 empresas, é só incluir aqui os ativos desejados.
# Exemplo:
# ativos = ['BRAP4', 'POMO4', 'ITUB4', 'VALE3', 'PETR4', 'WEGE3', 'BBAS3', 'B3SA3', 'SUZB3', 'LREN3']

ativos = ['BRAP4', 'POMO4', 'ITUB4']

# Esses identificadores só fazem sentido quando você estiver usando exatamente 3 ativos
ativo_A = 'BRAP4'
ativo_B = 'POMO4'
ativo_C = 'ITUB4'

# ----------------------------------------------------
# 3) Período
# ----------------------------------------------------
data_inicio = '2016-01-01'
data_fim = '2026-12-31'

# ----------------------------------------------------
# 4) Baixar preços diários
# ----------------------------------------------------
precos = yf.download(
    list(tickers.values()),
    start=data_inicio,
    end=data_fim,
    auto_adjust=False,
    progress=False
)['Close']

if isinstance(precos, pd.Series):
    precos = precos.to_frame()

precos = precos.rename(columns={v: k for k, v in tickers.items()})
precos = precos.sort_index()

# ----------------------------------------------------
# 5) Último pregão de cada mês
# ----------------------------------------------------
precos_mensais = precos.resample('M').last().dropna(how='all')

# ----------------------------------------------------
# 6) Retornos mensais
# ----------------------------------------------------
retornos_mensais = precos_mensais.pct_change().dropna()
retornos_mensais = retornos_mensais.rename(columns={'IBOV': 'Retorno_Mercado_IBOV'})

# ----------------------------------------------------
# 7) Análise individual dos ativos
# ----------------------------------------------------
analise_ativos = []
for ativo in ativos:
    retorno_medio = retornos_mensais[ativo].mean()
    volatilidade = retornos_mensais[ativo].std()

    analise_ativos.append({
        'Ativo': ativo,
        'Retorno_Medio_Historico_Mensal': retorno_medio,
        'Retorno_Medio_Historico_Anual': (1 + retorno_medio) ** 12 - 1,
        'Volatilidade_Mensal': volatilidade,
        'Volatilidade_Anual': volatilidade * np.sqrt(12),
        'Retorno_Medio_Positive': retorno_medio > 0
    })

df_analise_ativos = pd.DataFrame(analise_ativos)

# ----------------------------------------------------
# 8) Covariância e correlação
# ----------------------------------------------------
matriz_cov = retornos_mensais[ativos].cov()
matriz_corr = retornos_mensais[ativos].corr()

# ----------------------------------------------------
# 9) Carteiras (agora pronto para qualquer quantidade de ativos)
# ----------------------------------------------------
# Exemplo: mantém 3 carteiras para comparação,
# mas elas são criadas de forma automática a partir dos ativos escolhidos.
# Se houver mais de 3 ativos, as carteiras usam os 3 primeiros como base
# e o restante recebe peso zero.

carteiras = {}

if len(ativos) == 3:
    carteiras = {
        'Carteira_1': {ativos[0]: 0.75, ativos[1]: 0.25, ativos[2]: 0.00},
        'Carteira_2': {ativos[0]: 0.50, ativos[1]: 0.25, ativos[2]: 0.25},
        'Carteira_3': {ativos[0]: 0.33, ativos[1]: 0.33, ativos[2]: 0.34},
    }
else:
    # Estrutura-base para mais ativos
    carteira_1 = {a: 0.0 for a in ativos}
    carteira_2 = {a: 0.0 for a in ativos}
    carteira_3 = {a: 0.0 for a in ativos}

    # Distribuições exemplo usando os 3 primeiros ativos
    carteira_1[ativos[0]] = 0.75
    carteira_1[ativos[1]] = 0.25

    carteira_2[ativos[0]] = 0.50
    carteira_2[ativos[1]] = 0.25
    carteira_2[ativos[2]] = 0.25

    carteira_3[ativos[0]] = 0.33
    carteira_3[ativos[1]] = 0.33
    carteira_3[ativos[2]] = 0.34

    carteiras = {
        'Carteira_1': carteira_1,
        'Carteira_2': carteira_2,
        'Carteira_3': carteira_3,
    }

# Escolha da carteira principal
carteira_escolhida_nome = 'Carteira_2'
pesos_carteira_escolhida = np.array([carteiras[carteira_escolhida_nome].get(a, 0) for a in ativos])

resultados_carteiras = []
for nome_carteira, pesos in carteiras.items():
    pesos_array = np.array([pesos.get(a, 0) for a in ativos])

    retorno_carteira = retornos_mensais[ativos].mul(pesos_array, axis=1).sum(axis=1)
    retorno_esperado_mensal = retorno_carteira.mean()
    risco_mensal = retorno_carteira.std()

    risco_mensal_cov = np.sqrt(
        np.dot(pesos_array.T, np.dot(matriz_cov.values, pesos_array))
    )

    resultados_carteiras.append({
        'Carteira': nome_carteira,
        **{f'Peso_{a}': pesos.get(a, 0) for a in ativos},
        'Retorno_Esperado_Mensal': retorno_esperado_mensal,
        'Retorno_Esperado_Anual': (1 + retorno_esperado_mensal) ** 12 - 1,
        'Risco_Mensal': risco_mensal,
        'Risco_Mensal_Cov': risco_mensal_cov,
        'Risco_Anual': risco_mensal_cov * np.sqrt(12)
    })

df_carteiras = pd.DataFrame(resultados_carteiras)

# Extrair os dados da carteira escolhida para uso posterior
retorno_carteira_escolhida = retornos_mensais[ativos].mul(pesos_carteira_escolhida, axis=1).sum(axis=1)
retorno_hist_carteira_mensal = retorno_carteira_escolhida.mean()
risco_carteira_mensal = retorno_carteira_escolhida.std()
risco_carteira_anual = risco_carteira_mensal * np.sqrt(12)

df_carteira_escolhida = pd.DataFrame([{
    'Carteira': carteira_escolhida_nome,
    **{f'Peso_{a}': carteiras[carteira_escolhida_nome].get(a, 0) for a in ativos},
    'Retorno_Historico_Mensal': retorno_hist_carteira_mensal,
    'Retorno_Historico_Anual': (1 + retorno_hist_carteira_mensal) ** 12 - 1,
    'Risco_Mensal': risco_carteira_mensal,
    'Risco_Anual': risco_carteira_anual
}])

justificativa_carteira = (
    f"A {carteira_escolhida_nome} foi escolhida por apresentar equilíbrio entre risco e retorno. "
    f"Seu risco mensal foi {risco_carteira_mensal:.4f} e o retorno médio mensal foi {retorno_hist_carteira_mensal:.4f}."
)

print(justificativa_carteira)

# ----------------------------------------------------
# 11) Beta e CAPM (atualizado para qualquer quantidade de ativos)
# ----------------------------------------------------

# Taxa livre de risco (fornecida)
taxa_livre_risco_anual = 0.1450  # 14,50% ao ano
taxa_livre_risco_mensal = (1 + taxa_livre_risco_anual) ** (1/12) - 1

# Retorno do mercado (IBOV)
retorno_mercado = retornos_mensais['Retorno_Mercado_IBOV']

# ----------------------------------------------------
# Beta e CAPM para os ativos individualmente
# ----------------------------------------------------
resultados_beta = []

for ativo in ativos:
    # Junta retorno do ativo com o mercado
    df = pd.concat([retornos_mensais[ativo], retorno_mercado], axis=1).dropna()
    df.columns = ['ativo', 'mercado']

    # Excessos de retorno
    excesso_ativo = df['ativo'] - taxa_livre_risco_mensal
    excesso_mercado = df['mercado'] - taxa_livre_risco_mensal

    # Beta = Cov(excesso_ativo, excesso_mercado) / Var(excesso_mercado)
    beta = excesso_ativo.cov(excesso_mercado) / excesso_mercado.var()

    # Retorno histórico médio do ativo
    ret_hist_mensal = df['ativo'].mean()

    # Retorno esperado pelo CAPM (usando a média histórica do mercado)
    ret_capm_mensal = taxa_livre_risco_mensal + beta * (df['mercado'].mean() - taxa_livre_risco_mensal)

    # Alpha = retorno médio do ativo - retorno CAPM
    alpha_mensal = ret_hist_mensal - ret_capm_mensal

    resultados_beta.append({
        'Ativo': ativo,
        'Beta': beta,
        'Retorno_Historico_Mensal': ret_hist_mensal,
        'Retorno_Historico_Anual': (1 + ret_hist_mensal) ** 12 - 1,
        'Retorno_CAPM_Mensal': ret_capm_mensal,
        'Retorno_CAPM_Anual': (1 + ret_capm_mensal) ** 12 - 1,
        'Alpha_Mensal': alpha_mensal,
        'Alpha_Anual': (1 + alpha_mensal) ** 12 - 1,
        'Retorno_Mercado_Medio_Mensal': df['mercado'].mean()
    })

df_beta_ativos = pd.DataFrame(resultados_beta)

# ----------------------------------------------------
# Beta e CAPM da carteira escolhida
# ----------------------------------------------------
df_carteira = pd.concat([retorno_carteira_escolhida, retorno_mercado], axis=1).dropna()
df_carteira.columns = ['carteira', 'mercado']

excesso_carteira = df_carteira['carteira'] - taxa_livre_risco_mensal
excesso_mercado_carteira = df_carteira['mercado'] - taxa_livre_risco_mensal

beta_carteira = excesso_carteira.cov(excesso_mercado_carteira) / excesso_mercado_carteira.var()

ret_hist_carteira_mensal = df_carteira['carteira'].mean()
ret_capm_carteira_mensal = (
    taxa_livre_risco_mensal
    + beta_carteira * (df_carteira['mercado'].mean() - taxa_livre_risco_mensal)
)
alpha_carteira_mensal = ret_hist_carteira_mensal - ret_capm_carteira_mensal

# Texto dinâmico com os pesos da carteira escolhida
pesos_escolhidos_dict = carteiras[carteira_escolhida_nome]
descricao_pesos = ", ".join(
    [f"{a}: {pesos_escolhidos_dict.get(a, 0):.0%}" for a in ativos]
)

# Resumo para a carteira
df_beta_carteira_resumo = pd.DataFrame([{
    'Carteira': f'{carteira_escolhida_nome} ({descricao_pesos})',
    'Beta': beta_carteira,
    'Alpha_Mensal': alpha_carteira_mensal,
    'Alpha_Anual': (1 + alpha_carteira_mensal) ** 12 - 1,
    'Retorno_Historico_Mensal': ret_hist_carteira_mensal,
    'Retorno_Historico_Anual': (1 + ret_hist_carteira_mensal) ** 12 - 1,
    'Retorno_CAPM_Mensal': ret_capm_carteira_mensal,
    'Retorno_CAPM_Anual': (1 + ret_capm_carteira_mensal) ** 12 - 1
}])

# Exibição
display(df_beta_ativos)
display(df_beta_carteira_resumo)

print(f"Taxa livre de risco anual: {taxa_livre_risco_anual:.2%}")
print(f"Taxa livre de risco mensal: {taxa_livre_risco_mensal:.4%}")
print(f"Beta da carteira principal: {beta_carteira:.4f}")
print(f"Alpha mensal da carteira principal: {alpha_carteira_mensal:.4f}")
print(f"Retorno CAPM mensal da carteira principal: {ret_capm_carteira_mensal:.4f}")

# ----------------------------------------------------
# 12) Gráficos
# ----------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

os.makedirs(pasta_saida, exist_ok=True)
pio.renderers.default = "colab"

# ----------------------------------------------------
# 12.0) Preparação da fronteira eficiente para muitos ativos
# ----------------------------------------------------
# 'ativos' deve existir antes deste bloco.
# Exemplo:
# ativos = ['BRAP4', 'POMO4', 'ITUB4', 'VALE3', 'PETR4', 'WEGE3', 'BBAS3', 'B3SA3', 'SUZB3', 'LREN3']

mu = retornos_mensais[ativos].mean().values
cov = retornos_mensais[ativos].cov().values

# Limite de pontos da fronteira
n_carteiras = 800
rng = np.random.default_rng(42)

# Carteiras aleatórias com soma = 1
pesos = rng.dirichlet(np.ones(len(ativos)), size=n_carteiras)

retornos = pesos @ mu
riscos = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))

df_fronteira = pd.DataFrame(pesos, columns=[f"Peso_{a}" for a in ativos])
df_fronteira["Retorno_Esperado_Mensal"] = retornos
df_fronteira["Risco_Mensal"] = riscos

# Carteira de mínima variância
idx_gmv = df_fronteira["Risco_Mensal"].idxmin()
carteira_gmv = df_fronteira.loc[idx_gmv].copy()

# Separar eficiente / ineficiente
df_fronteira = df_fronteira.sort_values(["Risco_Mensal", "Retorno_Esperado_Mensal"]).reset_index(drop=True)

df_fronteira["Eficiente"] = False
melhor_retorno = -np.inf

for i, row in df_fronteira.iterrows():
    if row["Retorno_Esperado_Mensal"] > melhor_retorno:
        df_fronteira.at[i, "Eficiente"] = True
        melhor_retorno = row["Retorno_Esperado_Mensal"]

df_eficiente = df_fronteira[df_fronteira["Eficiente"]].copy()
df_ineficiente = df_fronteira[~df_fronteira["Eficiente"]].copy()
df_eficiente = df_eficiente.sort_values(["Risco_Mensal", "Retorno_Esperado_Mensal"]).reset_index(drop=True)

# ----------------------------------------------------
# 12.0.1) Gráfico 3D único
# Eixo Z = peso de um ativo de destaque
# ----------------------------------------------------
ativo_destaque = ativos[0]
col_z = f"Peso_{ativo_destaque}"

fig3d = go.Figure()

fig3d.add_trace(go.Scatter3d(
    x=df_ineficiente["Retorno_Esperado_Mensal"],
    y=df_ineficiente["Risco_Mensal"],
    z=df_ineficiente[col_z],
    mode="markers",
    marker=dict(size=3, color="blue", opacity=0.20),
    name="Ineficientes"
))

fig3d.add_trace(go.Scatter3d(
    x=df_eficiente["Retorno_Esperado_Mensal"],
    y=df_eficiente["Risco_Mensal"],
    z=df_eficiente[col_z],
    mode="markers",
    marker=dict(size=4, color="red", opacity=0.75),
    name="Eficientes"
))

fig3d.add_trace(go.Scatter3d(
    x=df_eficiente["Retorno_Esperado_Mensal"],
    y=df_eficiente["Risco_Mensal"],
    z=df_eficiente[col_z],
    mode="lines",
    line=dict(color="red", width=5),
    name="Fronteira eficiente"
))

fig3d.add_trace(go.Scatter3d(
    x=[carteira_gmv["Retorno_Esperado_Mensal"]],
    y=[carteira_gmv["Risco_Mensal"]],
    z=[carteira_gmv[col_z]],
    mode="markers",
    marker=dict(size=11, color="green", symbol="x"),
    name="Mínima variância"
))

if "df_carteiras" in globals() and not df_carteiras.empty:
    for _, row in df_carteiras.iterrows():
        peso_z = row.get(col_z, np.nan)
        if pd.notna(peso_z):
            fig3d.add_trace(go.Scatter3d(
                x=[row["Retorno_Esperado_Mensal"]],
                y=[row["Risco_Mensal"]],
                z=[peso_z],
                mode="markers+text",
                marker=dict(size=13, color="#f39c12", symbol="diamond", line=dict(width=1, color="white")),
                text=[row["Carteira"]],
                textposition="top center",
                name=row["Carteira"]
            ))

fig3d.update_layout(
    title=f"Fronteira eficiente 3D com {len(ativos)} ativos",
    scene=dict(
        xaxis_title="Retorno esperado mensal",
        yaxis_title="Risco mensal",
        zaxis_title=f"Peso em {ativo_destaque}"
    ),
    width=1000,
    height=750,
    legend=dict(x=1.02, y=1)
)

fig3d.show()
fig3d.write_html(os.path.join(pasta_saida, "fronteira_eficiente_3D.html"))


# ----------------------------------------------------
# 12.1) Fronteira eficiente (2D risco vs retorno)
# ----------------------------------------------------
plt.figure(figsize=(9, 6))
plt.scatter(
    df_ineficiente['Risco_Mensal'],
    df_ineficiente['Retorno_Esperado_Mensal'],
    alpha=0.35,
    label='Ineficientes'
)
plt.scatter(
    df_eficiente['Risco_Mensal'],
    df_eficiente['Retorno_Esperado_Mensal'],
    alpha=0.85,
    label='Eficientes'
)
plt.scatter(
    carteira_gmv['Risco_Mensal'],
    carteira_gmv['Retorno_Esperado_Mensal'],
    s=120,
    marker='x',
    label='Mínima variância'
)
plt.scatter(
    df_carteira_escolhida['Risco_Mensal'].iloc[0],
    df_carteira_escolhida['Retorno_Historico_Mensal'].iloc[0],
    s=180,
    marker='D',
    label=f'Carteira escolhida ({carteira_escolhida_nome})'
)

plt.xlabel('Risco (desvio-padrão mensal)')
plt.ylabel('Retorno esperado mensal')
plt.title('Fronteira eficiente (risco vs retorno)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, 'fronteira_eficiente_2D.png'),
    dpi=200,
    bbox_inches='tight'
)
plt.show()



# ----------------------------------------------------
# Fronteira eficiente (2D)
# ----------------------------------------------------
plt.figure(figsize=(9,6))

plt.scatter(
    df_ineficiente["Risco_Mensal"],
    df_ineficiente["Retorno_Esperado_Mensal"],
    alpha=0.35,
    label="Ineficientes"
)

plt.scatter(
    df_eficiente["Risco_Mensal"],
    df_eficiente["Retorno_Esperado_Mensal"],
    alpha=0.85,
    label="Eficientes"
)

plt.plot(
    df_eficiente_ordenado["Risco_Mensal"],
    df_eficiente_ordenado["Retorno_Esperado_Mensal"],
    color="red",
    linewidth=3,
    label="Fronteira eficiente"
)

plt.scatter(
    carteira_gmv["Risco_Mensal"],
    carteira_gmv["Retorno_Esperado_Mensal"],
    s=120,
    marker="x",
    color="green",
    label="Mínima variância"
)

if "df_carteira_escolhida" in globals():

    plt.scatter(
        df_carteira_escolhida["Risco_Mensal"].iloc[0],
        df_carteira_escolhida["Retorno_Historico_Mensal"].iloc[0],
        s=180,
        marker="D",
        color="gold",
        edgecolors="darkorange",
        label=f"Carteira escolhida ({carteira_escolhida_nome})"
    )

plt.xlabel("Risco (desvio-padrão mensal)")
plt.ylabel("Retorno esperado mensal")
plt.title("Fronteira eficiente (risco vs retorno)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(pasta_saida, "fronteira_eficiente_2D.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()



# ----------------------------------------------------
# 12.2) Fronteira + perfis de carteiras (legenda externa)
# ----------------------------------------------------
plt.figure(figsize=(12, 6))

plt.scatter(
    df_ineficiente["Risco_Mensal"],
    df_ineficiente["Retorno_Esperado_Mensal"],
    c="blue", alpha=0.15, s=5, label="Ineficientes"
)

plt.scatter(
    df_eficiente["Risco_Mensal"],
    df_eficiente["Retorno_Esperado_Mensal"],
    c="red", alpha=0.5, s=12, label="Eficientes"
)

df_eficiente_ordenado = df_eficiente.sort_values("Retorno_Esperado_Mensal")
plt.plot(
    df_eficiente_ordenado["Risco_Mensal"],
    df_eficiente_ordenado["Retorno_Esperado_Mensal"],
    color="red", linewidth=4, linestyle="-", label="Fronteira eficiente"
)

plt.scatter(
    carteira_gmv["Risco_Mensal"],
    carteira_gmv["Retorno_Esperado_Mensal"],
    s=180, marker="x", color="green", linewidth=2, label="Mínima variância"
)

if "df_carteiras" in globals() and not df_carteiras.empty:
    palette = ["#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#16a085", "#8e44ad"]

    def descrever_pesos(row):
        return ", ".join([f"{a}: {row.get(f'Peso_{a}', 0):.0%}" for a in ativos])

    for i, (_, row) in enumerate(df_carteiras.iterrows()):
        cor = palette[i % len(palette)]
        plt.scatter(
            row["Risco_Mensal"],
            row["Retorno_Esperado_Mensal"],
            s=150,
            color=cor,
            edgecolors="white",
            linewidth=1.5,
            label=f'{row["Carteira"]} ({descrever_pesos(row)})',
            zorder=5
        )

    if "df_carteira_escolhida" in globals():
        escolhida = df_carteira_escolhida.iloc[0]
        plt.scatter(
            escolhida["Risco_Mensal"],
            escolhida["Retorno_Historico_Mensal"],
            s=300,
            color="gold",
            marker="D",
            edgecolors="darkorange",
            linewidth=2.5,
            label=f"⭐ Carteira Escolhida ({carteira_escolhida_nome})",
            zorder=10
        )

plt.xlabel("Risco (desvio-padrão mensal)")
plt.ylabel("Retorno esperado mensal")
plt.title("Fronteira eficiente com perfis de carteiras – legenda externa")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(
    os.path.join(pasta_saida, "fronteira_perfis_legenda_externa.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------
# 12.3) Garantias mínimas para a regressão beta
# ----------------------------------------------------
if "df_beta_carteira" not in globals():
    if "retorno_carteira_escolhida" in globals() and "retorno_mercado" in globals():
        df_beta_carteira = pd.concat(
            [retorno_carteira_escolhida, retorno_mercado],
            axis=1
        ).dropna()
        df_beta_carteira.columns = ["carteira", "mercado"]
    elif "df_carteira" in globals():
        df_beta_carteira = df_carteira.copy()
    else:
        raise NameError(
            "Não foi possível criar 'df_beta_carteira'. "
            "Verifique se 'retorno_carteira_escolhida' e 'retorno_mercado' foram criados antes."
        )

# ----------------------------------------------------
# 12.4) Regressão linear da carteira contra o mercado (beta)
# ----------------------------------------------------
x = df_beta_carteira["mercado"].values
y = df_beta_carteira["carteira"].values

beta_regressao, intercepto = np.polyfit(x, y, 1)
beta_carteira = beta_regressao
reta = np.poly1d([beta_regressao, intercepto])
x_linha = np.linspace(x.min(), x.max(), 200)
y_linha = reta(x_linha)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, label="Observações")
plt.plot(
    x_linha,
    y_linha,
    linewidth=2,
    label=f"Regressão linear (beta = {beta_regressao:.2f})"
)
plt.title("Regressão linear da carteira contra o mercado (beta)")
plt.xlabel("Retorno do mercado")
plt.ylabel("Retorno da carteira")
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, "regressao_carteira_contra_mercado.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------
# 12.5) Beta dos ativos e da carteira
# ----------------------------------------------------
betas = df_beta_ativos["Beta"].values
nomes = df_beta_ativos["Ativo"].tolist() + ["Carteira"]
valores_beta = list(betas) + [beta_carteira]

plt.figure(figsize=(8, 5))
barras = plt.barh(nomes, valores_beta)
plt.title("Beta dos ativos e da carteira")
plt.xlabel("Beta")
plt.ylabel("Ativo / Carteira")
plt.grid(True, axis="x", alpha=0.3)

for barra in barras:
    largura = barra.get_width()
    plt.text(
        largura,
        barra.get_y() + barra.get_height() / 2,
        f"{largura:.2f}",
        va="center",
        ha="left"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, "beta_ativos_e_carteira.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------
# 12.6) Retorno Histórico x CAPM
# ----------------------------------------------------
x = np.arange(len(df_beta_ativos))
largura = 0.35

plt.figure(figsize=(9, 5))
plt.bar(
    x - largura / 2,
    df_beta_ativos["Retorno_Historico_Anual"],
    width=largura,
    label="Histórico"
)
plt.bar(
    x + largura / 2,
    df_beta_ativos["Retorno_CAPM_Anual"],
    width=largura,
    label="CAPM"
)

plt.xticks(x, df_beta_ativos["Ativo"])
plt.title("Retorno Histórico x CAPM")
plt.ylabel("Retorno anual")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, "historico_vs_capm.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------
# 12.7) Retorno acumulado dos ativos
# ----------------------------------------------------
acumulado = (1 + retornos_mensais[ativos]).cumprod()

plt.figure(figsize=(10, 6))
for ativo in ativos:
    plt.plot(acumulado.index, acumulado[ativo], label=ativo)

plt.title("Retorno acumulado dos ativos")
plt.xlabel("Data")
plt.ylabel("Valor acumulado")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, "retorno_acumulado_ativos.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------
# 12.8) Risco dos ativos versus carteira
# ----------------------------------------------------
riscos_ativos = [retornos_mensais[ativo].std() for ativo in ativos]
nomes_risco = ativos + ["Carteira"]

if "df_carteira_escolhida" in globals():
    valores_risco = riscos_ativos + [df_carteira_escolhida["Risco_Mensal"].iloc[0]]
else:
    valores_risco = riscos_ativos

plt.figure(figsize=(8, 5))
barras = plt.bar(nomes_risco, valores_risco)
plt.title("Risco dos ativos versus carteira")
plt.ylabel("Desvio-padrão mensal")
plt.grid(True, axis="y", alpha=0.3)

for barra in barras:
    altura = barra.get_height()
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        altura,
        f"{altura:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(pasta_saida, "risco_ativos_vs_carteira.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.show()


# ----------------------------------------------------
# 13) Resumos interpretativos
# ----------------------------------------------------
resumo_economico = pd.DataFrame({
    'Tópico': [
        'Beta',
        'Risco total',
        'CAPM',
        'Fronteira eficiente',
        'Carteira escolhida'
    ],
    'Interpretação': [
        'Beta mede a sensibilidade do ativo em relação ao mercado. Beta acima de 1 indica maior sensibilidade às variações do mercado; abaixo de 1 indica menor sensibilidade.',
        'O risco total é a volatilidade dos retornos. Ele mostra o quanto o retorno oscila ao redor da média.',
        'O CAPM estima o retorno exigido pelo risco sistemático. Se o retorno histórico for maior que o CAPM, o ativo entregou desempenho acima do esperado pelo modelo.',
        'A fronteira eficiente reúne as combinações que oferecem maior retorno para cada nível de risco. Pontos abaixo dela são dominados e, portanto, ineficientes.',
        f'A carteira escolhida ({carteira_escolhida_nome}) representa a combinação destacada entre os {len(ativos)} ativos analisados. Ela deve ser interpretada em conjunto com a fronteira eficiente e com o perfil de risco desejado.'
    ]
})

# ----------------------------------------------------
# 14) Exibir tabelas
# ----------------------------------------------------
print("Preços mensais:")
display(precos_mensais.head())

print("Retornos mensais:")
display(retornos_mensais.head())

print("Análise individual dos ativos:")
display(df_analise_ativos)

print("Covariância:")
display(matriz_cov)

print("Correlação:")
display(matriz_corr)

print("Carteiras:")
display(df_carteiras)

print("Fronteira eficiente (pontos):")
display(df_fronteira.head())

print("Beta dos ativos:")
display(df_beta_ativos)

print("Beta da carteira:")
display(df_beta_carteira_resumo)

print("Resumo econômico:")
display(resumo_economico)

# ----------------------------------------------------
# 15) Salvar em Excel
# ----------------------------------------------------
arquivo_saida = os.path.join(pasta_saida, 'dados_mensais_CAPM_Varios_ativos.xlsx')

with pd.ExcelWriter(arquivo_saida, engine='openpyxl') as writer:
    precos_mensais.to_excel(writer, sheet_name='Precos_Mensais')
    retornos_mensais.to_excel(writer, sheet_name='Retornos_Mensais')
    df_analise_ativos.to_excel(writer, sheet_name='Analise_Ativos', index=False)
    matriz_cov.to_excel(writer, sheet_name='Covariancia')
    matriz_corr.to_excel(writer, sheet_name='Correlacao')
    df_carteiras.to_excel(writer, sheet_name='Carteiras', index=False)
    df_fronteira.to_excel(writer, sheet_name='Fronteira_Eficiente', index=False)
    df_eficiente.to_excel(writer, sheet_name='Fronteira_Pontos_Eficientes', index=False)
    df_ineficiente.to_excel(writer, sheet_name='Fronteira_Pontos_Ineficientes', index=False)
    df_beta_ativos.to_excel(writer, sheet_name='Beta_Ativos', index=False)
    df_beta_carteira_resumo.to_excel(writer, sheet_name='Beta_Carteira', index=False)
    df_carteira_escolhida.to_excel(writer, sheet_name='Carteira_Escolhida', index=False)
    resumo_economico.to_excel(writer, sheet_name='Resumo_Economico', index=False)

print(f'Arquivo salvo em: {arquivo_saida}')
