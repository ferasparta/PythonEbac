import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

caminho_arquivo = r'C:\Users\Vini\Downloads\ecommerce_estatistica.csv'
df = pd.read_csv(caminho_arquivo)

def limpar_qtd_vendidos(valor):
    valor = str(valor).replace('+', '')
    if 'mil' in valor:
        valor = valor.replace('mil', '')
        try:
            return float(valor) * 1000
        except ValueError:
            return np.nan
    else:
        try:
            return float(valor)
        except ValueError:
            return np.nan

df['Qtd_Vendidos'] = df['Qtd_Vendidos'].apply(limpar_qtd_vendidos)
df.dropna(subset=['Qtd_Vendidos'], inplace=True)

n_bins = 10
df['Preço_Bin'] = pd.cut(df['Preço'], bins=n_bins, include_lowest=True)
df['Qtd_Vendidos_Bin'] = pd.cut(df['Qtd_Vendidos'], bins=n_bins, include_lowest=True)
heatmap_data = pd.crosstab(df['Preço_Bin'], df['Qtd_Vendidos_Bin'])

# Gráfico de Histograma para 'Preço'
def create_price_histogram(dataframe):
    fig = px.histogram(dataframe, x='Preço', nbins=30, marginal="rug",
                       title='Histograma dos Preços')
    fig.update_layout(xaxis_title='Preço', yaxis_title='Frequência')
    return fig

# Gráfico de Dispersão para 'Qtd_Vendidos' vs 'Preço'
def create_scatter_plot(dataframe):
    fig = px.scatter(dataframe, x='Preço', y='Qtd_Vendidos',
                     title='Gráfico de Dispersão: Qtd_Vendidos vs Preço')
    fig.update_layout(xaxis_title='Preço', yaxis_title='Quantidade Vendida')
    return fig

# Mapa de Calor entre 'Preco' e 'Qtd_Vendidos'
def create_heatmap(dataframe):
    n_bins = 10
    df_temp = dataframe.copy() # Criar uma cópia para não modificar o DataFrame original
    df_temp['Preço_Bin'] = pd.cut(df_temp['Preço'], bins=n_bins, include_lowest=True)
    df_temp['Qtd_Vendidos_Bin'] = pd.cut(df_temp['Qtd_Vendidos'], bins=n_bins, include_lowest=True)
    heatmap_data = pd.crosstab(df_temp['Preço_Bin'], df_temp['Qtd_Vendidos_Bin'])

    fig = px.imshow(heatmap_data,
                    labels=dict(x=f"Quantidade Vendida (em {n_bins} intervalos)", y=f"Preço (em {n_bins} intervalos)", color="Contagem"),
                    x=heatmap_data.columns.astype(str),
                    y=heatmap_data.index.astype(str),
                    title='Mapa de Calor: Preço vs Quantidade Vendida (Contagem)')
    return fig

# Gráfico de Barras: Marca vs Média de 'Qtd_Vendidos_Cod'
def create_brand_sales_bar_chart(dataframe):
    vendas_por_marca = dataframe.groupby('Marca')['Qtd_Vendidos'].sum()
    marcas_com_min_vendas = vendas_por_marca[vendas_por_marca >= 100]
    marcas_com_min_vendas_ordenado = marcas_com_min_vendas.sort_values(ascending=False)

    fig = px.bar(marcas_com_min_vendas_ordenado,
                 x=marcas_com_min_vendas_ordenado.index,
                 y=marcas_com_min_vendas_ordenado.values,
                 title='Gráfico de Barras: Total de Vendas por Marca (Mínimo de 100 Vendas)')
    fig.update_layout(xaxis_title='Marca', yaxis_title='Total de Vendas')
    return fig

# Gráfico de Pizza: Comparação das Vendas por Marcas principais e outros
def create_pie_chart(dataframe):
    vendas_por_marca = dataframe.groupby('Marca')['Qtd_Vendidos'].sum()
    total_vendas = vendas_por_marca.sum()
    porcentagens = (vendas_por_marca / total_vendas) * 100

    marcas_pequenas = porcentagens[porcentagens < 2]
    vendas_outros = vendas_por_marca[marcas_pequenas.index].sum()
    vendas_principais = vendas_por_marca[porcentagens >= 2].copy()

    if vendas_outros > 0:
        vendas_principais['Outros'] = vendas_outros

    # Reordenar para que "Outros" fique por último se necessário, embora px.pie já faça isso bem
    vendas_principais = vendas_principais.sort_values(ascending=False)


    fig = px.pie(names=vendas_principais.index,
                 values=vendas_principais.values,
                 title='Gráfico de Pizza: Comparação das Vendas por Marca (Agrupando < 2% em "Outros")')
    fig.update_traces(textposition='inside', textinfo='percent+label') # Para mostrar % e label dentro das fatias
    return fig

# Gráfico de Densidade: Preço vs Qtd_Vendidos
def create_density_plot(dataframe):
    # sns.kdeplot não é diretamente plotly, então vamos usar plotly.graph_objects.Figure
    # ou plotly.express para uma aproximação.
    # plotly.express.density_heatmap é uma boa alternativa para kdeplot 2D
    fig = px.density_heatmap(dataframe, x='Preço', y='Qtd_Vendidos',
                             title='Gráfico de Densidade: Preço vs Quantidade Vendida')
    fig.update_layout(xaxis_title='Preço', yaxis_title='Quantidade Vendida')
    return fig

# Gráfico de Regressão: Marca vs Quantidade vendida
def create_regression_plot(dataframe):
    df_filtrado = dataframe[dataframe['Qtd_Vendidos'] >= 10000]
    fig = px.scatter(df_filtrado, x='Preço', y='Qtd_Vendidos', trendline="ols",
                     title='Gráfico de Regressão: Preço vs Quantidade Vendida (>= 10000 vendas)')
    fig.update_layout(xaxis_title='Preço', yaxis_title='Quantidade Vendida')
    return fig

# --- 3. Inicialização do Aplicativo Dash ---
app = dash.Dash(__name__)

# --- 4. Layout do Aplicativo ---
app.layout = html.Div(children=[
    html.H1(children='Análise de Dados de E-commerce', style={'textAlign': 'center'}),

    html.Div(children='Visualização Interativa dos Gráficos', style={'textAlign': 'center'}),

    html.Div(children=[
        dcc.Graph(
            id='histograma-preco',
            figure=create_price_histogram(df)
        ),
        dcc.Graph(
            id='dispersao-qtd-preco',
            figure=create_scatter_plot(df)
        ),
        dcc.Graph(
            id='heatmap-preco-qtd',
            figure=create_heatmap(df)
        ),
        dcc.Graph(
            id='barras-vendas-marca',
            figure=create_brand_sales_bar_chart(df)
        ),
        dcc.Graph(
            id='pizza-vendas-marca',
            figure=create_pie_chart(df)
        ),
        dcc.Graph(
            id='densidade-preco-qtd',
            figure=create_density_plot(df)
        ),
        dcc.Graph(
            id='regressao-preco-qtd',
            figure=create_regression_plot(df)
        )
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}), # Estilo para envolver os gráficos

    html.Hr(), # Linha divisória
    html.Div(children='Desenvolvido com Dash por [Seu Nome/Empresa]', style={'textAlign': 'center', 'fontSize': 12})
])

# --- 5. Execução do Aplicativo ---
if __name__ == '__main__':
    app.run(debug=True)