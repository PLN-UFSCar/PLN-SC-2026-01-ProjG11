# -*- coding: utf-8 -*-
"""
  Colab usado ->  https://colab.research.google.com/drive/1MD0KBItoglQRpxUI6jq7HptxZQDPt0XM
"""

!pip install -q datasets sentence-transformers spacy scipy scikit-learn
!python -m spacy download pt_core_news_lg -q

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("nilc-nlp/assin2")

df_test = pd.DataFrame(dataset['test'])

print(f"Total de linhas no teste: {len(df_test)}")
df_test[['premise', 'hypothesis', 'relatedness_score']].head()

import spacy

nlp = spacy.load("pt_core_news_lg")

def extrair_tuplas_sintaticas(texto):
    '''
    Tokeniza, remove pontuação/tokens não-alfabéticos e stopwords,
    e retorna o conjunto de tuplas (lema, rótulo de dependência).
    '''
    doc = nlp(texto.lower())
    tuplas = set(
        (token.lemma_, token.dep_)
        for token in doc
        if token.is_alpha and not token.is_stop
    )
    return tuplas

def calcular_overlap_jaccard(premise, hypothesis):
    tuplas_p = extrair_tuplas_sintaticas(premise)
    tuplas_h = extrair_tuplas_sintaticas(hypothesis)

    uniao = tuplas_p.union(tuplas_h)
    if len(uniao) == 0:
        return 1.0

    intersecao = tuplas_p.intersection(tuplas_h)
    jaccard = len(intersecao) / len(uniao)

    score_mapeado = 1.0 + (jaccard * 4.0)
    return score_mapeado

df_test['pred_sintatica'] = df_test.apply(
    lambda row: calcular_overlap_jaccard(row['premise'], row['hypothesis']), axis=1
)

def caminho_raiz_token(token):
    caminho = []
    t = token
    while t.head != t:
        caminho.append(t.dep_)
        t = t.head
    caminho.append(t.dep_)
    return tuple(reversed(caminho))

def extrair_tuplas_estruturais(texto):
    doc = nlp(texto.lower())
    return set(
        (token.lemma_, caminho_raiz_token(token))
        for token in doc
        if token.is_alpha and not token.is_stop
    )

def calcular_overlap_estrutural(premise, hypothesis):
    tuplas_p = extrair_tuplas_estruturais(premise)
    tuplas_h = extrair_tuplas_estruturais(hypothesis)

    uniao = tuplas_p.union(tuplas_h)
    if len(uniao) == 0:
        return 1.0

    intersecao = tuplas_p.intersection(tuplas_h)
    jaccard = len(intersecao) / len(uniao)
    return 1.0 + (jaccard * 4.0)

df_test['pred_sintatica_estrutural'] = df_test.apply(
    lambda row: calcular_overlap_estrutural(row['premise'], row['hypothesis']), axis=1
)

import plotly.express as px
import plotly.graph_objects as go

fig_dist = px.histogram(
    df_test,
    x='pred_sintatica_estrutural',
    nbins=30,
    title='Distribuição do Overlap Sintático Estrutural',
    labels={'pred_sintatica_estrutural': 'Score Estrutural (1.0 a 5.0)'},
    template='plotly_white'
)
fig_dist.update_layout(
    yaxis_title='Frequência (Quantidade de Pares)',
    bargap=0.05
)
fig_dist.show()

coluna_label_real = 'entailment_judgment'

if coluna_label_real in df_test.columns:
    fig_box = px.box(
        df_test,
        x=coluna_label_real,
        y='pred_sintatica_estrutural',
        color=coluna_label_real,
        title='Capacidade de Separação do Score Estrutural por Classe Real',
        labels={
            'pred_sintatica_estrutural': 'Score Estrutural',
            coluna_label_real: 'Classe Real'
        },
        template='plotly_white'
    )
    fig_box.show()

from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

print("Gerando Embeddings e calculando as Similaridades...")
premises = df_test['premise'].tolist()
hypotheses = df_test['hypothesis'].tolist()

embeddings_p = model.encode(premises, convert_to_tensor=True)
embeddings_h = model.encode(hypotheses, convert_to_tensor=True)

cos_scores = util.cos_sim(embeddings_p, embeddings_h)
pred_cosseno = np.diag(cos_scores.cpu().numpy())

# Mapeia o cosseno (que varia de -1 a 1) para a escala de 1 a 5
df_test['pred_neural'] = 1.0 + ((pred_cosseno + 1.0) / 2.0) * 4.0
print("Concluído!")

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

y_true = df_test['relatedness_score'].values

# Estratégia 1
p_corr_sintatica, _ = pearsonr(y_true, df_test['pred_sintatica'].values)
mse_sintatica = mean_squared_error(y_true, df_test['pred_sintatica'].values)

# Estratégia 1.1
p_corr_estrutural, _ = pearsonr(y_true, df_test['pred_sintatica_estrutural'].values)
mse_estrutural = mean_squared_error(y_true, df_test['pred_sintatica_estrutural'].values)

# Estratégia 2
p_corr_neural, _ = pearsonr(y_true, df_test['pred_neural'].values)
mse_neural = mean_squared_error(y_true, df_test['pred_neural'].values)

print("\n" + "="*60)
print("               TABELA DE RESULTADOS FINAIS               ")
print("="*60)
print(f"{'Estratégia':<38} | {'Pearson (r)':<12} | {'MSE':<6}")
print("-"*60)
print(f"{'1. Sintática (lema+stopwords+Jaccard)':<38} | {p_corr_sintatica:<12.4f} | {mse_sintatica:<6.4f}")
print(f"{'1.1 Sintática estrutural (bônus)':<38} | {p_corr_estrutural:<12.4f} | {mse_estrutural:<6.4f}")
print(f"{'2. Neural (Transformers)':<38} | {p_corr_neural:<12.4f} | {mse_neural:<6.4f}")
print("="*60)
print("Nota: na Correlação de Pearson, quanto mais próximo de 1.0, melhor.")
print("Nota: no MSE, quanto mais próximo de 0.0, melhor.")

'''
--Descomente para visualizar os gráficos
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# 1. GRÁFICO DE LINHAS: Tendência e Calibração (Gabarito vs Predições)
df_sample = df_test.sample(50, random_state=42).sort_values(by='relatedness_score')

fig_linhas = go.Figure()
fig_linhas.add_trace(go.Scatter(y=df_sample['relatedness_score'], mode='lines+markers', name='Gabarito Real', line=dict(color='black', width=3)))
fig_linhas.add_trace(go.Scatter(y=df_sample['pred_neural'], mode='lines', name='Estratégia Neural', line=dict(color='blue', dash='dash')))
fig_linhas.add_trace(go.Scatter(y=df_sample['pred_sintatica'], mode='lines', name='Estratégia Sintática (corrigida)', line=dict(color='red', dash='dot')))

fig_linhas.update_layout(
    title='Comparação de Tendência e Alinhamento (Amostra de 50 Pares)',
    xaxis_title='Índice do Par (Ordenado pelo Gabarito)',
    yaxis_title='Relatedness Score (1 a 5)',
    template='plotly_white',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig_linhas.show()


# 2. GRÁFICO DE DISPERSÃO (Scatter Plot)
fig_scatter = make_subplots(rows=1, cols=2, subplot_titles=("Estratégia Sintática (corrigida)", "Estratégia Neural"))

fig_scatter.add_trace(
    go.Scatter(x=df_test['relatedness_score'], y=df_test['pred_sintatica'], mode='markers',
               marker=dict(color='red', opacity=0.4), name='Sintática'), row=1, col=1
)
fig_scatter.add_trace(
    go.Scatter(x=df_test['relatedness_score'], y=df_test['pred_neural'], mode='markers',
               marker=dict(color='blue', opacity=0.4), name='Neural'), row=1, col=2
)

for col in [1, 2]:
    fig_scatter.add_trace(go.Scatter(x=[1, 5], y=[1, 5], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False), row=1, col=col)

fig_scatter.update_layout(
    title='Dispersão das Predições em Relação ao Gabarito Ideal (Linha Cinza)',
    xaxis_title='Gabarito Real',
    yaxis_title='Predição',
    template='plotly_white',
    showlegend=False
)
fig_scatter.show()


# 3. HISTOGRAMA DE ERROS ABSOLUTOS
df_test['erro_sintatica'] = np.abs(df_test['relatedness_score'] - df_test['pred_sintatica'])
df_test['erro_neural'] = np.abs(df_test['relatedness_score'] - df_test['pred_neural'])

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=df_test['erro_sintatica'], name='Erro Sintática (corrigida)', marker_color='red', opacity=0.6))
fig_hist.add_trace(go.Histogram(x=df_test['erro_neural'], name='Erro Neural', marker_color='blue', opacity=0.6))

fig_hist.update_layout(
    title='Distribuição do Erro Absoluto',
    xaxis_title='Magnitude do Erro (Valor Absoluto)',
    yaxis_title='Frequência de Ocorrência',
    barmode='overlay',
    template='plotly_white'
)
fig_hist.show()'''