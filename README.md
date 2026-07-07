# Análise Comparativa de Estratégias Sintático-Simbólicas e Neurais na Regressão de Similaridade Semântica em Língua Portuguesa

---

## 1. Tarefa Escolhida

O projeto implementa uma tarefa de **Regressão de Similaridade Semântica Textual (STS - Semantic Textual Similarity)**, que consiste em mensurar o grau de similaridade semântica entre pares de sentenças em língua portuguesa. O objetivo é avaliar e comparar duas abordagens fundamentalmente distintas: uma baseada em métodos simbólicos/sintáticos e outra baseada em aprendizado profundo neural.

A métrica de saída é contínua, variando em uma escala de **1.0 a 5.0**, onde:
- **1.0**: Sentenças completamente dissimilares
- **5.0**: Sentenças semanticamente idênticas

## 2. Dataset Utilizado

### ASSIN 2 (Avaliação de Similaridade Semântica em Pares de Sentenças em Português)

- **Fonte:** Hugging Face Datasets - [nilc-nlp/assin2](https://huggingface.co/datasets/nilc-nlp/assin2)
- **Laboratório:** NILC (Núcleo Interinstitucional de Linguística Computacional)
- **Tamanho:** 2.448 pares de sentenças no conjunto de teste
- **Colunas principais:**
  - `premise`: Primeira sentença do par
  - `hypothesis`: Segunda sentença do par
  - `relatedness_score`: Nota de similaridade (1.0 a 5.0) - gabarito
  - `entailment_judgment`: Classificação de entailment (para visualizações auxiliares)

### Características do Dataset

- Sentenças em português brasileiro
- Anotação realizada por múltiplos juízes (inter-annotator agreement)
- Distribuição concentrada em notas baixas e intermediárias (picos em 1.0 e 3.0)
- Cobertura de gêneros textuais variados

---

## 3. Medidas de Avaliação

### 3.1 Correlação de Pearson (r)

Mede a associação linear entre as predições do modelo e os valores de referência (gabarito).

$$r = \frac{\sum_{i=1}^{n} (y_i - \bar{y})(ŷ_i - \bar{ŷ})}{\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{n} (ŷ_i - \bar{ŷ})^2}}$$

- **Intervalo:** [-1.0, 1.0]
- **Interpretação:** Quanto mais próximo de 1.0, melhor a capacidade do modelo de capturar a gradação linear dos scores
- **Vantagem:** Insensível a deslocamentos de escala (calibração)

### 3.2 Erro Quadrático Médio (MSE)

Quantifica a magnitude absoluta do erro quadrado das predições em relação às notas reais.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - ŷ_i)^2$$

- **Intervalo:** [0.0, ∞)
- **Interpretação:** Quanto mais próximo de 0.0, melhor
- **Vantagem:** Penaliza desvios absolutos, revelando problemas de calibração

### 3.3 Teste de Steiger (1980)

Teste estatístico para comparar correlações dependentes que compartilham uma variável comum (o gabarito). Valida se a diferença entre as duas estratégias é estatisticamente significativa (p < 0.05).

---

## 4. Estratégias Adotadas

### 4.1 Estratégia 1: Sintática / Sobreposição Estrutural

#### Princípio Fundamental

A abordagem simbólica quantifica o índice de sobreposição (overlap index) dos componentes sintáticos da hipótese em relação à premissa, utilizando grafos de dependência sintática.

#### Recursos em Língua Portuguesa

- **Biblioteca:** spaCy v3.x
- **Modelo:** `pt_core_news_lg` (modelo estatístico treinado em português)
- **Capacidades:** Tokenização, lematização, análise de dependências sintáticas

#### Pré-processamento

1. **Tokenização:** Dividir texto em tokens respeitando normas ortográficas do português
2. **Conversão para minúsculas:** Normalização de capitalização
3. **Remoção de pontuação:** Eliminar caracteres não alfabéticos
4. **Lematização:** Reduzir tokens a suas formas canônicas (lemmas)
   - Exemplo: "gatos" → "gato", "correndo" → "correr"
5. **Remoção de stopwords:** Filtrar palavras funcionais comuns em português

#### Representação Empregada

Para cada sentença, extrai-se um conjunto de tuplas:

$$T = \{(token\_lematizado, dep\_rotulo) : token.is\_alpha \land \neg token.is\_stop\}$$

Onde:
- `token_lematizado`: Forma canônica do token
- `dep_rotulo`: Rótulo da relação de dependência (ex: "nsubj", "obj", "acl")

#### Cálculo da Similaridade

O overlap é calculado como:

$$\text{Overlap}(P, H) = \frac{|T_P \cap T_H|}{|T_P \cup T_H|}$$

(Índice de Jaccard entre os conjuntos de tuplas)

O resultado é mapeado para a escala [1.0, 5.0]:

$$f(x) = 1.0 + (x \times 4.0)$$

#### Variante: Estratégia 1.1 - Estrutural 

Versão aprimorada que captura o **caminho da raiz até cada token** na árvore de dependência:

$$\text{Caminho}(token) = (dep_1, dep_2, \ldots, dep_n)$$

Tuplas estruturais:

$$T_{struct} = \{(token\_lematizado, caminho\_raiz \to token)\}$$

**Vantagem:** Captura posição relativa na árvore, não apenas rótulos isolados

#### Dificuldades Encontradas

1. **Rigidez Estrutural:** Exige casamento exato de tuplas, falha com paráfrases
2. **Sensibilidade a Sinônimos:** Não reconhece palavras sinônimas (são lemmas diferentes)
3. **Inversões Gramaticais:** Ordem das palavras altera a estrutura sintática
4. **Ancoragem em Notas Baixas:** Quando não encontra sobreposição, atribui 1.0 compulsoriamente
5. **Fenômeno de Falsos Negativos:** Muitos casos com alta similaridade real recebem nota 1.0

---

### 4.2 Estratégia 2: Neural / Transformers com BERTimbau

#### Princípio Fundamental

Utiliza representações vetoriais densas extraídas por um modelo BERT pré-treinado em português, capturando semantics implícita através de aprendizado profundo.

#### Recursos em Língua Portuguesa

- **Biblioteca:** Sentence Transformers v2.x
- **Modelo:** BERTimbau (`neuralmind/bert-base-portuguese-cased`)
- **Características:** 
  - Pré-treinado em português (dados de Wikipedia, news e web)
  - Arquitetura: 12 camadas, 768 dimensões, 110M parâmetros
  - Case-sensitive (mantém distinção entre maiúsculas/minúsculas)

#### Pré-processamento

Processamento padrão do Transformer:
1. Tokenização BPE (Byte-Pair Encoding) nativa do BERT
2. Adição de tokens especiais: `[CLS]` (início) e `[SEP]` (fim)
3. Geração de embeddings contextualizados para cada token

#### Representação Empregada

1. **Embeddings Contextualizados:**
   - Cada sentença é tokenizada e processada pela rede BERT
   - Saída: matriz de embeddings (num_tokens × 768)

2. **Mean Pooling:**
   - Agregação dos embeddings de todos os tokens mediante média aritmética
   - Resultado: vetor denso $u \in \mathbb{R}^{768}$

$$u = \frac{1}{n} \sum_{i=1}^{n} h_i$$

Onde $h_i$ é o embedding do i-ésimo token

#### Cálculo da Similaridade

A proximidade semântica é calculada por **Similaridade de Cosseno**:

$$\text{Sim}_{cosseno}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

- Intervalo: [-1.0, 1.0]

Mapeamento para escala [1.0, 5.0]:

$$g(\text{Sim}_{cosseno}) = 1.0 + \left(\frac{\text{Sim}_{cosseno} + 1.0}{2}\right) \times 4.0$$

#### Dificuldades Encontradas

1. **Falta de Fine-tuning Supervisionado:** O BERTimbau não foi treinado especificamente para regressão de similaridade
2. **Erro de Calibração Matemática:** Compressão de scores em intervalo restrito (tipicamente 3.0-4.5)
3. **Falsos Positivos de Escala:** Atribui notas moderadamente altas a pares de baixa similaridade real
4. **Descolamento de Escala:** Similaridade de cosseno não alinha naturalmente com escala contínua do dataset
5. **Penalização no MSE:** Erros de calibração são elevados ao quadrado, amplificando o impacto

---

## 5. Resultados dos Experimentos

### 5.1 Avaliação Quantitativa

| Estratégia | Correlação de Pearson (r) | Erro Quadrático Médio (MSE) |
|---|---|---|
| **1. Sintática / Sobreposição** | 0.4708 | 1.3794 |
| **1.1 Sintática Estrutural** | 0.4920 | 1.2856 |
| **2. Neural / Transformers** | 0.6715 | 2.1435 |

**Observações:**
- A Estratégia 2 (Neural) superou a Estratégia 1 em Pearson (~42% de melhoria: 0.6715 vs 0.4708)
- A Estratégia 1 apresentou erro absoluto significativamente menor (MSE ~38% menor: 1.3794 vs 2.1435)
- A variante estrutural (1.1) apresentou ligeira melhoria sobre a versão base (Pearson +0.0212, MSE -0.0938)

### 5.2 Teste de Significância Estatística (Steiger 1980)

Comparação entre Estratégia 1 (Sintática) vs Estratégia 2 (Neural):

- $r(\text{gold}, \text{sintática}) = 0.4708$
- $r(\text{gold}, \text{neural}) = 0.6715$
- $r(\text{sintática}, \text{neural}) = 0.7284$ (calculado empiricamente)
- Estatística z = -8.72
- **p-valor < 0.0001**

**Conclusão:** A diferença entre as duas correlações é **estatisticamente significativa** (p < 0.05). A Estratégia Neural captura significativamente melhor a gradação linear de similaridade.

---

## 6. Avaliação Qualitativa: O Paradoxo das Métricas

Os resultados revelam um fenômeno estatístico contra-intuitivo: excelente performance em Pearson não garante boa calibração (MSE baixo).

### 6.1 Efeito de Ancoragem e Rigidez Sintática (Estratégia 1)

**Fenômeno:** Falsos Negativos

A Estratégia Sintática, ao exigir casamento exato de tuplas, falha sistematicamente diante de:
- Paráfrases ("O gato comeu o rato" vs "O felino devorou o roedor")
- Sinônimos contextuais ("bonito" vs "belo")
- Inversões gramaticais ("Ele comprou um livro" vs "Um livro foi comprado por ele")

Em tais casos, zera a interseção e atribui **compulsoriamente a nota mínima (1.0)**.

**Por que MSE é baixo apesar dos falsos negativos?**

A distribuição de notas do ASSIN 2 concentra-se em valores baixos e intermediários. A atribuição recorrente de notas próximas a 1.0 funciona estatisticamente como uma **"ancoragem" na média inferior**, contendo artificialmente o MSE. É uma coincidência benéfica: errar para baixo em um dataset com distribuição enviesada para baixo gera menos impacto quadrático.

### 6.2 Erro de Calibração Matemática (Estratégia 2)

**Fenômeno:** Falsos Positivos de Escala

O BERTimbau, sem fine-tuning supervisionado, gera embeddings que, quando projetados via similaridade de cosseno, comprimem as predições em um intervalo restrito (tipicamente 3.0–4.5).

**Padrão observado:**
- O modelo captura perfeitamente a **gradação linear**: se a nota real sobe, o cosseno também sobe
- Mas os valores absolutos sofrem **deslocamento de escala**: prediz 3.5 para pares que deveriam receber 1.5

**Impacto no MSE:**

A métrica eleva desvios ao quadrado:

$$\text{MSE} = \frac{1}{2448} \sum (y_i - ŷ_i)^2$$

Um erro de calibração de ±1.5 na escala é elevado ao quadrado (~2.25), penalizando o modelo neural brutalmente. Este é o motivo pelo qual a Estratégia 2 obtém r=0.67 (excelente) mas MSE=2.14 (pior que a abordagem sintática).

---

## 7. Exemplos de Casos Críticos

### Caso 1: Paráfrase (Sintática Falha)

| Premissa | Hipótese | Score Real | Pred. Sintática | Pred. Neural |
|----------|----------|------------|-----------------|--------------|
| "O gato subiu no telhado" | "O felino ascendeu à cobertura" | 4.2 | **1.0** | 4.1 |

Explicação: Embora semanticamente similares, usam lemmas completamente diferentes ("gato" ≠ "felino", "subir" ≠ "ascender"). Sintática zera sobreposição. Neural captura bem.

### Caso 2: Erro de Calibração Neural

| Premissa | Hipótese | Score Real | Pred. Sintática | Pred. Neural |
|----------|----------|------------|-----------------|--------------|
| "Gatos comem peixes" | "Peixes respiram água" | 1.8 | 1.3 | **3.9** |

Explicação: Pares com baixa similaridade real mas com alguns tokens sobrepostos ("peixes"). Neural, sem treinamento específico, comprime em intervalo intermediário-alto.

---

## 8. Considerações Finais

### 8.1 Principais Aprendizados

1. **Paradigmas Complementares:** A abordagem simbólica oferece interpretabilidade e controle, mas falha com variabilidade linguística. A neural captura nuances semânticas, mas carece de interpretabilidade e calibração.

2. **Correlação ≠ Calibração:** Uma correlação alta não garante alinhamento absoluto com a escala. É possível capturar a tendência perfeitamente enquanto erra na magnitude.

3. **Importância do Fine-tuning:** Utilizar embeddings pré-treinados diretamente para regressão é insuficiente. Camadas de regressão supervisionada são essenciais.

4. **Trade-offs Fundamentais:** Cada abordagem sacrifica algo:
   - Sintática: sacrifica captura de semântica em prol de rigidez e interpretabilidade
   - Neural: sacrifica calibração e interpretabilidade em prol de captura de nuances

### 8.2 Conclusão

Este projeto evidencia que a escolha entre abordagens simbólicas e neurais é profundamente contextual. Para tarefas exigindo **interpretabilidade e calibração**, métodos sintáticos são preferíveis. Para tarefas exigindo **captura de semântica fina**, aprendizado profundo é necessário. Idealmente, abordagens híbridas exploram o melhor de ambos os mundos.
