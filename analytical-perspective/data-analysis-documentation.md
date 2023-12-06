# Perspectiva Analítica

<p align="justify">
Este documento apresenta os resultados de uma análise de dados exploratória de sentimentos. A pesquisa visa compreender as complexidades emocionais em um conjunto de dados fornecido para esta atividade, utilizando técnicas de análise para proporcionar uma visão holística e detalhada. A análise de sentimentos desempenha um papel fundamental na compreensão da interação humana, sendo de grande importância em áreas como marketing, saúde mental e pesquisa social. Este relatóroio busca explorar, os sentimentos expressos em um conjunto de dados, a fim de extrair padrões, tendências e padrões relevantes.


## Índice
- [Análise Exploratória](#análise-exploratória)
- [Tratamento de Dados](#tratamento-de-dados)
- [Treinamento e Otimização de Algoritmo](#treinamento-e-otimização-de-algoritmo)
- [Análise de Importância de Variáveis e Interpretabilidade](#análise-de-importância-de-variáveis-e-interpretabilidade)
- [Validação dos Resultados e Performance do Algoritmo](#validação-dos-resultados-e-performance-do-algoritmo)

## Análise Exploratória

A análise exploratória de dados é uma fase crucial no processo de exploração e compreensão de conjuntos de dados. Trata-se de uma abordagem inicial que visa identificar padrões, tendências, relações e características fundamentais nos dados.

 
### Biblioteca
 
<p align="justify">
A condução desta pesquisa fundamenta-se na utilização da Natural Language Toolkit (NLTK) como a principal biblioteca para análise de sentimentos. A escolha da NLTK justifica-se pela sua ampla gama de recursos específicos para processamento de linguagem natural, oferecendo ferramentas robustas para tokenização, lematização, e análise semântica, elementos essenciais para uma análise aprofundada dos sentimentos presentes nos dados.

 
### Instalando e importando bibliotecas

```python
nltk.download('punkt')
nltk.download('stopwords')
```

O código acima é destinado ao download de recursos específicos da biblioteca Natural Language Toolkit (NLTK) no Python. `nltk.download('punkt')`: Esta linha de código faz o download do modelo de tokenização chamado "punkt" da NLTK. A tokenização é o processo de dividir um texto em unidades menores, chamadas de tokens. O modelo "punkt" é treinado para realizar a tokenização em diversos idiomas. Após executar essa linha, o modelo de tokenização "punkt" é baixado e disponível localmente para ser usado em análises de texto.
 
`nltk.download('stopwords')`: Esta linha faz o download de uma lista de palavras de parada, também conhecidas como "stop words". Stop words são palavras comuns que geralmente são removidas durante a pré-processamento de texto, pois não contribuem significativamente para a análise de sentimentos ou tópicos. Exemplos de stop words incluem artigos, preposições e algumas conjunções. Baixar a lista de stop words permite filtrar essas palavras irrelevantes durante o processamento de texto.



```python

import pandas as pd
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

```
 
`import pandas as pd`: Importa a biblioteca pandas e a renomeia para "pd". O pandas é amplamente utilizado para manipulação e análise de dados.

`import nltk`: Importa a biblioteca Natural Language Toolkit (NLTK), que fornece ferramentas para trabalhar com dados de linguagem natural, como tokenização, stemming, lematização, entre outros.

`import matplotlib.pyplot as plt`: Importa a biblioteca Matplotlib para visualização de dados e a renomeia para "plt". A Matplotlib é uma biblioteca popular para criar gráficos e visualizações em Python.

`import seaborn as sns`: Importa a biblioteca Seaborn, que é uma extensão visualmente atraente e informativa para o Matplotlib, facilitando a criação de gráficos estatísticos.

`from nltk.tokenize import word_tokenize`: Importa a função word_tokenize da biblioteca NLTK, que é usada para dividir um texto em palavras (tokenização de palavras).

`from nltk.corpus import stopwords`: Importa a lista de stopwords (palavras comuns que geralmente são removidas durante a análise de texto) da biblioteca NLTK.

`from nltk.probability import FreqDist`: Importa a classe FreqDist da biblioteca NLTK, que é usada para calcular a distribuição de frequência de palavras em um texto.

`from nltk.sentiment import SentimentIntensityAnalyzer`: Importa a classe SentimentIntensityAnalyzer da biblioteca NLTK, que é usada para realizar análise de sentimento em textos, atribuindo uma pontuação de polaridade aos textos.

### Visualizando dados iniciais

O código abaixo, está lendo um arquivo CSV chamado 'data.csv' e exibindo as primeiras linhas dos dados no DataFrame 'sentimentos'. Isso é útil para dar uma rápida olhada nos dados e entender sua estrutura.


```python
sentimentos = pd.read_csv('/work/data.csv')
sentimentos.head()
```

`sentimentos.head()` exibe a seguinte tabela com os dados:


|  | Sentence | Sentiment |
|----|---------------------------------------------------------------------------|-------------|
|  0 | The GeoSolutions technology will leverage Benefon 's GPS solutions...     | positive     | 
|  1 | $ESI on lows, down $1.50 to $2.50 BK a real possibility     | negative      | 
|  2 | For the last quarter of 2010 , Componenta 's net sales doubled to...  | positive |
|  3 | According to the Finnish-Russian Chamber of Commerce , all the... | neutral |
|  4 | The Swedish buyout firm has sold its remaining 22.4 percent... | neutral |

Sabendo da estrutura dos dados, segue-se para as informações gerais dos dados presentes. Para isso, usamos a função `.info()`.

```python
sentimentos.info()
```
Saída:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5842 entries, 0 to 5841
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   Sentence   5842 non-null   object
 1   Sentiment  5842 non-null   object
dtypes: object(2)
memory usage: 91.4+ KB
```
A saída mostra que o DataFrame tem duas colunas chamadas "Sentence" e "Sentiment", ambas contendo 5842 entradas não nulas do tipo objeto. O índice vai de 0 a 5841, e o DataFrame está consumindo aproximadamente 91.4 KB de memória. Em seguida, foi utilizado a função `.value_counts()` para avaliar a quantide de cada sentimento presente na base de dados. 


```python
sentimentos['Sentiment'].value_counts()
```
A saída dessa função forneceu a quantidade de sentimentos identificados na base de dados. Observa-se que comentários neutros são a maoioria (53,58%).

```
neutral     3130
positive    1852
negative     860
```
O gráfico abaixo, possui uma representação gráfica da quantidade dos três sentimentos identificados.

![grafico 1](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/c13eb3ac-56d2-4b8f-aabd-faf515ae8296)


## Tratamento de Dados

O tratamento de dados refere-se ao conjunto de processos e técnicas aplicadas para manipular, organizar e preparar dados de forma a torná-los mais úteis e acessíveis para análises e tomada de decisões.

### Remoção das Stop Words

O objetivo é limpar e processar as sentenças contidas na coluna 'Sentence' desse conjunto de dados.

```python
stop_words = set(stopwords.words('english'))
```

Aqui, o código está utilizando a biblioteca nltk para trabalhar com o idioma inglês. Especificamente, está criando um conjunto (set) de palavras de parada (stop words) em inglês. Stop words são palavras comuns que geralmente são removidas em etapas de pré-processamento, pois não contribuem significativamente para a compreensão do significado de uma frase.

```python
def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return filtered_words
```
Esta função recebe um texto como entrada e realiza as seguintes etapas:
- `word_tokenize(text)`: Divide o texto em palavras individuais. Isso é feito usando a função `word_tokenize` da biblioteca nltk.
- `[word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]`: Cria uma lista de palavras filtradas. Para cada palavra no texto, converte-a para minúsculas `(word.lower())`, verifica se é alfanumérica `(word.isalnum())`, e se não está na lista de stop words `(word.lower() not in stop_words)`. A lista resultante contém apenas as palavras relevantes após o pré-processamento.

### Distribuição da Quantidade de Palavras por Sentença

Em seguida, realizamos a visualização da distribuição da quantidade de palavras por sentença do conjunto de dados. Primeiro, é definido o nome da coluna que contém a quantidade de palavras por sentença. No código, ela é referenciada como qtd_palavras.

```python
word_count_column = 'qtd_palavras'
```

É criado um gráfico com duas subplots (duas áreas de plotagem) lado a lado. O primeiro subplot (ax1) é usado para o histograma, e o segundo subplot (ax2) é usado para o boxplot.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})
```

Em seguida, é plotado o histograma da quantidade de palavras por sentença. O eixo x representa a quantidade de palavras, o eixo y representa o número de sentenças com base na quantidade de palavras. O histograma é dividido em 20 bins, e a cor de preenchimento é 'skyblue' com bordas pretas.

```python
ax1.hist(sentimentos[word_count_column], bins=20, color='skyblue', edgecolor='black')
ax1.set_xlabel('Quantidade de Palavras por Sentença')
ax1.set_ylabel('Número de Sentenças')
ax1.set_title('Distribuição da Quantidade de Palavras por Sentença')
```

Duas linhas verticais são adicionadas ao histograma para representar a média (em vermelho) e a mediana (em verde) da quantidade de palavras por sentença. As linhas são pontilhadas e têm rótulos indicando os valores.

```python
mean_value = sentimentos[word_count_column].mean()
median_value = sentimentos[word_count_column].median()
ax1.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Média ({mean_value:.2f})')
ax1.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Mediana ({median_value:.2f})')
ax1.legend()
```

No segundo subplot (ax2), é plotado o boxplot da quantidade de palavras por sentença. O retângulo do boxplot é preenchido com a cor 'skyblue'. As etiquetas no eixo x são removidas, e rótulos são adicionados para os eixos y e título.

```python
boxplot = ax2.boxplot(sentimentos[word_count_column], vert=True, patch_artist=True)
for patch in boxplot['boxes']:
    patch.set_facecolor('skyblue')
ax2.set_xticklabels('')
ax2.set_ylabel('Quantidade de Palavras por Sentença')
ax2.set_title('Boxplot da Quantidade de Palavras por Sentença')
```

Finalmente, o gráfico completo é exibido com a função `plt.show()`. A figura inclui tanto o histograma quanto o boxplot, proporcionando uma visão abrangente da distribuição da quantidade de palavras por sentença no conjunto de dados.

![grafico 2](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/c0e74e2f-b483-4c04-89b1-25f33bcfdb1b)

O gráfico mostra a distribuição da quantidade de palavras por sentença em um conjunto de dados de 30 sentenças. O histograma mostra que a maioria das sentenças (cerca de 60%) tem entre 10 e 15 palavras. O boxplot mostra que a mediana da quantidade de palavras por sentença é de 10 palavras, com um intervalo interquartílico de 5 palavras. Isso significa que 50% das sentenças têm entre 5 e 15 palavras.

Em geral, o gráfico mostra que as sentenças neste conjunto de dados são relativamente curtas, com uma média de 11,6 palavras. No entanto, há uma variação significativa na quantidade de palavras por sentença, com algumas sentenças tendo até 40 palavras.

### As 15 palavras que mais se repetemm

Essa seção tem como objetivo plotar um gráfico de barras das palavras mais comuns, após os processo de stop words.

Primeiro, cria-se uma lista chamada `all_words` que contém todas as palavras da coluna especificada `(column_name)` do DataFrame `df`. As palavras são convertidas para minúsculas.
Em seguida, é criada uma distribuição de frequência `(freq_dist)` usando a classe `FreqDist` do módulo `nltk`, que conta a frequência de cada palavra na lista.

```python
all_words = [word.lower() for words in df[column_name] for word in words]
freq_dist = FreqDist(all_words)
```

Em seguida, a função `most_common` é usada para obter as `num_words` palavras mais comuns, juntamente com suas frequências, a partir da distribuição de frequência.

```python
most_common_words = freq_dist.most_common(num_words)
```

Um gráfico de barras é criado usando a biblioteca `matplotlib`. As barras são criadas usando as palavras mais comuns e suas frequências.

```python
plt.figure(figsize=(18, 6))
bars = plt.bar(*zip(*most_common_words))
```

Rótulos são adicionados às barras indicando a frequência de cada palavra. Rótulos e título são adicionados ao gráfico, indicando o eixo x (Palavras), o eixo y (Frequência) e o título do gráfico. Finalmente, o gráfico é exibido. A função é então utilizada plotando as 15 palavras mais comuns da coluna 'texto_processado' do DataFrame 'sentimentos'.

```python
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), ha='center', va='bottom')

plt.xlabel('Palavras')
plt.ylabel('Frequência')
plt.title(f'Top {num_words} Palavras Mais Repetidas')
plt.show()
```

![grafico 3](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/52e7dc8d-c415-4027-8125-f73ee1658a95)

O gráfico mostra as 15 palavras mais repetidas em um conjunto de dados de texto. As palavras são representadas por seus tamanhos, com as palavras mais repetidas sendo maiores. As três palavras mais repetidas são "eur", "company" e "mn". Isso sugere que o conjunto de dados é composto principalmente de texto relacionado a empresas e finanças. As palavras "profit" e "sales" também aparecem com frequência, o que sugere que o conjunto de dados pode incluir informações sobre o desempenho financeiro das empresas.

Outras palavras que aparecem com frequência incluem "finnish", "said", "net", "million", "operating", "2009", "min", "2008" e "period". Essas palavras podem fornecer mais informações sobre o contexto do conjunto de dados, como o idioma, o período de tempo ou o setor específico a que se refere.

Para concluir o tratamento de dados, foi realizado uma média de pontuação de sentimento. A média deste conjunto de dados foi de 0.15049840807942486.

```python
sia = SentimentIntensityAnalyzer()
sentimentos['sentiment_score'] = sentimentos['Sentence'].apply(lambda x: sia.polarity_scores(x)['compound'])
print("Média de pontuação de sentimento:", sentimentos['sentiment_score'].mean())
```

## Treinamento e Otimização de Algoritmo

Seleção, Treinamento e Otimização de Algoritmo são etapas fundamentais no desenvolvimento de modelos de aprendizado de máquina (machine learning) e algoritmos em geral. A seleção de algoritmo refere-se à escolha do modelo ou algoritmo mais apropriado para resolver um determinado problema de aprendizado de máquina. O treinamento de algoritmo é o processo pelo qual um modelo é ajustado aos dados de treinamento para aprender padrões e fazer previsões ou tomar decisões. A otimização de algoritmo refere-se ao aprimoramento do desempenho do modelo, seja em termos de precisão, eficiência computacional ou outras métricas relevantes.

Essas etapas formam um ciclo iterativo, já que a escolha do algoritmo inicial pode ser ajustada com base nos resultados obtidos durante o treinamento e na otimização. O sucesso do processo depende da compreensão profunda do problema, da escolha adequada de algoritmos e da aplicação de técnicas eficazes de treinamento e otimização. Para essa etapa vamos precisar de importar novas bibliotecas.

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
```

É importado a biblioteca `NumPy` com o alias `np`. NumPy é amplamente utilizado para operações numéricas em Python, especialmente para manipulação de arrays. Também é importado algumas funções e classes relacionadas à validação cruzada e divisão de conjuntos de dados:
- `train_test_split` é usado para dividir o conjunto de dados em treino e teste;
- `cross_val_score` é usado para realizar validação cruzada;
- `cross_val_predict` é usado para obter previsões durante a validação cruzada.

A classe `TfidfVectorizer` que é importada e utilizada para converter uma coleção de documentos de texto em uma matriz de recursos TF-IDF. TF-IDF (Term Frequency-Inverse Document Frequency) é uma técnica comum para representar texto como um vetor numérico. Vários classificadores de machine learning disponíveis no `scikit-learn`, como Naive Bayes, SVM, Random Forest, Gradient Boosting, Regressão Logística, entre outros são importados. Esses serão usados para treinar modelos de classificação.

Métricas de avaliação de desempenho de modelos, como `accuracy_score` (acurácia), `classification_report` (relatório de classificação) e `confusion_matrix` (matriz de confusão) são chamadas para o código. Por fim, importa a função `learning_curve` que é utilizada para visualizar o desempenho do modelo ao longo do tempo de treinamento, o que pode ser útil para identificar problemas de overfitting ou underfitting.

### Lematização

A lematização é um processo linguístico utilizado na análise morfológica de palavras, visando encontrar a forma base ou lema de uma palavra em um determinado idioma. O lema é a forma canônica da palavra que representa sua raiz ou forma fundamental. Esse processo é fundamental em linguística computacional, processamento de linguagem natural e outras aplicações relacionadas ao processamento de texto. O objetivo da lematização é reduzir palavras flexionadas ou derivadas a sua forma base, a fim de simplificar a análise e compreensão do texto. 

Para isso, importamos a classe `WordNetLemmatizer` da biblioteca NLTK, que será usada para realizar a lematização. Também é realizado o download de recursos necessários para a lematização, especificamente o banco de dados WordNet (usado pelo lematizador) e a base de dados multilíngue Open Multilingual Wordnet (omw).

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
```

É definido a função `preprocess_dataframe`. Esta função aceita um DataFrame `df`, o nome da coluna de texto `text_column` e um nome opcional para a nova coluna que armazenará o texto pré-processado `new_column_name`. Após, um objeto WordNetLemmatizer é criado para realizar a lematização, e um conjunto de palavras de parada `stop words` em inglês é criado usando a biblioteca NLTK.

```python
def preprocess_dataframe(df, text_column, new_column_name='preprocessed_text'):
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```

A lematização é aplicada a cada sentença na coluna de texto especificada `text_column`. A função `apply` é usada para aplicar a lematização a cada linha. A expressão `lambda` dentro do `apply` percorre cada palavra da sentença, lematiza as palavras, converte para minúsculas, e verifica se a palavra é alfanumérica e não está na lista de stop words. O resultado é uma nova coluna no DataFrame chamada `new_column_name` que contém o texto pré-processado. A função retorna o DataFrame modificado com a nova coluna de texto pré-processado.

```python
df[new_column_name] = df[text_column].apply(lambda sentence: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]))

return df
```

Saída:

![288532829-232a3ac2-97c6-4083-958e-d3b6c8c09edc](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/3d9e39e4-198f-4889-80cc-14fd2d022593)


### Testando 8 Algoritmos

Uma comparação de desempenho entre oito algoritmos de aprendizado de máquina usando validação cruzada para determinar qual modelo tem a melhor acurácia média é realizada. Os dados são divididos em conjuntos de treinamento (X_train, y_train) e teste (X_test, y_test) usando a função train_test_split do scikit-learn. Os dados consistem em textos processados (texto_processado2) e rótulos de sentimento.

```python
X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.3, random_state=42)
```

O texto é vetorizado usando o `TfidfVectorizer` para converter os textos em representações numéricas, levando em consideração a frequência dos termos nos documentos. O parâmetro `max_features` limita o número máximo de palavras consideradas.

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)
```

São criados vários modelos de classificação são inicializados, incluindo Naive Bayes, SVM, Random Forest, Gradient Boosting, Regressão Logística, Ridge Classifier, K-Nearest Neighbors e Decision Tree.

```python
models = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Ridge Classifier': RidgeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}
```

Um loop é utilizado para iterar sobre cada modelo. A função `cross_val_score` é usada para calcular as acurácias médias dos modelos usando validação cruzada (`cv=5` indica 5 folds). As acurácias médias e desvios padrão são impressos na tela.

```python
model_accuracies = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_features, y_train, cv=5, scoring=make_scorer(accuracy_score))
    model_accuracies[model_name] = cv_scores.mean()
    print(f'{model_name}: Acurácia média na validação cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')
```

O modelo com a maior acurácia média na validação cruzada é escolhido e a instância desse modelo é armazenada na variável `best_model_instance`. As acurácias médias de cada modelo são impressas no final do código.

```python
best_model = max(model_accuracies, key=model_accuracies.get)
best_model_instance = models[best_model]

print("\nAcurácias de cada modelo:")
for model_name, accuracy in model_accuracies.items():
    print(f'{model_name}: {accuracy:.2f}')

```

A saída consiste em:

```
Naive Bayes: Acurácia média na validação cruzada: 0.66 (+/- 0.01)
Support Vector Machine: Acurácia média na validação cruzada: 0.65 (+/- 0.02)
Random Forest: Acurácia média na validação cruzada: 0.65 (+/- 0.03)
Gradient Boosting: Acurácia média na validação cruzada: 0.65 (+/- 0.01)
Logistic Regression: Acurácia média na validação cruzada: 0.67 (+/- 0.02)
Ridge Classifier: Acurácia média na validação cruzada: 0.67 (+/- 0.01)
K-Nearest Neighbors: Acurácia média na validação cruzada: 0.61 (+/- 0.01)
Decision Tree: Acurácia média na validação cruzada: 0.59 (+/- 0.03)

Acurácias de cada modelo:
Naive Bayes: 0.66
Support Vector Machine: 0.65
Random Forest: 0.65
Gradient Boosting: 0.65
Logistic Regression: 0.67
Ridge Classifier: 0.67
K-Nearest Neighbors: 0.61
Decision Tree: 0.59

```
### Treinamento

Para o treinamento, foi adicionada uma nova celula.

```python
best_model_instance.fit(X_train_features, y_train)

X_test_features = vectorizer.transform(X_test)

predictions = best_model_instance.predict(X_test_features)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'\nMelhor modelo: {best_model}')
print(f'Acurácia nos dados de teste: {accuracy:.2f}')
print('Matriz de Confusão:')
print(conf_matrix)
print('Relatório de Classificação:')
print(classification_rep)
```

- `best_model_instance.fit(X_train_features, y_train)`: Esta linha está treinando o modelo utilizando os dados de treino `(X_train_features)` e os rótulos correspondentes `(y_train)`. `best_model_instance` provavelmente contém um modelo de machine learning previamente ajustado ou otimizado.
- `X_test_features = vectorizer.transform(X_test)`: Aqui, os dados de teste `(X_test)` estão sendo vetorizados usando um vectorizer (um objeto que transforma os dados de texto em uma representação numérica, geralmente usado em processamento de linguagem natural).
- `predictions = best_model_instance.predict(X_test_features)`: Este trecho faz previsões nos dados de teste usando o modelo treinado. As previsões são armazenadas na variável predictions.
- `accuracy = accuracy_score(y_test, predictions)`: A acurácia do modelo nos dados de teste está sendo calculada usando a função `accuracy_score` do `scikit-learn`, comparando as previsões `(predictions)` com os rótulos reais `(y_test)`.
- `conf_matrix = confusion_matrix(y_test, predictions)`: Uma matriz de confusão está sendo calculada para avaliar o desempenho do modelo em termos de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
- `classification_rep = classification_report(y_test, predictions)`: Um relatório de classificação é gerado, fornecendo métricas de precisão, recall, pontuação F1 e outras métricas para cada classe no conjunto de dados de teste.
- A última parte do código imprime os resultados, incluindo o melhor modelo utilizado, a acurácia nos dados de teste, a matriz de confusão e o relatório de classificação.


## Análise de Importância de Variáveis e Interpretabilidade
 
A análise de importância de variáveis e interpretabilidade desempenha um papel crucial na compreensão e confiança em modelos estatísticos. A tarefa de treinar um modelo de Regressão Logística para classificar sentimentos com base em um conjunto de dados chamado `sentimentos2`. 

Os dados são divididos em conjuntos de treinamento e teste usando a função `train_test_split` do `scikit-learn`. 80% dos dados são usados para treinamento `(X_train, y_train)`, e 20% são reservados para teste `(X_test, y_test)`.

```python
X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.2, random_state=42)
```

A técnica TF-IDF (Term Frequency-Inverse Document Frequency) é aplicada aos textos usando o `TfidfVectorizer` do `scikit-learn`. Isso transforma os textos em uma matriz numérica onde cada linha representa um documento (no caso, um texto processado) e cada coluna representa uma palavra única presente nos documentos.

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)
```

Um modelo de Regressão Logística é inicializado com a opção `class_weight='balanced`, o que significa que ele tentará lidar automaticamente com desequilíbrios nas classes. O solver utilizado é `newton-cg` e a estratégia de tratamento de múltiplas classes é multinomial. O modelo é treinado usando as características extraídas dos textos de treinamento `(X_train_features)` e os rótulos correspondentes `(y_train)`.

```python
logistic_regression_model = LogisticRegression(class_weight='balanced',solver='newton-cg',multi_class = 'multinomial')
logistic_regression_model.fit(X_train_features, y_train)
```

Os textos de teste são vetorizados usando o mesmo vetorizador que foi ajustado nos dados de treinamento. O modelo treinado é então usado para fazer previsões sobre os dados de teste.

```python
X_test_features = vectorizer.transform(X_test)
predictions = logistic_regression_model.predict(X_test_features)
```

Os nomes das características (palavras) e os coeficientes associados a cada classe são extraídos do modelo treinado.

```python
feature_names = vectorizer.get_feature_names_out()
coeficients_per_class = logistic_regression_model.coef_
```

Um DataFrame é criado para visualizar os coeficientes associados a cada classe para cada palavra. Os DataFrames para cada classe são concatenados em um DataFrame final `(final_coef_df)`. Os coeficientes são ordenados por magnitude em ordem decrescente, e os 10 principais coeficientes são exibidos.

```python
coef_dfs = []
for i, class_coeficients in enumerate(coeficients_per_class):
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': class_coeficients})
    coef_df['Class'] = f'Class_{i}'
    coef_dfs.append(coef_df)

final_coef_df = pd.concat(coef_dfs)

sorted_coef_df = final_coef_df.sort_values(by='Coefficient', ascending=False)
sorted_coef_df.head(10)
```

Saída:

![288562479-6107a19d-4484-48ea-b5ca-a95d8c9ceb68](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/8ebd16c0-81f4-4633-b740-b8f680c0dd51)


## Validação dos Resultados e Performance do Algoritmo

A validação dos resultados e a avaliação da performance de algoritmos são etapas cruciais no desenvolvimento e implementação de soluções em ciência de dados e aprendizado de máquina. Esses processos garantem que os modelos construídos sejam robustos, confiáveis e capazes de generalizar para novos dados. A função `plot_learning_curve_with_metrics` é uma função para visualizar a curva de aprendizado e avaliar o desempenho do modelo treinado usando métricas específicas. 

No trecho abaixo, podemos observar que `learning_curve` é uma função do `scikit-learn` que gera pontuações de treinamento e validação para diferentes tamanhos de conjunto de treinamento. `train_sizes` especifica os tamanhos relativos do conjunto de treinamento. `train_scores` contém as pontuações de treinamento para cada tamanho do conjunto de treinamento. `validation_scores` contém as pontuações de validação cruzada para cada tamanho do conjunto de treinamento.

```python
def plot_learning_curve_with_metrics(estimator, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
  
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring)
```

Em seguida, é calculado as médias e os desvios padrão das pontuações de treinamento e validação para posterior plotagem.

```python
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)
```

Em seguida, acontece a plotagem da curva de aprendizado com intervalos de confiança para as pontuações de treinamento e validação.

```python
  plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="orange", label="Cross-validation Score")

    plt.title("Curvas de aprendizdo")
    plt.xlabel("Amostras de Treino")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
```

Após, realiza o treinamento do modelo, gera previsões no conjunto de teste, imprime o relatório de classificação, exibe a matriz de confusão e imprime a acurácia do modelo.

```python
plt.subplot(2, 2, 2)
estimator.fit(X, y)
y_pred = estimator.predict(X)
print("Classification Report:")
print(classification_report(y, y_pred))
conf_matrix = confusion_matrix(y, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matriz de confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
plt.tight_layout()
plt.show()

```

Para o uso da função, é dividido o conjunto de dados em treino e teste, realiza a vetorização usando TfidfVectorizer e, em seguida, chama a função plot_learning_curve_with_metrics com o modelo de regressão logística, as características de treinamento e os rótulos de treinamento. A validação cruzada é configurada para 5 folds, e a métrica de avaliação é a acurácia.


```python
X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)
plot_learning_curve_with_metrics(logistic_regression_model, X_train_features, y_train, cv=5, scoring='accuracy')
```

A saída que obtemos e os gráficos plotados estão anexados a seguir.

```
Classification Report:
              precision    recall  f1-score   support

    negative       0.58      0.96      0.73       685
     neutral       0.95      0.79      0.86      2508
    positive       0.91      0.88      0.90      1480

    accuracy                           0.85      4673
   macro avg       0.81      0.88      0.83      4673
weighted avg       0.88      0.85      0.85      4673


Accuracy: 0.85
```

![grafico 4](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/84748508/917c89bd-34dd-400b-83cb-4d7d7287e7b3)


O gráfico mostra as curvas de aprendizado do modelo de classificação, com o score de treinamento (linha azul) e o score de validação cruzada (linha amarela). O score de treinamento aumenta com o aumento do número de amostras de treinamento, enquanto o score de validação cruzada atinge um platô após cerca de 2.500 amostras. Isso indica que o modelo está superajustando os dados de treinamento.

A matriz de confusão mostra os resultados do modelo para as três classes: negativo, neutro e positivo. O modelo teve um desempenho geral bom, com uma acurácia de 0,80. No entanto, houve alguns erros importantes. O modelo classificou 19 amostras negativas como positivas e 126 amostras positivas como neutras. Isso pode ser um problema para aplicações em que a precisão é crítica, como diagnóstico médico ou análise de crédito.
