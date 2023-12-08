# API documentation
---
<p align="justify"> Descrição: Esta API utiliza um modelo de Regressão Logística treinado para classificar o sentimento associado a comentários de texto. O modelo avalia se um comentário é positivo, negativo ou neutro com base em padrões aprendidos durante o treinamento</p>

----

  #### 🎯 | Objetivos
  <ul>
        <li>Apresentar as bibliotecas utilizadas</li>
        <li>Como a API funciona e outros</li>
  </ul>

 ----

#### 📚 | Componentes do Código

 ```python
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
```

 * `from flask import Flask`: Flask é um framework web leve para Python. Ele facilita a criação de aplicativos web e APIs com facilidade e simplicidade. Para poder usar o flask é necessário realizar sua instalação `pip install Flask`

* `from flask import request`: A biblioteca *request* faz parte do Flask e é usada para acessar os dados das requisições HTTP feitas para a sua aplicação. Ela fornece métodos para interagir com os dados do cliente, como parâmetros de consulta (query string), dados de formulário, e corpo da requisição em JSON.
* `from jsonify`: É uma função utilitária fornecida pelo Flask para simplificar a criação de respostas HTTP com dados no formato JSON. Ela converte objetos Python em JSON e configura a resposta HTTP com o cabeçalho Content-Type apropriado.
* `from flask_cors import CORS`: Flask-CORS é uma extensão do Flask que lida com as questões de Cross-Origin Resource Sharing (CORS). Ele permite que sua API seja acessada por diferentes domínios. Sendo também necessária a sua instalação através do `pip install flask-cors`
* `import joblib`: Joblib é uma biblioteca para processamento paralelo em Python, usada aqui para carregar modelos treinados a partir de arquivos.
* `from sklearn.feature_extraction.text import TfidVectorizer`: TfidfVectorizer é usado para vetorizar o texto de entrada antes de alimentá-lo ao modelo já treinado.

````python
  model = joblib.load('modelo_treinado.joblib')
  vectorizer = joblib.load('vectorizer.joblib')

  app = Flask(__name__)
  CORS(app) 
  @app.route('/', methods=['POST'])
````
 As variáveis `model` e `vectorizer` são fundamentais para a funcionalidade da API para a classificação de comentários. A variável `model` contém o modelo treinado que é utilizado para classificar o sentimento dos comentários de texto. Já o `vectorizer` contém um objeto de vetorização, que é responsável por transformar o texto do comentário em uma representação numérica adequada para ser alimentada ao modelo de Regressão Logística. A variável `app` é uma instância da classe Flask que representa o aplicativo web. O arqgumento __name__ é utilizado para determinar o local do pacote ou módulo. `CORS(app)` permite que a api seja acessada de diferente domínios. `@app.route('/', methods=['POST'])`, esta é uma decoradora que define a rota principal ('/') para a qual as requisições POST são tratadas pela função classificar.

 ````python
 def classificar():
    data = request.get_json()
    texto = data['texto']
    texto_vetorizado = vectorizer.transform([texto])
    resultado = model.predict(texto_vetorizado)[0]
    return jsonify({'sentimento': resultado})
````
 A função classificar é responsável por receber as requisições HTTP na rota principal ('/'), extrair o texto do comentário a ser classificado, vetorizar o texto usando o vetorizador treinado e, em seguida, realizar a predição de sentimento utilizando o modelo de Regressão Logística. O resultado da predição é então retornado como uma resposta JSON.

 
#### ⚙️| Como a API funciona 

![imagem png](https://github.com/eumateusdev/Data-Spectrum-mandacaru.dev/assets/64427453/9b490fb5-b6f0-4ea5-a9ac-66b94644d608)

A imagem mostra o funcionamento de uma API de análise de sentimento. A API foi hospedada gratuitamente no Raiway. No caso desta imagem, o sistema do usuário é o front end, e a API, que está hospedada em um servidor. O usuário envia um comentário pelo front end. O front end envia o comentário para a API. A API usa um analisador de sentimento para determinar o sentimento do comentário. O analisador de sentimento é um algoritmo que analisa o texto do comentário e tenta determinar se é positivo, negativo ou neutro.

A imagem mostra os seguintes passos:

 1. O usuário envia um comentário para o front end.
 2. O front end envia o comentário para a API.
 3. A API usa o modeo treinado para determinar o sentimento do comentário.
 4. A API retorna o sentimento de forma vísivel para o usuário pelo front-end.

