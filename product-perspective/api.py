from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('modelo_treinado.joblib')
vectorizer = joblib.load('vectorizer.joblib')

app = Flask(__name__)
CORS(app) 
@app.route('/', methods=['POST'])
def classificar():
    # Obter os dados da requisição
    data = request.get_json()

    # Extrair o texto a ser classificado da requisição
    texto = data['texto']

    # Vetorizar o texto usando o vetorizador treinado
    texto_vetorizado = vectorizer.transform([texto])

    # Fazer a predição usando o modelo treinado
    resultado = model.predict(texto_vetorizado)[0]

    # Retornar o resultado como JSON
    return jsonify({'sentimento': resultado})

if __name__ == '__main__':
    app.run(debug=True)
