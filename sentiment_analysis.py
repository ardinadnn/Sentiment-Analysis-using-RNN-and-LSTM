import re, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder

app = Flask(__name__)

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

app.json_encoder = LazyJSONEncoder

max_features = 100000
tokenizer = Tokenizer(num_words=max_features,split=' ',lower=True)
sentiment = ['positive', 'neutral', 'negative'] #yang bener gimana woy
# sentiment = ['negative', 'neutral', 'positive']

file = open("resources_of_rnn/x_pad_sequences.pickle",'rb')
feature_file_from_rnn = pickle.load(file)
file.close()

file = open("resources_of_rnn/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

rnn_model = load_model('model_of_rnn/modelRNN1.h5')
lstm_model = load_model('model_of_lstm/modelLSTM.h5')
#------------------------------------------------------------------------------------

@swag_from("docs/rnn.yml", methods=["POST"])
@app.route('/inputFormRNN', methods=['POST'])
def main_RNN():
    teks = request.form.get('teks')
    teks = teks.lower()
    teks = removePunctuation(teks)
    teks = removeWhitespace(teks)

    max_features = 100000
    tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
    tokenizer.fit_on_texts([teks])

    feature = tokenizer.texts_to_sequences([teks])
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = rnn_model.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response={
        'status_code': 200,
        'description': "Analisis Sentimen",
        'teks': teks,
        'sentiment': get_sentiment
    }

    response_data = jsonify(json_response)
    return response_data
#------------------------------------------------------------------------------------
# UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
api = Api(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadCSVRNN(Resource):
    @swag_from("docs/csv_upload_rnn.yml", methods=["POST"])
    def post(self):
        if 'file' not in request.files:
            return jsonify({'error': 'File tidak ditemukan'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Tidak ada file terpilih'}), 400

        if file and allowed_file(file.filename):
            datasetOri = pd.read_csv(file,encoding='latin-1')
            datasetOri = datasetOri.astype(str)
            output = pd.DataFrame()
            dataset = datasetOri.iloc[:,0]
            dataset = dataset.to_frame(name="Output")

            for column_name in dataset.columns:
                output[column_name] = output[column_name].apply(removePunctuation)
                output[column_name] = output[column_name].str.lower()
                output[column_name] = output[column_name].apply(removeWhitespace)

            # feature = tokenizer.texts_to_sequences(output???)
            # feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

            # # prediction = rnn_model.predict(feature)
            # get_sentiment = sentiment[np.argmax(prediction[0])]

            response_data = {
            'status_code': 200,
            'message': 'File berhasil diunggah.',
            'sqlite3_url': '/output.db',
            'output': "" #get_sentiment
            }
            return jsonify(response_data), 200
        else:
            return jsonify({'error': 'Format file tidak valid'}), 400

app.add_url_rule('/uploadCSVRNN', view_func=UploadCSVRNN.as_view('upload_csv'))
# api.add_resource(UploadCSVRNN, '/uploadCSVRNN')

#------------------------------------------------------------------------------------

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Documentation for Sentiment Analysis'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi API untuk Sentiment Analysis')

    },
    host = "127.0.0.1:5000/"
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template,
                  config=swagger_config)

def removePunctuation(teks):
    punctuation = string.punctuation
    output = ''.join(char if char not in punctuation else ' ' for char in teks)
    output = re.sub(r'[^a-zA-Z0-9\s]', ' ', output) #remove simbol aneh2
    output = re.sub(r'\b[x]\w{2}\b', ' ', output)
    return output

def removeWhitespace(teks):
    output = ' '.join(teks.split())
    return output


@swag_from("docs/lstm.yml", methods=["POST"])
@app.route('/inputFormLSTM',methods=['POST'])
def main_lstm():
    teks = request.form.get('teks')
    teks = teks.lower()
    teks = removePunctuation(teks)
    teks = removeWhitespace(teks)

    max_features = 100000
    tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
    tokenizer.fit_on_texts([teks])

    feature = tokenizer.texts_to_sequences([teks])
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    prediction = lstm_model.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response={
        'status_code': 200,
        'description': "Analisis Sentimen",
        'teks': teks,
        'sentiment': get_sentiment
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run() #debug=True