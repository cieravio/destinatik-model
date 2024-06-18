import os
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from google.cloud import storage

app = Flask(__name__)

MODEL_URL = 'https://storage.googleapis.com/capstone-model-bucket/model_destinatik_v2.h5'

def load_model():
    model_path = get_file('model/model_destinatik_v2.h5', MODEL_URL, cache_subdir='/tmp', extract=false)
    return model_path

MODEL_PATH = load_model()
model = load_model(MODEL_PATH, compile=False)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status':{
            'code': 200,
            'message': 'Server is running'
        }
    }), 200

@app.route('/recommendation', methods=['POST'])
def recommendPlace():
    data = request.get_json()
    if 'user_id' not in data:
        return jsonify({
            'status': {
                'code': 400,
                'message': 'No user ID provided'
            }
        }), 400
    
    user_id = data['user_Id']

    user_input = np.array([[int(user_id)]])

    recommendations = model.predict(user_input)

    return jsonify({
        'status': {
            'code': 200,
            'message': 'Recommendations generated successfully'
        },
        'recommendations': recommendations.tolist()
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)

    