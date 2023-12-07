from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import requests
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Chargement du model
saved_w = "premiers_poids.h5"
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(saved_w)

def preprocess(img):
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"hello": "kevin"})

#@app.before_request
"""
@app.before_first_request
def load():
    model_path = "best_model.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du model
model = load()
"""


@app.route("/predict", methods=['POST'])
def predict_and_detect():
    # recuperer l'image
    file = request.files['file']
    image = file.read()
    
    

    # Ouvrir l'image
    img = Image.open(BytesIO(image))
    
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    #traitement de l'image
    #img_processed = preprocess(img)

    # predictions
    pred = model.predict(img)
    rec = pred.tolist()
    print("le type de la prédiction : ", type(pred))
    print(pred)
    #rec = pred[0][0].tolist()

    return jsonify({"predictions" : rec})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    #app.run(debug=True)