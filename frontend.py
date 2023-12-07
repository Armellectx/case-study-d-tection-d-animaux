import streamlit as st
from PIL import Image
import requests
import numpy as np


st.title("Prévention - Reconnaissance d'animaux")

upload = st.file_uploader("Chargez l'image à détecter",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    
    #interaction avec le backend
    files = {"file" :  upload.getvalue()}
    req = requests.post("http://127.0.0.1:8080/predict", files=files)
    resultat = req.json()
    
    #traitement du résultat
    rec = resultat["predictions"] #from the backend
    predicted_classes = np.argmax(rec, axis=1)
    if 0.3 <= rec[0][0] <= 0.7:
        c2.write("No Wild boar or Deer detected on this picture")
    elif predicted_classes == 0: 
        c2.write("It is a wild boar. Il s'agit d'un sanglier (wild boar)")
    else:
        c2.write("It is a deer. Il s'agit d'un cerf (deer)")
    c1.image(Image.open(upload))
    
    #if prob_recyclable > 50:
        #c2.write(f"Je suis certain à {prob_recyclable:.2f} % que l'objet est recyclable")
    #else:
        #c2.write(f"Je suis certain à {prob_organic:.2f} % que l'objet n'est pas recyclable")
    