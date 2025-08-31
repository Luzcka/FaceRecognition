# frontend/streamlit_app.py

import streamlit as st
import cv2
import tempfile
import requests

st.title("Facial Recognition PoC")

action = st.radio("Ação", ["Registrar", "Pesquisar"])

camera = st.camera_input("Coloque seu rosto dentro da moldura e tire uma foto")

if camera:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(camera.read())
        image_path = f.name

    if action == "Registrar":
        name = st.text_input("Nome")
        reg = st.text_input("Número de Registro")
        if st.button("Registrar") and name and reg:
            files = {"image": open(image_path, "rb")}
            data = {"name": name, "registration_number": reg}
            r = requests.post("http://localhost:8000/register", files=files, data=data, headers={"x-api-key": "supersecret"})
            st.write(r.json())

    if action == "Pesquisar":
        if st.button("Buscar"):
            files = {"image": open(image_path, "rb")}
            r = requests.post("http://localhost:8000/search", files=files, headers={"x-api-key": "supersecret"})
            st.write(r.json())
