import streamlit as st
import pathlib
from pathlib import Path
from fastai.vision.all import *
from fastai.vision.widgets import *

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class Predictor:
    def __init__(self, filename: str):
        self.learn_interface =load_learner(Path() / filename)
        self.img = self.get_image_from_upload()

        if self.img is not None:
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():
        st.header("FrogApp")
        uploaded_file = st.file_uploader("Wybierz żabkę do oceny", type = ['png', 'jpg','jpeg'])
        
        if uploaded_file is not None:
            return PILImage.create(uploaded_file)

    def get_prediction(self):
        if st.button("Klasyfikuj"):
            prediction, index, probability = self.learn_interface.predict(self.img) 
            prob = probability[index].item() * 100
            left, right = st.columns(2)
            with left:
                st.metric("Klasa", prediction)
            with right:
                st.metric("Prawdopodobieństwo", "{0:.0f}".format(prob))

if __name__ == '__main__':
    file_name = 'model.pkl'
Predictor(filename=file_name)