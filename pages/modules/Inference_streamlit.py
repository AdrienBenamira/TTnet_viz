
import streamlit as st
from PIL import Image



def app():


    st.title("Inference With Rules - In progress")



    st.markdown(

        """ We would like to code an interface similar to the image below: """
    )
    image = Image.open('Images/img.png')
    st.image(image,width=1000)









if __name__ == '__main__':
    app()