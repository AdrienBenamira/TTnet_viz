
import streamlit as st
from PIL import Image



def app():


    st.title("Rules Visualisation ")


    for rules in []:
        image = Image.open()
        image = image.resize((28,28), Image.NEAREST)
        st.image(image, caption='Uploaded Image.', use_column_width=False)
        st.write("")
        st.write("Identifying...")




if __name__ == '__main__':
    app()