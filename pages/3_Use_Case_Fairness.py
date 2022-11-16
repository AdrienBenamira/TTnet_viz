from PIL import Image



import streamlit as st

st.set_page_config(page_title="TTnet 4 Fairness ", page_icon="⚖️")

st.markdown("# ⚖ Fairness & TTnet")
st.sidebar.header("Comparison method")
st.write(

""" The bellow Figure illustrates the TTnet competitivity to State-of-the-art on Adult and Compas on **fairness, accuracy and complexity**."""
)

image = Image.open('Images/fairness.png')
st.image(image, caption=' Graphical comparison of our TT-rules model to a series of rule-based models using the Adults or Compas fairness datasets. ',width=1000)
