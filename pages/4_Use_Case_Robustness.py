from PIL import Image
import streamlit as st

st.set_page_config(page_title="TTnet 4 Robustness ", page_icon="💪")

st.markdown("# 💪 Robustness Formally Verified & TTnet")
st.sidebar.header("Comparison method")
st.write(

"""The bellow Figure illustrates the superiority of TTnet verifed robustness over State-of-the-art on MNIST."""
)

image = Image.open('Images/Verification.png')
st.image(image, caption='Comparison of complete SAT and MIP methods for MNIST low noise (ε = 0.1, in blue) and high noise (ε = 0.3, in red) with regards to verified accuracy and verification time. ',width=1000)
