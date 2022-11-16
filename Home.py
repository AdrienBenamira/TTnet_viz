import streamlit as st
from PIL import Image
#st.title('Truth Table Net Demo')

st.set_page_config(
    page_title="TTnet Demo",
    page_icon="☑️",
)

st.write("# Welcome to Truth Table Net Demo! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    
    Truth Table Net (TTnet) is a Neural Network Model built specifically for
    **Formal Verification, Interpretability & Security**.
    
    👈 Select a demo from the sidebar to investiagte some examples
    of what TTnet can do!
    
    ### Introduction
    
    Machine/Deep Learning community needs a trade-off between Neural Networks models & Rule-Based models.
    
     We successfully propose one 😎
    
    """)
image = Image.open('Images/Intro_v2.png')
st.image(image, caption='A simple ML/DL models comparisons according Accuracy & Explainability performances.',width=500)
st.markdown("""
    
     📈 We scale our model to:
    
    - Finance & Healthcare Tabular Datasets with application to __Global Explainability__ & __Fairness__
    - Image Datasets (MNIST & CIFAR-10) with application to __Robustness Formal Verification__

    
   ### How it works? 
    
    TTnet is a Deep Convolutional Neural Network in which each filter can be, by design, translated into a truth table.
    
    Here is an explanation video:
    
        """)
video_file = open('Images/explanations.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)
st.markdown("""
    
    ### Want to learn more?
    - Check out our [paper on Formal Verification](https://arxiv.org/abs/2208.08609)
    - Check out our [paper on Interpretability](https://openreview.net/pdf?id=5pU6126YRp)
    - Check out our [paper on Security](https://eprint.iacr.org/2022/1247)
    
"""
)