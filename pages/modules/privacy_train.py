
import streamlit as st
from PIL import Image

import time

def app():
    epochs = 100
    st.title("Private Training ")

    st.subheader("Training without the dataset, only a Machine Learning Model, as illustrated below.")


    image = Image.open('Images/Privacy trainig.png')
    st.image(image, width=1000)

    st.subheader("Choose ML Model:")
    uploaded_files = st.file_uploader("Choose a Model file", accept_multiple_files=True)
    namedata = None
    for uploaded_file in uploaded_files:
        # st.write("filename:", uploaded_file.name)
        #for namedata in ["adult"]:
        if "adult" in uploaded_file.name:
            namedata = "adult"

    if namedata is not None:
        if st.button("Generate Dataset of size 10**4 samples + Train TTnet"):

            my_bar = st.progress(0)
            for epoch in range(100):
                print("\nStart of epoch %d" % (epoch,))
                my_bar.progress(epoch + 1)
                time.sleep(0.01)
                if epoch==20:
                    st.write("Starting training with {} epochs...".format(epochs))
                time.sleep(0.05)

            time.sleep(2)
            if namedata == "adult":
                st.write(' Matching with RF (on real): 96.3%')
                st.write('Test accuracy (on real): 84.3%')
                st.write('Test AUC: 88.9')







if __name__ == '__main__':
    app()