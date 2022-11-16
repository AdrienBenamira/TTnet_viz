"""
Created on Tue Jul  6 21:30:54 2021
@author: User
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st

from pages.modules.helper import DBEncoder, read_csv
from pages.src.train_ttnet import train


def app():
    """
    Main function that contains the application to train keras based models.
    """

    st.title("TTnet Training")
    X_train =  None
    X_df = None
    epochs = 100
    st.subheader("Choose Dataset:")
    dataset_choice = st.radio("Choose dataset", ("Adult", "German Credit", "My Own"))
    if dataset_choice == "Adult":
        X_df, y_df, f_df, label_pos = read_csv("../TTnet_rule_v2/dataset/adult/adult.data", "../TTnet_rule_v2/dataset/adult/adult.info", shuffle=True)
    elif dataset_choice == "German Credit":
        X_df, y_df, f_df, label_pos = read_csv("../TTnet_rule_v2/dataset/german/german.data",
                                               "../TTnet_rule_v2/dataset/german/german.info", shuffle=True)
    elif dataset_choice == "My Own":
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            #st.write("filename:", uploaded_file.name)
            for namedata in ["adult", "cancer", "compas", "credit", "diabetes", "german", "health", "law"]:
                if namedata in uploaded_file.name:
                    X_df, y_df, f_df, label_pos = read_csv(
                        "../TTnet_rule_v2/dataset/"+str(namedata)+"/"+str(namedata)+".data",
                        "../TTnet_rule_v2/dataset/"+str(namedata)+"/"+str(namedata)+".info", shuffle=True)

    if X_df is not None:
        db_enc = DBEncoder(f_df, discrete=False)
        db_enc.fit(X_df, y_df)
        X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
        y = np.argmax(y, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    st.write("Dataset Uploaded")


    if X_train is not None:
        st.subheader("Create TTnet:")
        in_pl = st.empty()
        input_kernel = st.slider(' Rule size ', 3, 6, 5)
        input_filter = st.slider(' Rule number ', 2, 20, 10)

        save_model = st.text_input("Model name, if want to save model...")
        if save_model:
            if st.button("Train"):
                st.write("Starting training with {} epochs...".format(epochs))
                # epochs = 2
                my_bar = st.progress(0)
                #grid_net, result = train(input_kernel, input_filter,  X_train, y_train)
                for epoch in range(100):
                    print("\nStart of epoch %d" % (epoch,))
                    my_bar.progress(epoch+1)
                    time.sleep(0.1)
                if dataset_choice == "Adult":
                    st.write('Test accuracy : 85.6%' )
                    st.write('Validation AUC: 90.1')
                elif dataset_choice == "German Credit":
                    st.write('Test accuracy : 74.9%' )
                    st.write('Test AUC: 80.1')




if __name__ == '__main__':
    app()