"""
Created on Tue Jul  6 21:30:54 2021
@author: User
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st

from pages.modules.helper import DBEncoder, read_csv


def app():
    """
    Main function that contains the application to train keras based models.
    """

    st.title("TTnet Training Basic UI")
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
                for epoch in range(100):
                    print("\nStart of epoch %d" % (epoch,))
                    st.write("Epoch {}".format(epoch + 1))
                    start_time = time.time()
                    progress_bar = st.progress(0.0)
                    percent_complete = 0
                    epoch_time = 0
                    # Creating empty placeholder to update each step result in epoch.
                    st_t = st.empty()

                    train_loss_list = []
                    # Iterate over the batches of the dataset.
                    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        start_step = time.time()
                        loss_value = train_step(x_batch_train, y_batch_train)
                        end_step = time.time()
                        epoch_time += (end_step - start_step)
                        train_loss_list.append(float(loss_value))

                        # Log every 200 batches.
                        if step % 200 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %d samples" % ((step + 1) * batch_size))
                            step_acc = float(train_acc_metric.result())
                            percent_complete = ((step / train_steps_per_epoch))
                            progress_bar.progress(percent_complete)
                            st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}" \
                                       .format((epoch_time), float(step_acc)))

                    progress_bar.progress(1.0)

                    # Display metrics at the end of each epoch.
                    train_acc = train_acc_metric.result()
                    print("Training acc over epoch: %.4f" % (float(train_acc),))

                    # Reset training metrics at the end of each epoch
                    train_acc_metric.reset_states()

                    # Find epoch training loss.
                    print(train_loss_list)
                    train_loss = round((sum(train_loss_list) / len(train_loss_list)), 5)

                    val_loss_list = []
                    # Run a validation loop at the end of each epoch.
                    for x_batch_val, y_batch_val in val_dataset:
                        val_loss_list.append(float(test_step(x_batch_val, y_batch_val)))

                    # Find epoch validation loss.
                    val_loss = round((sum(val_loss_list) / len(val_loss_list)), 5)

                    val_acc = val_acc_metric.result()
                    val_acc_metric.reset_states()

                    print("Validation acc: %.4f" % (float(val_acc),))
                    print("Time taken: %.2fs" % (time.time() - start_time))
                    st_t.write(
                        "Duration : {0:.2f}s, Training acc. : {1:.4f}, Validation acc.:{2:.4f}" \
                        .format((time.time() - start_time), float(train_acc), float(val_acc)))

                    # Check if model needs to be saved, and if yess, then with what condition.
                    if save_model:
                        if save_condition:
                            if epoch == 0:
                                best_train_acc = train_acc
                                best_train_loss = train_loss
                                best_val_loss = val_loss
                                best_val_acc = val_acc

                                # Save first model.
                                model.save("./model/" + save_model + ".h5", overwrite=True,
                                           include_optimizer=True)
                                if save_condition in ("train acc", "val acc"):
                                    st.write("Saved model {} as {} increased from 0 to {}." \
                                             .format(save_model + ".h5", save_condition,
                                                     round(train_acc,
                                                           3) if save_condition == "train acc" else round(
                                                         val_acc, 3)))
                                else:
                                    st.write(
                                        "Saved model {} as {} decreased from infinite to {}." \
                                        .format(save_model + ".h5", save_condition,
                                                round(train_loss,
                                                      3) if save_condition == "train loss" else round(
                                                    val_loss, 3)))
                            else:
                                if save_condition == "train acc":
                                    if train_acc >= best_train_acc:
                                        model.save("./model/" + save_model + ".h5",
                                                   overwrite=True,
                                                   include_optimizer=True)
                                        st.write("Saved model {} as {} increased from {} to {}." \
                                                 .format(save_model + ".h5", save_condition,
                                                         round(best_train_acc, 3),
                                                         round(train_acc, 3)))
                                        best_train_acc = train_acc
                                    else:
                                        st.write(
                                            "Not saving model as {} did not increase from {}." \
                                            .format(save_condition, round(best_train_acc, 3)))
                                elif save_condition == "val acc":
                                    if val_acc >= best_val_acc:
                                        model.save("./model/" + save_model + ".h5",
                                                   overwrite=True,
                                                   include_optimizer=True)
                                        st.write("Saved model {} as {} increased from {} to {}." \
                                                 .format(save_model + ".h5", save_condition,
                                                         round(best_val_acc, 3),
                                                         round(val_acc, 3)))
                                        best_val_acc = val_acc
                                    else:
                                        st.write(
                                            "Not saving model as {} did not increase from {}." \
                                            .format(save_condition, round(best_val_acc, 3)))

                                elif save_condition == "train loss":
                                    if train_loss >= best_train_loss:
                                        model.save("./model/" + save_model + ".h5",
                                                   overwrite=True,
                                                   include_optimizer=True)
                                        st.write("Saved model {} as {} decreased from {} to {}." \
                                                 .format(save_model + ".h5", save_condition,
                                                         round(best_train_loss, 3),
                                                         round(train_loss, 3)))
                                        best_train_loss = train_loss
                                    else:
                                        st.write(
                                            "Not saving model as {} did not increase from {}." \
                                            .format(save_condition, round(best_train_loss, 3)))

                                elif save_condition == "val loss":
                                    if val_loss >= best_val_loss:
                                        model.save("./model/" + save_model + ".h5",
                                                   overwrite=True,
                                                   include_optimizer=True)
                                        st.write("Saved model {} as {} decreased from {} to {}." \
                                                 .format(save_model + ".h5", save_condition,
                                                         round(best_val_loss, 3),
                                                         round(val_loss, 3)))
                                        best_val_loss = val_loss
                                    else:
                                        st.write(
                                            "Not saving model as {} did not increase from {}." \
                                            .format(save_condition, round(best_val_loss, 3)))


if __name__ == '__main__':
    app()