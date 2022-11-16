import streamlit as st
import pages.modules.train_streamlit as app1
import pages.modules.predict_streamlit as app2

PAGES = {
    "Train": app1,
    "Predict": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()