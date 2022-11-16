import streamlit as st
import pages.modules.train_streamlit as app1
import pages.modules.privacy_train as app3
import pages.modules.fairness_train as app4
import pages.modules.predict_streamlit as app2

PAGES = {
    "Train to scale": app1,
    "Train for privacy": app3,
    "Explore Rules": app2
}
st.set_page_config(page_title="TTnet 4 Explainability ", page_icon="🤯️")
st.sidebar.title('Exploration')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()