
import streamlit as st
from PIL import Image



def app():


    st.title("Rules Visualisation ")


    st.markdown(""" ### Rule 1   """)
    st.markdown(""" (Years of Education > 11) OR (Capital Gain > 4.2k\$) OR (Capital Loss < 4.2k\$)  """)

    image = Image.open('Images/rule1.png')
    st.image(image, width=200)

    st.markdown(""" #### Probabilities Rule 1 : P(Class=1| r is True) = 0.489 """)

    st.markdown(""" ----------------------------------------------------------------------------------------------------- """)

    st.markdown(""" ### Rule 2  """)
    st.markdown(""" (Is married ?) """)


    image = Image.open('Images/rule2.png')
    st.image(image, width=200)

    st.markdown(""" #### Probabilities Rule 2 : P(Class=1| r is True) = 0.435 """)

    st.markdown(""" ----------------------------------------------------------------------------------------------------- """)

    st.markdown(""" ### Rule 3 """)
    st.markdown("""  (Born in Mexico ?) OR (Born in Nicaragua ?) """)


    image = Image.open('Images/rule3.png')
    st.image(image, width=200)

    st.markdown(""" #### Probabilities Rule 3 : P(Class=0| r is True) = 0.959 """)

    st.markdown(""" ----------------------------------------------------------------------------------------------------- """)

    st.markdown(""" ### Rule 4 """)
    st.markdown("""  ( (Capital Gain < 4.2k\$) OR  ((Years of Education < 11) AND  (Capital Loss > 4.2k\$) OR ((Years of Education < 11) AND  (Work more than (per week) < 45 hours)) """)


    image = Image.open('Images/rule4.png')
    st.image(image, width=200)

    st.markdown(""" #### Probabilities Rule 4 : P(Class=0| r is False) = 0.833 """)



if __name__ == '__main__':
    app()