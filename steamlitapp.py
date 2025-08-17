import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Adding a header and some text
st.header("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")

# Create an interactive widget (slider)
age = st.slider("Select your age", 0, 100, 25)
st.write("Your selected age is:", age)
