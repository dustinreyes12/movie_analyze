import streamlit as st
from multiapp import MultiApp
from apps import actor_analysis # import your app modules here

app = MultiApp()

# Add all your application here
# app.add_app("Movie Analysis", movie_analysis.app)
app.add_app("Actor Analysis", actor_analysis.app)

# The main app
if __name__ == '__main__':
    app.run()