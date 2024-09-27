import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Überblick über die Projektaufgabe")

st.markdown("# Überblick über die Projektaufgabe")
st.sidebar.header("Überblick")
st.write(
    "Im Rahmen unseres Projekts zur prädiktiven Analyse von Fahrradunfällen untersuchen wir Beinahe-Unfälle und reale Unfälle, die im Jahr 2022 erfasst wurden. Ziel ist es, Risikofaktoren für Fahrradunfälle zu identifizieren und darauf aufbauend Maßnahmen zur Verbesserung der Verkehrssicherheit für Radfahrer zu entwickeln. Durch die Kombination der verschiedenen Datenquellen und die Nutzung maschineller Lernmethoden sollen Gefahrenstellen vorhergesagt und präventive Strategien entwickelt werden."

)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
