import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="SOP-Chatbot", page_icon="🩺", layout="centered")

# --- Sidebar: Kontext / Info ---
with st.sidebar:
    st.title("🩺 SOP-Chatbot")
    st.caption("Klinische SOPs · Charité")
    st.divider()
    st.markdown(
        "Dieser Assistent beantwortet Fragen ausschließlich "
        "auf Basis der hinterlegten klinischen SOP-Dokumente."
    )
    st.warning(
        "Dieser Chatbot ist eine KI. Er kann Fehler machen und "
        "falsche Informationen liefern. Antworten sind stets zu überprüfen."
    )
    if st.button("Neue Unterhaltung"):
        st.session_state.messages = []
        st.rerun()

st.title("SOP-Chatbot")

# --- Chat-Verlauf initialisieren ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Bisherigen Verlauf anzeigen ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Quellen anzeigen"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** · Chunk {src['chunk_index']}")
                    st.write(src["text"])
                    st.divider()

# --- Eingabe ---
if prompt := st.chat_input("Ihre Frage zu den SOPs..."):
    # Nutzer-Nachricht anzeigen + speichern
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Antwort holen
    with st.chat_message("assistant"):
        with st.spinner("Antwort wird generiert..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"query": prompt},
                    timeout=180
                )
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Backend nicht erreichbar. Läuft der FastAPI-Server?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Zeitüberschreitung – die Generierung hat zu lange gedauert.")
                st.stop()

        answer = data["answer"]
        sources = data["sources"]

        # Effekt "Tippen": bereits erhaltenen Text wortweise ausgeben
        def stream_words(text):
            for word in text.split():
                yield word + " "
                time.sleep(0.02)

        st.write_stream(stream_words(answer))

        with st.expander("Quellen anzeigen"):
            for src in sources:
                st.markdown(f"**{src['source']}** · Chunk {src['chunk_index']}")
                st.write(src["text"])
                st.divider()

    # Antwort + Quellen im Verlauf speichern
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })