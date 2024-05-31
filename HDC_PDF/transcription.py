import streamlit as st
import whisper as w


st.title("Whisper App")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])


model = w.load_model("base")
st.text("Whisper Model Loaded")

if st.sidebar.button("Transcribe Audio"):

    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        transcription = model.transcribe(audio_file.name)
        st.sidebar.success("Transcription Complete")
        st.markdown(transcription["text"])
    else: 
        st.sidebar.error("Please upload an audio file")

st.header("Play Original Audio File")
st.audio(audio_file)