import gradio as gr
from groq import Groq
from transformers import pipeline
import os

# Read secret from Hugging Face
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# Whisper speech-to-text model
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)


def voice_to_summary(audio_file, summary_type):

    try:

        if audio_file is None:
            return "Please upload an audio file.", ""

        # Convert speech to text
        transcription = transcriber(audio_file)["text"]

        if transcription.strip() == "":
            return "Could not transcribe audio.", ""

        # Summarization prompt
        prompt = f"""
Summarize the following transcript in a {summary_type} way.

Transcript:
{transcription}

Return only the summary.
"""

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,
            max_tokens=300
        )

        summary = response.choices[0].message.content

        return transcription, summary

    except Exception as e:

        return "Error: " + str(e), ""


demo = gr.Interface(

    fn=voice_to_summary,

    inputs=[

        gr.Audio(
            type="filepath",
            label="Upload Audio"
        ),

        gr.Radio(
            ["short", "medium", "detailed", "bullet points"],
            value="medium",
            label="Summary Type"
        )
    ],

    outputs=[

        gr.Textbox(
            lines=10,
            label="Transcript"
        ),

        gr.Textbox(
            lines=10,
            label="Summary"
        )
    ],

    title="AI Voice-to-Summary App",

    description="Upload audio → Whisper transcription → Groq summary."
)

demo.launch()