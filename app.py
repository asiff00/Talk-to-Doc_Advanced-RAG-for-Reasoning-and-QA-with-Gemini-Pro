import os
import gradio as gr

from src.init import Initializer
from dotenv import load_dotenv

load_dotenv()
AUG_TOKEN = os.environ.get("AUGMENT_MODEL")
RES_TOKEN = os.environ.get("RESPONSE_MODEL")

pdf_loaded = False
processing = False


def load_pdf(pdf_file_path):
    global pdf_loaded
    if pdf_file_path is None:
        return "Upload PDF First"
    filename = pdf_file_path.name
    global brain
    brain = Initializer.initialize(AUG_TOKEN, RES_TOKEN, filename)
    pdf_loaded = True
    return "Ask Questions Now!"


def response(query, history):
    global processing
    if not pdf_loaded or processing:
        return "Please wait...", history
    processing = True
    output = brain.generate_answers(query)
    history.append((query, output))
    processing = False
    return "", history


with open("src/style.css", "r") as file:
    css = file.read()

with open("src/content.html", "r") as file:
    html_content = file.read()
    parts = html_content.split("<!-- split here -->")
    title_html = parts[0]
    bts_html = parts[1] if len(parts) > 1 else ""


def loading():
    return "Loading ..."


with gr.Blocks(css=css) as app:
    with gr.Column(elem_id="column_container"):
        gr.HTML(title_html)

        with gr.Column():
            send = gr.Label(value="UPLOAD your PDF below")
            pdf = gr.File(label="Load your PDF document:", file_types=[".pdf"])
            with gr.Row():
                status = gr.Label(label="Status:", value="Process Document!")
                load_pdf_button = gr.Button(value="Process")

        chatbot = gr.Chatbot([], elem_id="chatbot")
        with gr.Column():
            send = gr.Label(value="Write your QUESTION bellow and hit ENTER")
        query = gr.Textbox(
            label="Type your questions here:",
            placeholder="What do you want to know?",
        )
        clear = gr.ClearButton([query, chatbot])
        gr.HTML(bts_html)

    load_pdf_button.click(loading, outputs=[status], queue=False)
    load_pdf_button.click(load_pdf, inputs=[pdf], outputs=[status], queue=True)
    query.submit(response, [query, chatbot], [query, chatbot], queue=True)

app.launch()
