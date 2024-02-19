[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hugging%20Face-ff69b4.svg)]([https://huggingface.co/](https://huggingface.co/spaces/asif00/Talk_to_Doc-Advanced_RAG_for_Reasoning_and_QA_with_Gemini_Pro))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WIP](https://img.shields.io/badge/status-WIP-orange.svg)](https://github.com/your-username/your-repo/wiki/Roadmap)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Gemini Pro](https://img.shields.io/badge/Gemini%20Pro-green.svg)](https://ai.google.dev/)


# Talk to Doc: Advanced RAG for Reasoning and QA with Gemini Pro

## Welcome to Talk to Doc!

Tired of struggling to find answers in your PDFs? Talk to Doc is here to help! Ask it questions, and it will use its advanced capabilities to extract the information you need quickly and easily.

## How It Works

1. **Upload Your PDF:** Select the PDF you have questions about.
2. **Click "Process":** Wait a few seconds for the system to analyze the document.
3. **Start Asking:** Type your question in the box and let Talk to Doc do the rest!

## Under the Hood

This system leverages the power of cutting-edge AI models to understand your PDFs and answer your questions:

* **Chunking:** LangChain splits your PDF into manageable sections for efficient processing.
* **Embedding:** Each section receives a unique representation using "GeminiEmbeddingFunction" for better understanding.
* **Storage and Search:** ChromaDB stores these embeddings, enabling rapid retrieval of relevant information.
* **Query Expansion:** "models/text-bison-001" expands your query to consider various ways it might be phrased.
* **Cross-Encoder Re-ranking:** "cross-encoder/ms-marco-MiniLM-L-6-v2" prioritizes the most relevant answers.
* **Response Generation:** "gemini-pro" synthesizes the information into a clear and concise response.

## Get Started Today!

**Upload your PDF and experience the power of Talk to Doc!**

## Features

* **Accurate and efficient information retrieval:** Talk to Doc uses cutting-edge models to understand the content of your PDFs and provide precise answers.
* **Flexible query expansion:** Don't worry about wording your question perfectly. Talk to Doc considers various ways your question might be phrased.
* **Easy to use:** No need for complex configurations. Simply upload your PDF and start asking questions.

## Try the Live Demo!

Experience Talk to Doc firsthand by trying it out in the Hugging Face platform:

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hugging%20Face-ff69b4.svg)](https://huggingface.co/spaces/asif00/Talk_to_Doc-Advanced_RAG_for_Reasoning_and_QA_with_Gemini_Pro)

Ask your questions directly in the interface and see how Talk to Doc uses its advanced capabilities to answer them quickly and accurately.


## Resources

* **LangChain:** https://github.com/huggingface/langchain
* **ChromaDB:** https://docs.trychroma.com
* **Gemini Pro:** https://ai.google.dev
