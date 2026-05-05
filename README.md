# 🎥 YouTube Video Chatbot (RAG + LangChain)

This project is a **YouTube Video Question Answering Chatbot** built using **LangChain, OpenAI, and Streamlit**.

It allows users to:
- Extract transcripts from YouTube videos
- Convert them into embeddings
- Store them in a vector database (ChromaDB)
- Ask questions based on the video content
- Get answers with **timestamps and references**

---

## 🚀 Features

- Extracts transcript from YouTube videos  
- Splits transcript into meaningful chunks with timestamps  
- Creates embeddings using OpenAI  
- Stores embeddings in Chroma vector database  
- Uses RAG (Retrieval-Augmented Generation) for answering  
- Supports:
-- Normal Q&A
-- Summarization questions  
-- Displays **timestamp-based sources**

---

## 🧠 How It Works

1. User enters a **YouTube Video ID**
2. Transcript is fetched using:
   - `YouTubeTranscriptApi`
3. Transcript is split into chunks
4. Chunks are converted into embeddings
5. Stored in **Chroma Vector DB**
6. When user asks a question:
   - Relevant chunks are retrieved
   - LLM (GPT) generates answer using context

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI (Embeddings + LLM)
- ChromaDB (Vector Database)

---

<img width="1266" height="651" alt="Screenshot from 2026-04-28 16-47-39" src="https://github.com/user-attachments/assets/c09508ba-2fdc-4878-b12c-75d6bf4b0705" />

