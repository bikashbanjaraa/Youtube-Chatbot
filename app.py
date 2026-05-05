import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

    transcript_data = [
        {
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.start + chunk.duration,
        }
        for chunk in fetched_transcript
    ]

    return transcript_data


def build_chunks_with_timestamps(transcript_data, max_chars=1000, overlap_chars=200):
    chunks = []
    current_text = ""
    current_start = None
    current_end = None

    for item in transcript_data:
        text = item["text"]
        start = item["start"]
        end = item["end"]

        if current_start is None:
            current_start = start

        if len(current_text) + len(text) <= max_chars:
            current_text += " " + text
            current_end = end
        else:
            chunks.append(
                Document(
                    page_content=current_text.strip(),
                    metadata={
                        "start": current_start,
                        "end": current_end,
                    },
                )
            )

            overlap_text = current_text[-overlap_chars:]
            current_text = overlap_text + " " + text
            current_start = start
            current_end = end

    if current_text:
        chunks.append(
            Document(
                page_content=current_text.strip(),
                metadata={
                    "start": current_start,
                    "end": current_end,
                },
            )
        )

    return chunks


def create_vector_store(chunks, video_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"youtube_{video_id}",
    )

    return vector_store


def is_summary_question(question):
    summary_keywords = [
        "summarize",
        "summary",
        "what is this video about",
        "explain in short",
        "short summary",
        "brief summary",
        "overview",
        "main idea",
        "main points",
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summary_keywords)


def answer_question(vector_store, question, video_id, chunks):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    if is_summary_question(question):
        # For summary, use more transcript chunks instead of only similarity search
        retrieved_docs = chunks[:15]
    else:
        retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(
        f"[{format_time(doc.metadata['start'])} - {format_time(doc.metadata['end'])}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = PromptTemplate(
        template="""
You are a helpful YouTube video assistant.

Answer using ONLY the transcript context below.
If the user asks for a summary, summarize the main idea from the context.
If the context does not contain the answer, say:
"I don't know based on this video."

Mention timestamp when useful.

Transcript context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    final_prompt = prompt.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    answer = llm.invoke(final_prompt)

    sources = []
    for doc in retrieved_docs:
        start_seconds = int(doc.metadata["start"])
        link = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"

        sources.append(
            {
                "time": f"{format_time(doc.metadata['start'])} - {format_time(doc.metadata['end'])}",
                "link": link,
                "text": doc.page_content[:300] + "...",
            }
        )

    return answer.content, sources


st.set_page_config(
    page_title="YouTube Video Chatbot",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube Video Chatbot")
st.write("Enter a YouTube video ID and ask questions based on its transcript.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

video_id = st.text_input(
    "Enter YouTube Video ID",
    placeholder="Example: Gfr50f6ZBvo",
)

question = st.text_input(
    "Ask a question about the video",
    placeholder="Example: Summarize the video in short",
)

if st.button("Process Video"):
    if not video_id:
        st.warning("Please enter a video ID.")
    else:
        try:
            with st.spinner("Fetching transcript and creating vector store..."):
                transcript_data = get_transcript(video_id)
                chunks = build_chunks_with_timestamps(transcript_data)

                vector_store = create_vector_store(chunks, video_id)

                st.session_state.vector_store = vector_store
                st.session_state.current_video_id = video_id
                st.session_state.chunks = chunks

            st.success(f"Video processed successfully. Total chunks: {len(chunks)}")

        except TranscriptsDisabled:
            st.error("No captions/transcript available for this video.")

        except NoTranscriptFound:
            st.error("No English transcript found for this video.")

        except Exception as e:
            st.error(f"Error: {e}")

if st.button("Ask"):
    if st.session_state.vector_store is None:
        st.warning("Please process the video first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer, sources = answer_question(
                st.session_state.vector_store,
                question,
                st.session_state.current_video_id,
                st.session_state.chunks,
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Relevant Timestamps")

        for source in sources:
            st.markdown(f"**Time:** {source['time']}")
            st.markdown(f"[Watch from here]({source['link']})")
            st.write(source["text"])
            st.divider()