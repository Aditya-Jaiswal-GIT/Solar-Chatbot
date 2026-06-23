import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(
    BASE_DIR,
    "data",
    "PM_Solar_Yojna.pdf"
)

load_dotenv()



# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Solar AI Assistant",
    page_icon="☀️",
    layout="wide"
)



# ==================================
# CSS
# ==================================

st.markdown(
"""
<style>

.stApp{

background:
radial-gradient(
circle at top left,
#1e293b,
#020617 50%
);

}


.main-title{

font-size:52px;
font-weight:900;

background:
linear-gradient(
90deg,
#facc15,
#fb923c,
#fde047
);

-webkit-background-clip:text;
color:transparent;

animation:fadeDown 1s;

}



.subtitle{

font-size:20px;
color:#94a3b8;

}



section[data-testid="stSidebar"]{

background:
linear-gradient(
180deg,
#111827,
#020617
);

}




.status{

padding:15px;

border-radius:18px;

background:
rgba(34,197,94,.15);

border:
1px solid #22c55e;

color:#4ade80;

font-weight:bold;

}



.chat-user{


background:
linear-gradient(
135deg,
#2563eb,
#1d4ed8
);


padding:18px;

border-radius:
25px 25px 5px 25px;

margin:15px 0;

color:white;

}



.chat-ai{


background:
rgba(31,41,55,.85);

padding:18px;

border-radius:
25px 25px 25px 5px;

margin:15px 0;

border:
1px solid #374151;

}



button{

border-radius:15px !important;

background:
linear-gradient(
135deg,
#f59e0b,
#ea580c
)!important;

color:white!important;

font-weight:bold!important;

}




@keyframes fadeDown{

from{

opacity:0;
transform:translateY(-30px);

}

to{

opacity:1;

}

}


</style>

""",
unsafe_allow_html=True
)





# ==================================
# HEADER
# ==================================

st.markdown(

"""
<div class="main-title">

☀️ Solar Intelligence Assistant

</div>


<div class="subtitle">

AI Powered Government Scheme Knowledge System

<br>

⚡ RAG + LangChain + Groq + Chroma DB

</div>

<br>

""",

unsafe_allow_html=True

)





# ==================================
# VECTOR DATABASE
# ==================================

VECTOR_DB_PATH = "solar_vector_db"



@st.cache_resource
def load_database():


    embeddings = GoogleGenerativeAIEmbeddings(
        model= "gemini-embedding-2",
        api_key=st.secrets['GEMINI_API']
    )



    # -----------------------------
    # LOAD EXISTING DATABASE
    # -----------------------------

    if os.path.exists(VECTOR_DB_PATH):


        st.success(
            "📂 Existing Vector Database Loaded"
        )


        db = Chroma(

            persist_directory=VECTOR_DB_PATH,

            embedding_function=embeddings

        )


        return db




    # -----------------------------
    # CREATE DATABASE
    # -----------------------------


    st.warning(
        "⚙️ Creating New Vector Database..."
    )



    loader = PyPDFLoader(

       pdf_path

    )



    documents = loader.load()



    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=50

    )



    chunks = splitter.split_documents(

        documents

    )



    db = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=VECTOR_DB_PATH

    )



    st.success(

        "✅ Vector Database Created"

    )


    return db





db = load_database()



retriever = db.as_retriever(

    search_kwargs={

        "k":5

    }

)





# ==================================
# LLM
# ==================================


llm = ChatGroq(

    model="openai/gpt-oss-120b",

    api_key=st.secrets['GROQ_API']

)



parser = StrOutputParser()



prompt = ChatPromptTemplate.from_template(

"""

You are Solar Government Scheme AI Assistant.


Answer only from given context.


Context:

{context}



Question:

{question}



Rules:

- Do not use external knowledge
- Give complete explanation
- Provide lists when needed

"""

)



chain = (

prompt

|

llm

|

parser

)







# ==================================
# SIDEBAR
# ==================================


with st.sidebar:


    st.title(
        "☀️ AI Control Center"
    )



    st.markdown(

    """

    <div class="status">

    🟢 Vector Database Ready

    </div>

    """,

    unsafe_allow_html=True

    )



    st.divider()



    st.write(
        "Embedding Model"
    )


    st.code(
        "nomic-embed-text"
    )



    st.write(
        "LLM"
    )


    st.code(
        "Groq GPT OSS 120B"
    )



    st.divider()



    if st.button(
        "🗑 Clear Chat"
    ):


        st.session_state.messages=[]

        st.rerun()







# ==================================
# CHAT MEMORY
# ==================================


if "messages" not in st.session_state:

    st.session_state.messages=[]





for msg in st.session_state.messages:



    if msg["role"]=="user":


        st.markdown(

        f"""

        <div class="chat-user">

        👤 {msg["content"]}

        </div>

        """,

        unsafe_allow_html=True

        )



    else:


        st.markdown(

        f"""

        <div class="chat-ai">

        🤖 {msg["content"]}

        </div>

        """,

        unsafe_allow_html=True

        )








# ==================================
# CHAT INPUT
# ==================================


question = st.chat_input(

    "Ask Solar Scheme Question..."

)



if question:



    st.session_state.messages.append(

        {

        "role":"user",

        "content":question

        }

    )




    st.markdown(

    f"""

    <div class="chat-user">

    👤 {question}

    </div>

    """,

    unsafe_allow_html=True

    )




    with st.status(

        "🔍 Searching Knowledge Base..."

    ):



        docs = retriever.invoke(

            question

        )



        context = "\n\n".join(

            [

            d.page_content

            for d in docs

            ]

        )



        time.sleep(1)



        answer = chain.invoke(

            {

            "context":context,

            "question":question

            }

        )





    st.markdown(

    f"""

    <div class="chat-ai">

    🤖 {answer}

    </div>

    """,

    unsafe_allow_html=True

    )





    st.session_state.messages.append(

        {

        "role":"assistant",

        "content":answer

        }

    )





    with st.expander(

        "📄 Retrieved PDF Context"

    ):

        st.write(context)