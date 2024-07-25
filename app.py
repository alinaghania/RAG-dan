### IMPORTS ###
import os

import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

### CONSTANTS ###

# Initialize Elasticsearch and OpenAI clients
es_client = Elasticsearch(
    "https://371e52c2ddc94eeda8d2dbeb8acc5645.us-central1.gcp.cloud.es.io:443",
    api_key=os.environ["elastic_host"]
)

st.set_page_config(page_title="RAG")
st.title("R.A.G - Ask me anything")

### FUNCTIONS ###

# Define the Elasticsearch query to retrieve the results
def get_elasticsearch_results(query, size=1):
    es_query = {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "title",
                                        "content"
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "standard": {
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "title",
                                        "content"
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        },
        "size": size
    }
    result = es_client.search(index="data_for_rag,documents", body=es_query)
    return result["hits"]["hits"]

# Define the fields to use for each index in Elasticsearch 
index_source_fields = {
    "data_for_rag": [
        "content"
    ],
    "documents": [
        "content"
    ]
}

# Format the context results from Elasticsearch
def format_context_results(results):
    context = ""
    for hit in results:
        source_field = index_source_fields.get(hit["_index"])[0]
        hit_context = hit["_source"][source_field]
        context += f"{hit_context}\n"
    return context

def get_response(user_prompt, chat_history):
    elasticsearch_context = get_elasticsearch_results(user_prompt)
    context = format_context_results(elasticsearch_context)
    
    template = f"""
        Instructions:
        
        - You are an assistant for question-answering tasks.
        - Answer questions truthfully and factually using only the information presented.
        - If you don't know the answer, just say that you don't know, don't make up an answer! and only use the information provided.
        
        - Use markdown format for code examples.
        - You are correct, factual, precise, and reliable.
        
        Answer the following questions considering the context and if you need the history of the conversation:
        Context:{context}
        Question: {user_prompt}
        History: {chat_history}
        Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o")
    chain = prompt | model | StrOutputParser()

    return chain.stream({"context": itemgetter('context'), 
                         "user_prompt": itemgetter("user_prompt"), 
                         "chat_history":chat_history
                         }
                        )

if __name__ == '__main__':
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=" Hello Dan, quelles informations ( sur le document)puis-je vous fournir aujourd'hui ?")
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            
            response = st.write_stream(get_response(user_query, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(content=response))

    
    st.markdown('<div class="QUESTIONS EXAMPLES">', unsafe_allow_html=True)
    questions = [
        "What's the price for a charge?",
        "Hello, I want to know what are the best applications?",
        "What is the customer satisfaction rate among French users who switched to electric vehicles?",
        "Can you provide a brief history of Peugeot's electric vehicles?",
        "What are the main factors influencing the autonomy of an electric vehicle?",
    ]
    
    for i, question in enumerate(questions, start=1):
        with st.expander(f"Question {i}"):
            st.write(question)  # Display the actual question when expanded
