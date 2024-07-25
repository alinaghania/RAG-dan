### IMPORTS ###
import os

import streamlit as st

from elasticsearch import Elasticsearch

# load_dotenv()
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

### CONSTANTS ###

# # Initialize Elasticsearch and OpenAI clients
# es_client = Elasticsearch(
#     "https://371e52c2ddc94eeda8d2dbeb8acc5645.us-central1.gcp.cloud.es.io:443",
#     api_key=os.environ["elastic_host"]
# )


es_client = Elasticsearch(
    st.secrets["elasticsearch"]["url"],
    api_key=st.secrets["elasticsearch"]["api_key"]
)
## key

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

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
                ## Instructions for the Assistant

                ### Overview
                - **Purpose**: You are designed to assist with question-answering tasks.
                - **Accuracy**: Answer questions based solely on the information presented.
                - **Admit Uncertainty**: If unsure, respond with "I don't know" rather than speculating.
                - **one or two sentences**: Keep responses concise and to the point.

                ### Formatting and Structure
                - **Markdown Usage**: Employ Markdown for code examples and to present structured data in tables.
                - **Precision and Reliability**: Ensure all responses are correct, factual, precise, and directly address the query.
                - **Conciseness**: Be brief and avoid unnecessary elaborations, just answer the question directly, if the user wants more information, they will ask,then you can delve deeper, so 1-2 sentences are enough.
                - **Avoid Incorrect Syntax**: Do not use incorrect markup like `\text`., never use this, always use markdown syntax.
                - **Use Tables for Structured Data**: When presenting structured data, use tables for clarity and organization.
                - **Use Code Blocks for Code**: For code examples, use code blocks to distinguish them from regular text.
                - **Use Bullet Points for Lists**: Use bullet points for lists to improve readability.
                - **Use Math Blocks for Formulas**: For mathematical formulas, use math blocks to ensure proper rendering.
                

                ### Engagement and Clarity
                - **Encourage Specifics**: Ask the user to clarify if the question is too broad.
                - **Keep the Conversation Going**: Always conclude responses with a contextually relevant question to encourage further dialogue.

                ### Example Responses

                **Incorrect Approach**:
                ```markdown
                Question: What is the price of a charge?
                Answer: The price for charging a vehicle depends on several factors including the capacity of the battery, the consumption rate, and the cost of electricity. You can estimate the cost of a full charge with the formula:
                \[ \text Capacity of the battery in kWh \times \text tariff per kWh \]
                Example for the E-208 with a 156 ch engine: Would you like more information on different energy costs?

                please answer like this for all the questions, like a conversation and not a lecture.
                **Correct Approach**:
                Question: What is the price of a charge?
                Answer: The price depends, do you have a specific vehicle in mind? Perhaps the average price, or for a model like the E-208?
                Follow-up Question: What is the price of a charge for the E-208?
                Response: The price is ...€ for a full charge. Would you like to know the cost for other distances or for other vehicles?
                Further Question: other km
                Response: Below is a table with the pricing for the E-208 for various distances:
                | Distance (km) | Price (€) |
                |---------------|-----------|
                | Example       | Example   |
                | Example       | Example   |
                | Example       | Example   |
                | Example       | Example   |

                Answer the following questions considering the context and if you need the history of the conversation:
                context: {context}
                Question: {user_prompt}
                History: {chat_history}
                Answer:
                
                
            
            """
            

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
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

# Updated CSS for background image
page_bg_img = '''
<style>
body {
background-image: url("https://upload.wikimedia.org/wikipedia/fr/9/9d/Peugeot_2021_Logo.svg");
background-size: contain;
background-repeat: no-repeat;
background-attachment: fixed;
background-position: center;
opacity: 0.95;
}

[data-testid="stAppViewContainer"] {
background: rgba(255, 255, 255, 0.7); /* Slight white overlay to keep the content readable */
}

[data-testid="stHeader"] {
background: rgba(0, 0, 0, 0);
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)