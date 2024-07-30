import os
import base64
from io import BytesIO
from typing import List

import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

### CONSTANTS ###

# Initialize Elasticsearch and OpenAI clients
es_client = Elasticsearch(
    st.secrets["elasticsearch"]["url"],
    api_key=st.secrets["elasticsearch"]["api_key"],
    verify_certs=False,
    request_timeout=200,
    max_retries=10,
    retry_on_timeout=True
)

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="RAG")
st.title("R.A.G - Ask me anything")

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the fields to use for each index in Elasticsearch 
index_source_fields = {
    "peugeot_ev_pdf": ["content"],
    "images_peugeot_ev_pdf": ["description", "base64"]
}

### FUNCTIONS ###

# Define the Elasticsearch query to retrieve the results
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_elasticsearch_results(query, size=3):
    embedding = model.encode(query).tolist()

    es_query_text = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "vector"}},  # Ensure the vector field exists
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": embedding}
                            }
                        }
                    }
                ]
            }
        },
        "size": size
    }

    es_query_images = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "vector"}},  # Ensure the vector field exists
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": embedding}
                            }
                        }
                    }
                ]
            }
        },
        "size": 3
    }

    try:
        text_results = es_client.search(index="peugeot_ev_pdf", body=es_query_text)
        image_results = es_client.search(index="images_peugeot_ev_pdf", body=es_query_images)
        return text_results["hits"]["hits"], image_results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Error querying Elasticsearch: {e}")
        # Capture and log the response body for more details
        if hasattr(e, 'info'):
            logger.error(f"Elasticsearch error info: {e.info}")
        return [], []

# Format the context results from Elasticsearch
def format_context_results(results):
    context = ""
    for hit in results:
        source_field = index_source_fields.get(hit["_index"])[0]
        hit_context = hit["_source"][source_field]
        context += f"{hit_context}\n"
    return context

# Define the Pydantic model for output parsing
class RelevanceOutput(BaseModel):
    relevant: str = Field(..., description="yes or no")

output_parser = JsonOutputParser(pydantic_object=RelevanceOutput)

def get_response(user_prompt, chat_history):
    text_results, image_results = get_elasticsearch_results(user_prompt)
    context = format_context_results(text_results)

    template = f"""
                ## Instructions for the Assistant

                ### Overview
                - **Purpose**: You are designed to assist with question-answering tasks.
                - **Accuracy**: Answer questions based solely on the information presented.
                - **Admit Uncertainty**: If unsure, respond with "I don't know" rather than speculating.
                - **one or two sentences**: Keep responses concise and to the point.
                - **don't answer if it's not in the context, and the question is in other topic **: Only use the information provided in the context.

                ### Formatting and Structure
                - **Markdown Usage**: Employ Markdown for code examples and to present structured data in tables.
                - **Precision and Reliability**: Ensure all responses are correct, factual, precise, and directly address the query.
                - **Conciseness**: Be brief and avoid unnecessary elaborations, just answer the question directly, if the user wants more information, they will ask,then you can delve deeper, so 1-2 sentences are enough.
                - **Avoid Incorrect Syntax**: Do not use incorrect markup like `\\text`., never use this, always use markdown syntax.
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
                \\[ \\text Capacity of the battery in kWh \\times \\text tariff per kWh \\]
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
    model = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    chain = prompt | model | StrOutputParser()

    # Convert chat history to list of messages
    chat_history_list = [{"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else {"role": "assistant", "content": msg.content} for msg in chat_history]

    return chain.stream({
        "context": context,
        "user_prompt": user_prompt,
        "chat_history": chat_history_list
    })

def check_image_relevance(user_prompt, image_description):
    class Relevant(BaseModel):
        relevant_yes_no: str = Field(description="yes or no")
        
    output_parser = JsonOutputParser(pydantic_object=Relevant)
 
    template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("Based on the following user query and image description, determine if the image is relevant to the user's query,Answer with 'yes' or 'no'. {format_instructions}"),
            HumanMessagePromptTemplate.from_template("User query: {user_query}\nImage description: {image_description}")
        ],
        input_variables=["user_query", "image_description"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    model = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    chain = template | model | output_parser

    try:
        result = chain.invoke({
            "user_query": user_prompt,
            "image_description": image_description
        })
        return result["relevant_yes_no"] == "yes"
    except Exception as e:
        print(f"Exception: {e}")
        return False

def display_images(user_prompt, image_results):
    for hit in image_results:
        image_description = hit["_source"].get("description", "No description")
        if check_image_relevance(user_prompt, image_description):
            image_base64 = hit["_source"].get("base64", "")
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                st.image(image)
                break  # Display only one relevant image

if __name__ == '__main__':
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello La team, quelles informations (sur le document) puis-je vous fournir aujourd'hui ?")
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

        text_results, image_results = get_elasticsearch_results(user_query)
        display_images(user_query, image_results)

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
