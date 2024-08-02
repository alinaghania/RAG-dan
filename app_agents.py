### IMPORTS ###
import base64
import json
import logging
import os
from io import BytesIO
from operator import itemgetter
from pathlib import Path
from typing import List, Optional

import nest_asyncio
import streamlit as st
import urllib3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from PIL import Image
from sentence_transformers import SentenceTransformer

nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAW_DATA_PATH = Path().cwd().parent / "data" / 'V11_Argumentaire_Peugeot_2024.pdf'
PARSE_DATA_PATH = Path().cwd() / "data" / "parsed_doc_GPT" / 'parsed_result_gpt.md'


### CONSTANTS ###
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY_3=os.getenv("LLAMA_CLOUD_API_KEY_3")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 1024
Settings.llm = OpenAI(model="gpt-4-turbo-2024-04-09")
# Initialize Elasticsearch and OpenAI clients
es_client = Elasticsearch(
    st.secrets["elasticsearch"]["url"],
    api_key=st.secrets["elasticsearch"]["api_key"],
    verify_certs=False,
    request_timeout=200,
    max_retries=10,
    retry_on_timeout=True
)
print("LLM / Embedding Models loaded")

### UTILS FUNCTION ###
def load_data(parsed_doc_path):
    
    if parsed_doc_path.exists():
        with open(parsed_doc_path, 'r', encoding='utf-8') as file:
            document_data = file.read()
            # Assuming the document data is in markdown format and needs to be converted to the appropriate format
            documents = [Document(text=document_data)]
    else:
        print("Parsed document not found in local storage. Please ensure the document is available.")
        documents = []

    return documents

def get_response(user_prompt, chat_history, context):
    template = f"""
        Reformulate the answer to be precise and to the point. Always answer based on the context provided.
        context: {context}
        Question: {user_prompt}
        History: {chat_history}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    chain = prompt | model | StrOutputParser()

    # Return the response generator for streaming
    response_generator = chain.stream({"context": context, "user_prompt": user_prompt, "chat_history": chat_history})
    
    return response_generator

def display_images(user_query, image_results, context):
    for hit in image_results:
        # Ensure hit is a dictionary and contains the '_source' key
        if isinstance(hit, dict) and "_source" in hit:
            image_description = hit["_source"].get("description", "No description")
            if check_image_relevance(user_query, image_description, context):
                image_base64 = hit["_source"].get("base64", "")
                if image_base64:
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(BytesIO(image_data))
                    st.image(image)
                    break  # Display only the first relevant image

def get_elasticsearch_results(response, size=3):
    # Encode the response
    response_embedding = model.encode(response).tolist()

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
                                "params": {"query_vector": response_embedding}
                            }
                        }
                    }
                ]
            }
        },
        "size": size
    }

    try:
        image_results = es_client.search(index="images_peugeot_ev_pdf", body=es_query_images)
        return image_results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Error querying Elasticsearch: {e}")
        if hasattr(e, 'info'):
            logger.error(f"Elasticsearch error info: {e.info}")
        return []




def check_image_relevance(user_prompt, image_description, context):
    class Relevant(BaseModel):
        relevant_yes_no: str = Field(description="yes or no")
        
    output_parser = JsonOutputParser(pydantic_object=Relevant)
 
    template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("Based on the following user query, image description and context if relevant, determine if the image is relevant to the user's query,Answer with 'yes' or 'no'. For example if the question is : Quel modele est en tête des ventes de véhicules électriques sur le marché français ?, the answer is : E-208, if you have an image of this model, return yes, but for a summary question don't need to return an image , so answer no{format_instructions}"),
            HumanMessagePromptTemplate.from_template("User query: {user_query}\nImage description: {image_description},\nContext: {context}")
        ],
        input_variables=["user_query", "image_description", "context"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    model = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    chain = template | model | output_parser

    try:
        result = chain.invoke({
            "user_query": user_prompt,
            "image_description": image_description,
            "context": context
        })
        return result["relevant_yes_no"] == "yes"
    except Exception as e:
        print(f"Exception: {e}")
        return False

st.cache_resource(show_spinner=False)
def init_agent(query):
    
    documents = load_data(PARSE_DATA_PATH)
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    summary_index = SummaryIndex(nodes)
    
    def vector_query(
        query: str,
        page_numbers: Optional[List[str]] = None,
    ) -> str:
        """Use to answer questions over the MetaGPT paper.
    
        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
    
    # Set up my vector query tool = classic RAG retriever
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_peugeot_electric_car",
        fn=vector_query,
        description=(
            "Useful for retrieving specific context from the document."
        ),
    )
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_peugeot_electric_car",
        query_engine=summary_query_engine,
        description=(
                        "Useful for summarization questions related to the document"

        ),
    )
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_query_tool, summary_tool], 
        verbose=True
    )

    agent = AgentRunner(agent_worker)
    response = agent.chat(query)
    return str(response)



### STREAMLIT APP ###
st.set_page_config(page_title="RAG")
st.title("R.A.G - Ask me anything")

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the fields to use for each index in Elasticsearch 
index_source_fields = {
    "peugeot_ev_pdf": ["content"],
    "images_peugeot_ev_pdf": ["description", "base64"]
}

            
if __name__ == '__main__':
    
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=" Hello Dan, quelles informations (sur le document) puis-je vous fournir aujourd'hui ? A.G.E.N.T.S")
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
            context = init_agent(query=user_query)
            response = st.write_stream(get_response(user_query, st.session_state.chat_history, context=context))

        st.session_state.chat_history.append(AIMessage(content=response))
        image_results = get_elasticsearch_results(str(response))
        display_images(user_query, image_results, context)

    st.markdown('<div class="QUESTIONS EXAMPLES">', unsafe_allow_html=True)
    questions = [
        "Résume moi le document",
        "La meilleure app pour vérifier l’état de la charge par exemple ?",
        "Pour la E-3008, quels sont les types de moteurs disponibles ?",
        "Tell me what's the price to charge my e-208, and then the time to recharge on a born.",
        "Quel est le modèle en tête des ventes de véhicules électriques sur le marché français ?",
    ]

    for question in questions:
        if st.button(question):
            # Simulate user query
            user_query = question
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                context = init_agent(query=user_query)
                response = st.write_stream(get_response(user_query, st.session_state.chat_history, context=context))

            st.session_state.chat_history.append(AIMessage(content=response))
            image_results = get_elasticsearch_results(str(response))
            display_images(user_query, image_results, context)

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
