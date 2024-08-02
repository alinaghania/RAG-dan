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

RAW_DATA_PATH = Path().cwd().parent / "data" / 'V11_Argumentaire_Peugeot_2024.pdf'
PARSE_DATA_PATH = Path().cwd() / "data" / 'parsed_result_gpt.md'

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
            print("Loaded parsed document from local storage.")
    else:
        print("Parsed document not found in local storage. Please ensure the document is available.")
        documents = []

    return documents

def get_response(user_prompt, context):
    template = f"""
    
        Reformulate the answer be precise concise and to the point, and always answer based on the context provided.
        context: {context}
        Question: {user_prompt}
        """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    chain = prompt | model | StrOutputParser()

    return chain.stream({"context": itemgetter('context'), 
                         "user_prompt": itemgetter("user_prompt")
                         }
                        )

def get_elasticsearch_results(query, size=3):
    embedding = model.encode(query).tolist()

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

        image_results = es_client.search(index="images_peugeot_ev_pdf", body=es_query_images)
        return image_results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Error querying Elasticsearch: {e}")
        # Capture and log the response body for more details
        if hasattr(e, 'info'):
            logger.error(f"Elasticsearch error info: {e.info}")
        return [], []

def display_images(user_prompt, image_results):
    for hit in image_results:
        image_description = hit["_source"].get("description", "No description")
        if check_image_relevance(user_prompt, image_description):
            image_base64 = hit["_source"].get("base64", "")
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                st.image(image)
                break 

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
            "image_description": image_description,
            "context": context
        })
        return result["relevant_yes_no"] == "yes"
    except Exception as e:
        print(f"Exception: {e}")
        return False

def init_agent(query, vector_index):
    
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
            "Use ONLY IF you have specific questions over Peugeot electric car."
            "Do NOT use if you want to get a holistic summary of Peugeot electric car."
        ),
    )
        
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_peugeot_electric_car",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of Peugeot electric car."
            "Do NOT use if you have specific questions over Peugeot electric car."
        ),
    )
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_query_tool, summary_tool], 
        verbose=True
    )

    agent = AgentRunner(agent_worker)
    response = agent.query(query)
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
    documents = load_data(PARSE_DATA_PATH)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    print("Vector index loaded")
    
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
            context = init_agent(query=user_query, vector_index=vector_index)
            response = st.write_stream(get_response(user_query, context=context))

        st.session_state.chat_history.append(AIMessage(content=response))
        image_results = get_elasticsearch_results(user_query)
        display_images(user_query, image_results)

    
    st.markdown('<div class="QUESTIONS EXAMPLES">', unsafe_allow_html=True)
    questions = [
        "What's the price to charge?",
        "Hello, I want to know what are the best applications?",
        "What is the customer satisfaction rate among French users who switched to electric vehicles?",
        "Can you provide a brief history of Peugeot's electric vehicles?",
        "What are the main factors influencing the autonomy of an electric vehicle?",
        "Tell me what's the price to charge my e-208, and then the time to recharge on a born."
    ]
    
    for i, question in enumerate(questions, start=1):
        with st.expander(f"Question {i}"):
            st.write(question)  # Display the actual question when expanded