import os
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults 
from data_models.models import RagToolSchema
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv
from pinecone import Pinecone
load_dotenv()
api_wrapper = PolygonAPIWrapper()
model_loader=ModelLoader()
config = load_config()





@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """ALWAYS use this tool first when the user asks to summarize, analyze, 
    explain, or find information from uploaded documents, files, reports, 
    or the knowledge base. This tool searches the internal vector database 
    of uploaded documents and returns relevant content."""

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    vector_store = PineconeVectorStore(index=pc.Index(config["vector_db"]["index_name"]), 
                            embedding= model_loader.load_embeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": config["retriever"]["top_k"] , "score_threshold": config["retriever"]["score_threshold"]},
    )
    retriever_result=retriever.invoke(question)
    
    return retriever_result

tavilytool = TavilySearch(
    max_results=config["tools"]["tavily"]["max_results"],
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    description="Search the web for latest news, market trends, or any real-time financial information."
    )

financials_tool = PolygonFinancials(api_wrapper=api_wrapper,description="Get financial statements and data for publicly traded companies using their stock ticker symbol."
)