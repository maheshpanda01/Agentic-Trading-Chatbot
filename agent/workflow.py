from typing import TypedDict,Annotated
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.prebuilt.tool_node import tools_condition,ToolNode
from utils.model_loaders import ModelLoader
from toolkit.tools import *


class State(TypedDict):
    pass








class GraphBuilder:
    def __init__(self):
        pass

    def _chatbot_node(self,state:State):
        pass

    def build():
        pass

    def get_graph():
        pass
