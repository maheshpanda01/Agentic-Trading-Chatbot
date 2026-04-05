from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *

class BotState(TypedDict):
    messages: Annotated[list, add_messages]

class GraphBuilder:
    def __init__(self):
        self.model_loader=ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.tools = [retriever_tool, financials_tool, tavilytool]
        llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.llm_with_tools = llm_with_tools
        self.graph = None

    def _chatbot_node(self,state:BotState):
        return {"messages":[self.llm_with_tools.invoke(state["messages"])]}

    def build(self):
        graph_builder=StateGraph(BotState)
        graph_builder.add_node("chatnode",self._chatbot_node)
        tool_node=ToolNode(tools=self.tools)
        graph_builder.add_node("tools",tool_node)
        
        graph_builder.add_edge(START,"chatnode")
        graph_builder.add_conditional_edges("chatnode",tools_condition)
        graph_builder.add_edge("tools","chatnode")

        self.graph=graph_builder.compile()

    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph is not Built")
        return self.graph

