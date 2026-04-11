from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *

class BotState(TypedDict):
    messages: Annotated[list, add_messages]

class GraphBuilder:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.tools = [retriever_tool, financials_tool, tavilytool]
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.graph = None

    def _chatbot_node(self, state: BotState):
        system = SystemMessage(content="""You are a trading research assistant with memory of the full conversation.
        You have access to these tools:
        - retriever_tool: ALWAYS use this first when the user asks to summarize, analyze,
          explain, or find information from uploaded documents, files, reports, or the knowledge base.
        - tavilytool: Use for real-time web search, market news, and current financial trends.
        - financials_tool: Use for stock financial statements using ticker symbols like AAPL, TSLA, MSFT.

        Always remember previous messages in the conversation and maintain context.""")

        messages = [system] + state["messages"]
        return {"messages": [self.llm_with_tools.invoke(messages)]}

    def build(self):
        graph_builder = StateGraph(BotState)
        graph_builder.add_node("chatnode", self._chatbot_node)

        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "chatnode")
        graph_builder.add_conditional_edges("chatnode", tools_condition)
        graph_builder.add_edge("tools", "chatnode")

        checkpointer = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph is not built. Call build() first.")
        return self.graph
