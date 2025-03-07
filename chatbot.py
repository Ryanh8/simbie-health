from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import MessagesState


class State(TypedDict):
    messages: Annotated[list, add_messages]


class Chatbot:
    def __init__(self):
        load_dotenv()
        self.memory = MemorySaver()
        self.openai_client = OpenAI()
        self.llm = ChatOpenAI(model="gpt-4")

        # Define tools
        self.tools = [self.get_date_and_time]

        # Create React agent
        self.agent = create_react_agent(
            model=self.llm, tools=self.tools, checkpointer=self.memory
        )
        self.llm_with_tools = self.llm.bind_tools([self.get_date_and_time])

    @tool
    def get_date_and_time(self) -> dict:
        """Call tool to fetch the current date and time from an API."""
        try:
            response = requests.get(
                "https://timeapi.io/api/Time/current/zone?timeZone=Europe/Brussels"
            )
            response.raise_for_status()
            data = response.json()
            return {"date_time": data["dateTime"], "timezone": data["timeZone"]}
        except requests.RequestException as e:
            return {"error": str(e)}

    def chatbot(self, state: State):
        """Process messages using the React agent"""
        print(state)
        # response = self.agent.invoke(state["messages"])
        # print(response)
        # return {"messages": state["messages"] + [response]}
        response = self.llm_with_tools.invoke(state["messages"])
        print(response)
        return {"messages": [response]}


chatbot = Chatbot()


def create_graph(state: MessagesState):
    """Create and return the compiled graph"""
    workflow = StateGraph(state)
    workflow.add_node("chatbot", chatbot.chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile(checkpointer=chatbot.memory)
