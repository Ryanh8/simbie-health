from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from langgraph.graph import MessagesState, START, END

load_dotenv()
memory = MemorySaver()
openai_client = OpenAI()

llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_date_and_time() -> dict:
    """
    Call tool to fetch the current date and time from an API.
    """
    try:
        response = requests.get(
            "https://timeapi.io/api/Time/current/zone?timeZone=Europe/Brussels"
        )
        response.raise_for_status()
        data = response.json()
        return {"date_time": data["dateTime"], "timezone": data["timeZone"]}
    except requests.RequestException as e:
        return {"error": str(e)}


llm_with_tools = llm.bind_tools([get_date_and_time])


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def graph(state: MessagesState):
    # initiate the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    # Add START and END connections
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile(checkpointer=memory)
