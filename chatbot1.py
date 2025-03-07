import datetime
from fuzzywuzzy import process
from typing import Annotated, Any, Dict
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
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def check_supported_banks(bank_name: str) -> Dict[str, Any]:
    """
    Checks if a given bank name is supported and returns a response with a status and reasoning.
    Parameters:
    bank_name (str): The name of the bank to check.
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "status" (str): "true" if the bank is accepted, "possible" if it is EQBank (with conditions), or "false" if unsupported.
            - "reasoning" (str): Explanation of the status.
    """
    supported_banks = {
        "RBC Royal Bank",
        "TD",
        "RBC",
        "BMO Bank of Montreal",
        "ATB",
        "TD Canada Trust",
        "Scotiabank",
        "Servus CU",
        "KOHO",
        "PC Financial",
        "Tangerine - Personal",
        "Simplii Financial",
        "Laurentian Bank",
        "Simplii",
        "National Bank of Canada",
        "BMO",
        "CIBC",
        "ATB Online - Personal",
        "Scotia",
        "Tangerine",
        "National",
        "Servus Credit Union - Personal Online Banking",
        "Vancity",
        "Laurentienne",
    }

    special_cases = {
        "EQBank": "Pushes allowed for AllStar+ users who previously repaid voluntarily.",
        "Desjardins": "Some debit cards are not supported — agent to troubleshoot & validate.",
    }
    best_match, score = process.extractOne(
        bank_name, supported_banks.union(special_cases.keys())
    )

    if best_match in supported_banks and score > 85:
        return {"status": "true", "reasoning": "Bank is accepted."}
    elif best_match in special_cases and score > 85:
        return {"status": "possible", "reasoning": special_cases[best_match]}
    else:
        return {
            "status": "false",
            "reasoning": "Bank is not supported, do not suggest that Bree will be adding support for the bank in the future.",
        }


class Chatbot1:
    def __init__(self):
        load_dotenv()
        self.memory = MemorySaver()
        self.openai_client = OpenAI()
        self.llm = ChatOpenAI(model="gpt-4")
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools([check_supported_banks])
        self.agent = create_react_agent(
            model=self.llm, tools=[check_supported_banks], checkpointer=self.memory
        )

    def chatbot(self, state: State):
        """Process messages using the LLM with tools"""
        print("this is the state", state["messages"])
        response = self.agent.invoke(state)

        print("this is the response", response)
        response["messages"][-1] = HumanMessage(
            content=response["messages"][-1].content, name="agent"
        )
        return {"messages": response["messages"][-1]}

    def create_graph(self, state: MessagesState):
        """Create and return the compiled graph"""
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile(checkpointer=self.memory)


# Create instance for use in main.py
chatbot1 = Chatbot1()
# Export the graph creation function
graph = chatbot1.create_graph
