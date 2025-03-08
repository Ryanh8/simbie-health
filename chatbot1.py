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
from yaml import safe_load
import yaml
from langchain_core.messages import SystemMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_patient_info(patient_identifier: str) -> Dict[str, Any]:
    """
    Retrieves patient information from the system.

    Args:
        patient_identifier (str): Could be patient ID, phone number, or email

    Returns:
        Dict containing patient information or error message
    """
    try:
        # Mock API call - replace with your actual patient database API
        # Example return structure
        return {
            "status": "success",
            "patient_data": {
                "patient_id": "12345",
                "name": "John Doe",
                "date_of_birth": "1980-01-01",
                "phone": "123-456-7890",
                "email": "john@example.com",
                "insurance_provider": "Blue Cross",
                "insurance_id": "INS123456",
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving patient information: {str(e)}",
        }


@tool
def schedule_appointment(
    patient_id: str,
    appointment_type: str,
    preferred_date: str = None,
    preferred_time: str = None,
) -> Dict[str, Any]:
    """
    Books an appointment for the patient.

    Args:
        patient_id (str): Patient's unique identifier
        appointment_type (str): Type of appointment requested
        preferred_date (str, optional): Preferred date (YYYY-MM-DD)
        preferred_time (str, optional): Preferred time (HH:MM)

    Returns:
        Dict containing appointment details or error message
    """
    try:
        # Mock API call - replace with your actual scheduling API
        return {
            "status": "success",
            "appointment": {
                "appointment_id": "APT789012",
                "date": "2024-03-20",
                "time": "14:30",
                "doctor": "Dr. Smith",
                "location": "Main Clinic",
                "notes": "Please arrive 15 minutes early",
            },
        }
    except Exception as e:
        return {"status": "error", "message": f"Error scheduling appointment: {str(e)}"}


class Chatbot1:
    def __init__(self):
        load_dotenv()
        self.memory = MemorySaver()
        self.openai_client = OpenAI()
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [get_patient_info, schedule_appointment]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # self.yaml_file = safe_load(open("prompts/chatbot.yaml", "r"))
        with open("prompts/chatbot.yaml") as stream:
            self.system_prompt = yaml.safe_load(stream)["format"]

        print(self.system_prompt)

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=self.system_prompt),
        )

    def chatbot(self, state: State):
        """Process messages using the LLM with tools"""
        print("this is the state", state["messages"])
        response = self.agent.invoke(state)
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
