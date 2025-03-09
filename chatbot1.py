import datetime
import os
from browser_use import Agent
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
import yaml
from langchain_core.messages import SystemMessage
from playwright.sync_api import Playwright, sync_playwright
from browserbase import Browserbase
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
import asyncio
from langchain_core.callbacks import AsyncCallbackHandler


class State(TypedDict):
    messages: Annotated[list, add_messages]


load_dotenv()
bb = os.getenv("BROWSERBASE_API_TOKEN")


@tool
async def run() -> Dict[str, Any]:
    """
    This tool is used to run a browser session.
    Use the response from this tool to retrieve information from the page.
    """
    # # Create a session on Browserbase
    # playwright = sync_playwright()
    # session = bb.sessions.create(project_id=os.getenv("BROWSERBASE_PROJECT_ID"))

    # # Connect to the remote session
    # chromium = playwright.chromium
    # browser = chromium.connect_over_cdp(session.connect_url)
    # context = browser.contexts[0]
    # page = context.pages[0]

    # try:
    #     # Execute Playwright actions on the remote browser tab
    #     page.goto("https://news.ycombinator.com/")
    #     page_title = page.title()
    #     print(f"Page title: {page_title}")
    #     page.screenshot(path="screenshot.png")
    # finally:
    #     page.close()
    #     browser.close()

    # print(f"Done! View replay at https://browserbase.com/sessions/{session.id}")

    # sync_browser = create_sync_playwright_browser()
    # toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    # tools = toolkit.get_tools()
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # prompt = hub.pull("hwchase17/openai-tools-agent")
    # agent = create_openai_tools_agent(llm, tools, prompt)
    # # agent = create_react_agent(llm, tools, prompt) #Use this if using hwchase17/react prompt
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # command = {
    #     "input": "Start from the page https://www.google.com/, in the search bar type 'what is the weather in boston' and press enter. Once you get the results, return the first result."
    # }
    # response = agent_executor.invoke(command)
    # return response

    agent = Agent(
        task="Book some time on my calendar on the link https://calendly.com/ryanhu20/30min?month=2025-03 for any random day with a random time and name and use the email ryanhu20@gmail.com to book the time",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()


@tool
async def get_patient_info(patient_identifier: str) -> Dict[str, Any]:
    """Async version of get_patient_info"""
    try:
        # Simulate async API call
        await asyncio.sleep(0.1)  # Simulate network delay
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
async def schedule_appointment(
    patient_id: str,
    appointment_type: str,
    preferred_date: str = None,
    preferred_time: str = None,
) -> Dict[str, Any]:
    """Async version of schedule_appointment"""
    try:
        # Simulate async API call
        await asyncio.sleep(0.1)  # Simulate network delay
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
        self.tools = [get_patient_info, schedule_appointment, run]
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

    async def chatbot(self, state: State):
        """Process messages using the LLM with tools"""
        print("this is the state", state["messages"])
        response = await self.agent.ainvoke(state)
        response["messages"][-1] = HumanMessage(
            content=response["messages"][-1].content, name="agent"
        )
        return {"messages": response["messages"][-1]}

    def create_graph(self, state: State):
        """Create and return the compiled graph"""
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile(checkpointer=self.memory)


# Create instance for use in main.py
