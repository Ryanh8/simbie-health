from langgraph.graph import StateGraph, MessagesState, END, START

from speech_recognition import record_audio_until_silence
from text_to_speech import play_audio
from langgraph.checkpoint.memory import MemorySaver
from chatbot import create_graph
from chatbot1 import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.pregel.remote import RemoteGraph

load_dotenv()
memory = MemorySaver()

local_url = "http://localhost:3000"
graph_name = "chatbot1"
remote_graph = RemoteGraph(graph_name, url=local_url)
# Define parent graph
builder = StateGraph(MessagesState)
# Add remote graph directly as a node
builder.add_node("audio_input", record_audio_until_silence)
builder.add_node("agent", graph)
builder.add_node("audio_output", play_audio)
builder.add_edge(START, "audio_input")
builder.add_edge("audio_input", "agent")
builder.add_edge("agent", "audio_output")
builder.add_edge("audio_output", "audio_input")
audio_graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

for chunk in audio_graph.stream(
    input={"messages": HumanMessage(content="Follow the user's instructions:")},
    stream_mode="values",
    config=config,
):
    chunk["messages"][-1].pretty_print()
