import os
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()
# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))


def play_audio(state: MessagesState):
    """Plays the audio response from the remote graph with ElevenLabs."""

    # Response from the agent
    response = state["messages"][-1]

    # Prepare text by replacing ** with empty strings
    cleaned_text = response.content.replace("**", "")

    # Call text_to_speech API with turbo model for low latency
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_44100_128",
        text=cleaned_text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
        optimize_streaming_latency=4,
    )

    # Play the audio back
    play(response)


def play_intermediate_response(message: str):
    """Plays an intermediate response while waiting for tool execution"""
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_44100_128",
        text=message,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
        optimize_streaming_latency=4,
    )
    play(response)
