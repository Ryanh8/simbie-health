import io
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from openai import OpenAI

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
openai_client = OpenAI()
SAMPLE_RATE = 16000  # Adequate for human voice frequency
THRESHOLD = 500  # Silence detection threshold (adjust if needed)
SILENCE_DURATION = 1.5  # Duration (seconds) of silence before stopping
CHUNK_SIZE = 1024  # Number of frames per audio chunk


def record_audio_until_silence(state: MessagesState):
    """Waits for the user to start speaking, records the audio, and stops after detecting silence."""

    audio_data = []  # List to store audio chunks
    silent_chunks = 0  # Counter for silent chunks
    started_recording = False  # Flag to track if recording has started

    def record_audio():
        """Continuously records audio, waiting for the user to start speaking."""
        nonlocal silent_chunks, audio_data, started_recording

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        ) as stream:
            print("Waiting for you to start speaking...")

            # Keep waiting indefinitely for the user to start talking
            while not started_recording:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Check if there is voice input
                if np.abs(audio_array).max() > THRESHOLD:
                    started_recording = True
                    print("Voice detected. Recording started.")
                    audio_data.append(audio_chunk)
                    break

            # Start recording once voice is detected
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_data.append(audio_chunk)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Detect silence after user has finished speaking
                if np.abs(audio_array).max() < THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Reset if sound is detected

                # Stop if silence is detected for the specified duration
                if silent_chunks > (SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE):
                    print("Silence detected. Stopping recording.")
                    break

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    recording_thread.join()

    # Stack all audio chunks into a single NumPy array and write to file
    audio_data = np.concatenate(audio_data, axis=0)

    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    write(
        audio_bytes, SAMPLE_RATE, audio_data
    )  # Use scipy's write function to save to BytesIO
    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  # Set a filename for the in-memory file

    # Transcribe via Whisper
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_bytes, language="en"
    )

    # Print the transcription
    print("Here is the transcription:", transcription.text)

    # Write to messages
    return {"messages": [HumanMessage(content=transcription.text)]}
