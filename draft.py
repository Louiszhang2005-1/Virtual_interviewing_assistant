#text to speech converter
#run this in terminal
#pip install playsound

from openai import OpenAI
import playsound

client = OpenAI(api_key="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")
response = "Hello"  #remove this

response = client.audio.speech.create(
  model="tts-1",
  voice="fable",
  input = response #change to response.content
)
response.stream_to_file("response.mp3")

player = playsound.playsound("response.mp3", True)
