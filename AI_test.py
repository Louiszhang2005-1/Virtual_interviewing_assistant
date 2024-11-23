"""
Hackathon Mari Hacks
Daniel Tang
Audrey Liu Bai
Louis Zhang
Yufei Dong
"""


# Imports

from openai import OpenAI
import random
import pyaudio
import numpy as np
from scipy.io import wavfile
import os
import playsound

# Install this on the terminal before running
# pip install SpeechRecognition
# pip install openai
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install portaudio
# pip install pyaudio
# pip install wheel
# pip install playsound

# record audio and save to WAVE file
class Recorder():
    def __init__(self, filename, question):
        """define variables foraudio recorder."""
        self.audio_format = pyaudio.paInt16  # store sound data in 16bit binary string
        self.channels = 1  # number of audio streams
        self.sample_rate = 44100  # number of frames per second
        self.chunk = int(0.03*self.sample_rate)  # number of frames in the buffer
        self.filename = filename
        
        # Control the specific time given for different questions
        if questions[question] == 0:
            self.timer = 8
        elif questions[question] == 1:
            self.timer = 10
        elif questions[question] == 2:
            self.timer = 15
        elif questions[question] == 3:
            self.timer = 20
        

    def record(self):
        """to record audio for self.timer amount of seconds and prints the time passed every interval of seconds."""
        recorded_data = []
        p = pyaudio.PyAudio()  # instantiate PyAudio

        stream = p.open(format=self.audio_format, channels=self.channels,
                        rate=self.sample_rate, input=True,
                        frames_per_buffer=self.chunk)  # open a stream with parameters to write/read an audio file
        
        time = 0  # current time
        for i in range(0, int(self.sample_rate/self.chunk * self.timer)):  
            # in every iteration, self.chunk bytes are recorded
            # each second self.sample_rate samples are recorded
            # number of chunks/second * number of seconds
            # number of chunks we record total

            interval = 1 # prints every interval seconds
            if i % (interval * int(self.sample_rate/self.chunk)) == 0:
                print(time)
                time += interval
            data = stream.read(self.chunk)  # read the recorded data
            recorded_data.append(data)  # add data
            
        print("Recording stopped.")
        # stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        # convert recorded data to numpy array
        recorded_data = [np.frombuffer(frame, dtype=np.int16) for frame in recorded_data]
        wav = np.concatenate(recorded_data, axis=0)  # combine the arrays together
        wavfile.write(self.filename, self.sample_rate, wav)  # create wavfile
        print("New wav file created.")


    def listen(self):
        """create a new wav file and delete the old one (if any)."""
        print("You have", self.timer, "seconds to answer this question.")
        if os.path.exists("answer.wav"):
            os.remove("answer.wav")  #remove previous audio file
        self.record()  #start recording

SUPERPROMPT =  """
                Your name is Pythia
                You are a AI assistant. You are giving questions to the interviewee.
                You are a world renowned interviewer and your bedside manner is amazing. 
                You have a charming personality. You will offer advice on personality display.
                You must display sympathy and empathy to make the interviewee feel comfortable and safe.
                You are careful not to overwhelm the interviewee slowly with follow up questions. You will deal with one question at a time"""
                
# describes what and how the chatbot should respond

# Questions

questions = {
    "How much experience do you have?": 0,
    "Describe yourself in three words.": 1,
    "What is your biggest fear?": 0,
    "Give me three of your strengths and tell me about your biggest strength.": 2,
    "What is your major weakness?": 0,
    "Tell me about your education.": 1,
    "What do you think about books?": 1,
    "Tell me a bit about yourself?": 2,
    "How did you get into programming?": 2,
    "How was your childhood?": 2,
    "What is your worst workplace experience?": 3,
    "Why do you want to work here?": 2,
    "Give an example of how you solved a conflict in the past.": 3,
    "Tell me about one time you failed, and what did you learn from it?": 3
}

# OPENAI with API_KEY

OPENAI_API_KEY="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"

client = OpenAI(api_key=OPENAI_API_KEY)

def get_response(prompt):
    """Get chat gpt to respond to the message"""
    response = client.chat.completions.create(  # Creates a chat with the bot
        model = "gpt-3.5-turbo",
        messages = [
    {"role": "assistant", "content": prompt}, # Assigning a role and a prompt
]
    )
    return response.choices[0].message

user_answers = [] # Creates a list containing the users answers in case we need to use that sort of data

def get_random_question():
    """Return a random list of questions in a list of questions"""
    return random.choice(list(questions.keys()))  # We use the .keys since questions is a dictionnary which we convert to a list with the questions

# Number of questions to ask => 2
   
num_questions_to_ask = 2
total_num_questions = len(questions)


while len(questions)!= total_num_questions - num_questions_to_ask:
    random_question = get_random_question() # intend to be displayed on website
    print(random_question)

# Languages
#to finalize
language = "English"  # default language is english
lang = input("English or Français: ")  # two choices
if lang.isalpha():  #valid input
    if lang.lower() == "english" or lang.lower() == "french":  #update language
        language = lang.capitalize()


    
    # Text to speech converter: question
    
    if os.path.exists("question.mp3"):
        os.remove("question.mp3")  # remove previous question audio file
        
    question = client.audio.speech.create(  # Create audio speech
    model="tts-1",
    voice="fable",
    input = random_question
    )
    question.stream_to_file("question.mp3")  # create mp3 file with audio
    q_player = playsound.playsound("question.mp3", True)  # play mp3 file
    
    
        
    recorder = Recorder("answer.wav", random_question)  # name of output file
    recorder.listen()  # record

    # Speech to text convertion

    MODEL = "gpt-3.5-turbo"
    client = OpenAI(api_key=OPENAI_API_KEY)

    audio_file= open("answer.wav", "rb")  
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format = "text"
    )
        
    user_input = transcript
    prompt = f"Give feedback the user's answer: {user_input} to this question: {random_question}. (in an interview context)"
    response = get_response(prompt)
    print(response.content)
    
    # text to speech converter: response
    if os.path.exists("response.mp3"):
        os.remove("response.mp3")  # remove previous response audio file
        
    response = client.audio.speech.create(
    model="tts-1",
    voice="fable",
    input = response.content
    )
    response.stream_to_file("response.mp3")  # create mp3 file with audio
    r_player = playsound.playsound("response.mp3", True)  # play mp3 file
    

    user_answers.append(user_input)
    del(questions[random_question])