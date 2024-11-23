import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import openai
import json
from elevenlabs import generate, play
from elevenlabs import set_api_key
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import re
import string
import time
import pymongo




try:
    with open("config.json", "r") as json_file:
        data = json.load(json_file)
except:
    print("Unable to open JSON file.")
    exit()

url = data["keys"][0]["mongodb_url"]
# connects to the database/cluster

OPENAIAPI = data["keys"][0]["OAI_key"]

key = data["keys"][0]["EL_key"]
set_api_key(key)
#api key for 11elevenlabs




MODEL = "gpt-3.5-turbo"


SUPERPROMPT =  """
                Your name is Pythia
                You are a doctors AI assistant. Your job is to help capture symptoms from patients for doctors to review later. 
                You are a world renowned interviewer and your bedside manner is amazing. 
                You have a charming personality. You will not offer any medical diagnosis but you will make your patients feel comfortable as you gather information.
                You must display sympathy and empathy to make the patient feel comfortable and safe.
                You are careful not to overwhelm the patient and gather information slowly with follow up questions. You will deal with one of the patients reported symptoms at a time"""
# describes what and how the chatbot should respond

chat_history = [{"role": "system", "content": SUPERPROMPT}]
# remembers the entire conversation, so that it can feed it back to the chatbot to give it a memory
# by including the superprompt inside the chat history as being said by system, it tells the chatbot how it should respond





@click.command()
@click.option("--model", default="tiny", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
# default pause used to be 0.8, I think reducing the number can reduce waiting time
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
def main(model, english,verbose, energy, pause,dynamic_energy,save_file,device):
    temp_dir = tempfile.mkdtemp() if save_file else None
    #there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model).to(device)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    text = []
    record = {}
    id = 0
    threading.Thread(target=record_audio,
                     args=(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir)).start()
    threading.Thread(target=transcribe_forever,
                     args=(audio_queue, result_queue, audio_model, english, verbose, save_file)).start()




    print("Hello there, what is your name?")
    chat_history.append({"role": "assistant", "content": "Hello there, what is your name?"})
    audio = generate(
        text="Hello there, what is your name?",
        voice="Bella",
        model='eleven_multilingual_v1'
    )
    play(audio)
    name = result_queue.get()
    name = str(name).strip(".")
    myclient = pymongo.MongoClient(url)
    mydb = myclient["Records"]
    mycol = mydb["Patient: " + name]
    # introduction = chat asking for a name, name is then used to create a file on the database


    chat_history.append({"role": "user", "content":  name})

    print("Nice to meet you " + name + "! How can I help you?")
    chat_history.append({"role": "assistant", "content": "Nice to meet you " + name + "! How can I help you?"})
    audio = generate(
        text="Nice to meet you " + name + "! How can I help you?",
        voice="Bella",
        model='eleven_multilingual_v1'
    )
    play(audio)


    while True: 
        user_resp = result_queue.get() #Check if there is a next one

        chat_resp = chatbot(user_resp) # chatbot function
        audio = generate(  # Generate from voice AI
            text=chat_resp,
            voice="Bella",
            model='eleven_multilingual_v1'
        )
        play(audio)
        date = time.asctime()
        date = time.strftime("%y%m%d%H%M")
        # print(date)
        my_file = "./Chat_logs/Chat_history"+date+".txt"
        # print(my_file)
        with open(my_file, "w") as outfile:
            json.dump(chat_history, outfile, indent=4)

        # code below for identifying and logging relevant data

        text.clear()
        text.append(user_resp)


        if float(classify(text)) >= 0.5:
            #print("irrelevant")
            date = time.strftime("%H:%M %y/%m/%d")
            # prevent repetition in id
            nb = mycol.count_documents({})
            if nb == 0:
                id = 0
            else:
                id = nb
            record.clear()

            # send text to mongo db( online database to)
            record = {"_id": id, "time": date, "discussion": str(text)}
            mycol.insert_one(record)

        #else:
            #print("relevant")



# the while loop is where most of the things that we see happen, it's responsible for printing most of the messages we see, saving the messages in the txt file and generating most of the audio for the TTS



def chatbot(patient_says):
    # patient_says is type string, represents what the person is saying
    # function returns object of type str, chatbot_says is what the chatbot responds to the patient
    user_content = {"role": "user", "content": patient_says}
    chat_history.append(user_content)
    openai.api_key = OPENAIAPI
    print("You said: " + patient_says)
    response = openai.ChatCompletion.create(
        model= MODEL,
        messages=chat_history,
        # Feed it back the entire chat history, including the superprompt and the new message, to which it will respond
        temperature=0,
    )
    chat_history.append(response['choices'][0]['message'])
    # chat_history.append functions add the latest message to the chat history
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']
# prints and returns only the response that we want




# functions for whisper mic below
def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        i = 0
        while True:
            #get and save audio to wav file
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                audio_data = filename
            else:
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1

def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data,language='english')
        else:
            result = audio_model.transcribe(audio_data)

        if not verbose:
            predicted_text = result["text"]
            result_queue.put_nowait(predicted_text)
        else:
            result_queue.put_nowait(result)

        if save_file:
            os.remove(audio_data)



def classification():

    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    max_features = 10000
    sequence_length = 250

    vectorize_layer = keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
   

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    embedding_dim = 16

    # building the model
    model = tf.keras.Sequential([
        keras.layers.Embedding(max_features + 1, embedding_dim),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)])

    model.summary()

    # loss function and optimizer
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    # train the model
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

#activate function when needed
    def evaluate_plot():
        # evaluate the model
        loss, accuracy = model.evaluate(val_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        # plot accuracy and lost over time
        history_dict = history.history
        history_dict.keys()

        accuracy = history_dict['binary_accuracy']
        val_accuracy = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(accuracy) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()
    #evaluate_plot()

    # export model
    global export_model
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )



def classify(text):
    prediction = export_model.predict(text)
    return (prediction)


if __name__ == "__main__":
    classification()
    main()