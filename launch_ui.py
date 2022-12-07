#imports
import time
import os
import gradio as gr
import functions
import tts_models_manager
from datetime import datetime
from TTS.utils.synthesizer import Synthesizer
from IPython.display import display
#global variables
audio_dir = "ui_outputs"
#configs
model_path, config_path, model_item = tts_models_manager.fetchLanguageModel("en")
synthesizer = Synthesizer(model_path, config_path, None, None, None)

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
    
def TTSPersonalModels(name):
  return f"Hello, {name}!"
    
def coquiDefaultModel(givenText = '你好吗？我很好'):
    localTimestamp = str(int(time.time()))
    filename = str(localTimestamp) + ".wav"    
    wavs = synthesizer.tts(givenText)
    filepath = os.path.join(audio_dir, filename) # create the file path for the wav file
    synthesizer.save_wav(wavs, filepath) # save the wav file to the file path
    return audio_dir + "/" + str(localTimestamp) + ".wav"

def coquiSingleAudio(givenAudio):
    localTimestamp = str(int(time.time()))
    filename = str(localTimestamp) + ".wav"    
    wavs = synthesizer.tts(givenAudio)
    # create the file path for the wav file
    filepath = os.path.join(audio_dir, filename)
    # save the wav file to the file path
    synthesizer.save_wav(wavs, filepath)    
    #cmd = "$ tts --text \"{}\" --model_name \"<model_type>/<language>/<dataset>/<model_name>\" --{}/{}.wav".format(givenText, audio_dir, str(localTimestamp))
    ## save wav to audio_dir
    #execute the command
    #subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_dir + "/" + str(localTimestamp) + ".wav"

def coquiBatchAudio(input_batch_directory, output_batch_directory):
    localTimestamp = str(int(time.time()))
    filename = str(localTimestamp) + ".wav"    
    #wavs = synthesizer.tts(givenAudio)
    # create the file path for the wav file
    filepath = os.path.join(audio_dir, filename)
    # save the wav file to the file path
    #synthesizer.save_wav(wavs, filepath)    
    #cmd = "$ tts --text \"{}\" --model_name \"<model_type>/<language>/<dataset>/<model_name>\" --{}/{}.wav".format(givenText, audio_dir, str(localTimestamp))
    ## save wav to audio_dir
    #execute the command
    #subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_dir + "/" + str(localTimestamp) + ".wav"

# Create a gradio interface for the greeting function
coqui_ui_interface_000 = gr.Interface(
    fn=coquiDefaultModel, # the function to use
    inputs="text", # the input type # the input type
    examples=[["Hello, world!"], ["你好吗？我很好"]], # examples to show in the UI
    outputs="audio", # the output type
    title="Coqui TTS", # the title of the UI
    description="Enter your text and get a audio file message" # brief description of the function
)
coqui_ui_interface_001 = gr.Interface(
    fn=coquiSingleAudio, # the function to use
    inputs="audio", # the input type # the input type
    outputs="audio", # the output type
    title="Coqui TTS", # the title of the UI
    description="Enter your text and get a audio file message" # brief description of the function
)
coqui_ui_interface_002 = gr.Interface(
    fn=coquiBatchAudio, # the function to use
    inputs="text", # the input type # the input type
    outputs="audio", # the output type
    title="Coqui TTS", # the title of the UI
    description="Enter your text and get a audio file message" # brief description of the function
)
coqui_ui = gr.TabbedInterface([coqui_ui_interface_000, coqui_ui_interface_001, coqui_ui_interface_002], ["Text-to-speech", "Train Model Audio", "Train model on Batch"])
# Display the UI
coqui_ui.launch(inline=True)