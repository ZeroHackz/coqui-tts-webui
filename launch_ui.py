#imports
import time
import os
import gradio as gr
import functions
import tts_models_manager
import numpy as np
from logmmse import logmmse
from datetime import datetime
from TTS.utils.synthesizer import Synthesizer
from IPython.display import display
#global variables
audio_dir = "ui_outputs"
user_has_gpu = False;
#configs
model_path, config_path, model_item = tts_models_manager.fetchLanguageModel("en")
synthesizer = Synthesizer(model_path, config_path, None, None, None)
#paths and directories
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
#SUPPORTED FEATURES
def coquiDefaultModel(givenText):
    localTimestamp = str(int(time.time()))
    filename = str(localTimestamp) + ".wav"    
    wavs = synthesizer.tts(givenText)
    filepath = os.path.join(audio_dir, filename) # create the file path for 
    enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15) #normalization on the wavs file
    synthesizer.save_wav(enhanced, filepath) # save the wav file to the file path
    wav_path = audio_dir + "/" + str(localTimestamp) + ".wav"
    return wav_path
# Create a gradio interface for the greeting function
coqui_ui_interface_000 = gr.Interface(
    fn=coquiDefaultModel, # the function to use
    inputs="text", # the input type # the input type
    examples=[["This is a cool way to use Coqui!"]], # examples to show in the UI
    outputs="audio", # the output type
    title="Coqui TTS", # the title of the UI
    description="Enter your text and get a audio file message" # brief description of the function
)
coqui_ui = gr.TabbedInterface([coqui_ui_interface_000,], ["Text-to-speech"])

# Display the UI
coqui_ui.launch(inline=True)