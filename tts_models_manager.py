
from TTS.utils.manage import ModelManager
manager = ModelManager()

def fetchLanguageModel(language):
    if language == "nl":
        return manager.download_model("tts_models/nl/mai/tacotron2-DDC")
    elif language == "en":
        return manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
    elif language == "zh":
        return manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")
    else :
        pass