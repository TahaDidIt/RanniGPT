import os
import re   # regular expressions, for splitting text file into sentences
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from rvc_python.infer import RVCInference

'''
NOTES:
- Regex text splitting method currently loses final sentence if it doesn't end with proper punctuation



CODE STRUCTURE:
- Config: model file paths, set SPEAKER REFERENCE here
- Input text handling
- XTTS model setup

'''
#for reference
#base_dir::   C:\Taha\projects\Taha Programming\RanniGPT





### CONFIG

# Directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))
#By default output audio goes to temp folder
temp_dir = os.path.join(base_dir, "temp")


# Model file locations
#XTTS base model vocab file
xtts_vocab_path = os.path.join(base_dir, "voices/xtts/vocab.json")
#XTTS trained model checkpoint file
xtts_checkpoint_path = os.path.join(base_dir, "voices\\xtts\\Ranni1\\best_model.pth")
#XTTS trained model config
xtts_config_path = os.path.join(base_dir, "voices\\xtts\\Ranni1\\config.json")


# RVC model file location
rvc_models_dir = os.path.join(base_dir, "voices\\rvc")
#set name of rvc model here and it will look for it in the rvc models folder
rvc_model_name = "ranni2"


# REFERENCE AUDIO - by default looks in temp folder
speaker_ref_path = os.path.join(temp_dir, "audio153.wav")





### INPUT TEXT HANDLING

# Load sentences.txt - temp folder by default
with open(os.path.join(temp_dir, "sentences.txt"), "r", encoding = "utf-8") as file:
    text = file.read()

# Split sentences apart for sequential batched inference
"""
HOW THIS WORKS:
1. We use the `re` module for splitting text based on punctuation.
2. The regex `(\.{3}|[.!?])` ensures we handle:
    - Ellipses (`...`) as a single punctuation unit.
    - or Full stops (`.`), exclamation marks (`!`), and question marks (`?`).
3. Capturing groups `()` in the regex allow punctuation to be retained in the results rather than lost upon splitting.
4. Whitespace and empty strings are removed by strip() to ensure clean results.
5. How the if condition for .strip() works:
    - Python accepts an empty sentence ('') as a result of .strip() as False, anything else true
    - So it gets a true for non empty sentences and knows it can strip
6. After splitting, we combine each sentence with its respective punctuation by iterating
   through the list in pairs (sentence + punctuation) for the sake of the TTS engine.
"""
input_text = [sentence.strip() for sentence in re.split(r"(\.{3}|[.!?])", text) if sentence.strip()]

# combine sentences back with their punctuation mark
sentences = [
    input_text[i] + input_text[i + 1]
    for i in range(0, len(input_text) - 1, 2)  # range(inclusive, exclusive, step)
    #We stop it 1 element before the end to prevent error if last element has no punc.
]
print(sentences)





### XTTS SET-UP

print("\n Loading XTTS model...")
config = XttsConfig()
config.load_json(xtts_config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint_path, vocab_path=xtts_vocab_path, use_deepspeed=False)
model.cuda()

print("\n Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_ref_path])
