import os
import re   # regular expressions, for splitting text file into sentences
from datetime import datetime
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from rvc_python.infer import RVCInference

'''
NOTES:
- Regex text splitting method currently loses final sentence if it doesn't end with proper punctuation
- Sample rate var in xtts for now due to deciding location (i.e. if I decide to go a gen param section)
- Currently not modularised into functions- might want to do this AFTER main script runs in sequential mode
- Question on code flow: Load XTTS -> XTTS inference -> Load RVC -> RVC inference
                    OR: Load XTTS -> Load RVC -> XTTS inference -> RVC inference


CODE STRUCTURE:
- Config: model file paths, set SPEAKER REFERENCE here
- Parameters section (TBD: how to deal with)
- Input text handling
- XTTS
- RVC

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
xtts_vocab_path = os.path.join(base_dir, "voices\\xtts\\vocab.json")
#XTTS trained model checkpoint file
xtts_checkpoint_path = os.path.join(base_dir, "voices\\xtts\\Ranni1\\best_model.pth")
#XTTS trained model config
xtts_config_path = os.path.join(base_dir, "voices\\xtts\\Ranni1\\config.json")
#RVC trained model file location
rvc_models_dir = os.path.join(base_dir, "voices\\rvc")






### PARAMETERS 

# REFERENCE AUDIO - by default looks in temp folder
speaker_ref_path = os.path.join(temp_dir, "audio153.wav")
# OUTPUT AUDIO name and path
output_wav_name = "output_" + datetime.now().strftime("%H-%M-%S") + ".wav"
output_wav_path = os.path.join(temp_dir, output_wav_name)
# Output Sample rate
sample_rate = 24000


# XTTS Settings

# Generate silence duration between sentences
silence_duration = 0.3 #(seconds)
silence_waveform = torch.zeros(int(sample_rate * silence_duration))
# XTTS Parameters
xtts_temp = 0.7
xtts_spd = 0.95


# RVC Settings

#set name of rvc model here and it will look for it automatically
# in the rvc models folder specified earlier
rvc_model_name = "ranni2"

rvc_model = RVCInference(models_dir = rvc_models_dir, device = "cuda:0")
# RVC Parameters
rvc_model.f0method = "rmvpe"
rvc_model.rms_mix_rate = 0.25
rvc_model.index_rate = 0.75




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
print("SENTENCES:\n", sentences)





### XTTS

# Init audio list that each generated XTTS sentence will be appended to
joined_audio = []

# Init XTTS model
print("\n Loading XTTS model...")
config = XttsConfig()
config.load_json(xtts_config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint_path, vocab_path=xtts_vocab_path, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_ref_path])

# Inference
print("Performing TTS generation")
for sentence in sentences:
    out = model.inference(
        sentence,
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )

    #Hold output as variable so we can append it to joined audio
    temp_waveform = torch.tensor(out["wav"])

    joined_audio.append(temp_waveform)
    joined_audio.append(silence_waveform)

# Concatinate all audio waveforms together and save file
joined_audio = torch.cat(joined_audio)
torchaudio.save(output_wav_path, joined_audio.unsqueeze(0), sample_rate)
print("XTTS inference complete!")
print(f"Saved TTS output to: {output_wav_path}")





### RVC

#We initialised the rvc_model object earlier, now we load an actual model
print("\nLoading RVC model...")
rvc_model.load_model(rvc_model_name)
print(f"Loaded RVC model: {rvc_model.models}")

# Inference
print("Performing RVC Inference...")
rvc_model.infer_file(output_wav_path, output_wav_path)
print(f"RVC Inference complete. Output: {output_wav_path}")
