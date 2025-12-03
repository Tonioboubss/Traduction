import torch
from transformers import pipeline

def load_translator():
        translate_agent = pipeline(
              task="text2text-generation",
              model="google-t5/t5-base",
              dtype=torch.float16,
              device=0)
        return translate_agent



def translate(text, model, src_lang, tgt_lang):
    return model(f"translate {src_lang} to {tgt_lang}: {text}")[0]['generated_text']

#language 
#   - en
#   - fr
#   - ro
#   - de
#   - multilingual

source = 'French'
target = 'Je suis ton père'
text = 'I am your father'

print(translate(text, source, target))
      
#translate_agent("translate English to French: The weather is nice today.")
