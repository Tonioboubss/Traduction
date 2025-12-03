import torch
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Text(BaseModel):
    orig_text: str
    src_lang: str
    tgt_lang: str
    translated_text: str 

class Text_id(BaseModel):
    id : int
    text: str

list_Text = []

def load_translator():
        translate_agent = pipeline(
              task="text2text-generation",
              model="google-t5/t5-base",
              dtype=torch.float16,
              device=0)
        return translate_agent

agent = load_translator()

def translate(text: Text, model):
    return model(f"translate {text.src_lang} to {text.tgt_lang}: {text.orig_text}")[0]['generated_text']

@app.get("/")
async def read_HW():
    return {'Hello':'World'}

@app.get("/input/{item_id}")
async def read_origText(item_id: int):
    if len(list_Text)==0:
        return "No text to be translated"
    elif item_id >= len(list_Text):
        raise HTTPException(status_code=404, detail="Index too high")
    else:
         return list_Text[item_id].orig_text
    
@app.get("/input")
async def read_origTexts():
    if len(list_Text)==0:
        return "No text to be translated"
    else:
        return [list_Text[i].orig_text for i in range (len(list_Text))]

@app.post("/input")
def add_Text(text: str, srcLang: str, tgtLang: str):
    newText = Text(
        orig_text=text,
        src_lang=srcLang,
        tgt_lang=tgtLang,
        translated_text='')
    list_Text.append(newText)
    list_Text[-1].translated_text = translate(newText, agent)
    return f"New text to be translated in {newText.tgt_lang} : id={len(list_Text)-1}, text = {newText.orig_text}"

@app.put("/input/{item_id}")
def update_tgtLang(item_id:int, text: str):
    if len(list_Text)==0:
        return "No text to be updated"
    elif item_id > len(list_Text):
        raise HTTPException(status_code=404, detail="Index too high")
    else:
        list_Text[item_id]['orig_text'] = text
        list_Text[item_id].translated_text = translate(list_Text[item_id], agent)
        return list_Text[item_id].orig_text

@app.put("/input/{item_id}")
def update_tgtLang(item_id:int, tgtLang: str):
    if len(list_Text)==0:
        return "No text to be updated"
    elif item_id > len(list_Text):
        raise HTTPException(status_code=404, detail="Index too high")
    else:
        list_Text[item_id]['tgt_lang'] = tgtLang
        list_Text[item_id].translated_text = translate(list_Text[item_id], agent)
        return list_Text[item_id].tgt_lang

@app.get("/output/{item_id}")
async def read_translatedText(item_id: int):
    if len(list_Text)==0:
        return "No translated text"
    elif item_id >= len(list_Text):
        raise HTTPException(status_code=404, detail="Index too high")
    else:
         return list_Text[item_id].translated_text
    
@app.get("/output")
async def read_translatedTexts():
    if len(list_Text)==0:
        return "No translated text"
    else:
        return [list_Text[i].translated_text for i in range (len(list_Text))]

# source = 'French'
# target = 'Je suis ton père'
# text = 'I am your father'

# print(translate(text, agent, source, target))
      
#translate_agent("translate English to French: The weather is nice today.")
