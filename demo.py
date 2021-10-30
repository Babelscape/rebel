import streamlit as st
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from time import time
import torch

@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        AutoTokenizer: lambda x: None,
        AutoModelForSeq2SeqLM: lambda x: None,
    },
    suppress_st_warning=True
)
def load_models():
    st_time = time()
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    print("+++++ loading Model", time() - st_time)
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    if torch.cuda.is_available():
        _ = model.to("cuda:0") # comment if no GPU available
    _ = model.eval()
    print("+++++ loaded model", time() - st_time)
    dataset = load_dataset('datasets/rebel-short.py', data_files={'train': 'data/rebel/sample.jsonl', 'dev': 'data/rebel/sample.jsonl', 'test': 'data/rebel/sample.jsonl', 'relations': "data/relations_count.tsv"}, split="validation")
    return (tokenizer, model, dataset)

def extract_triplets(text):
    triplets = []
    relation = ''
    for token in text.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append((subject, relation, object_))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append((subject, relation, object_))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    triplets.append((subject, relation, object_))
    return triplets


tokenizer, model, dataset = load_models()

agree = st.checkbox('Free input', False)
if agree:
    text = st.text_input('Input text', 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.')
    print(text)
else:
    dataset_example = st.slider('dataset id', 0, 1000, 0)
    text = dataset[dataset_example]['context']
length_penalty = st.slider('length_penalty', 0, 10, 0)
num_beams = st.slider('num_beams', 1, 20, 3)
num_return_sequences = st.slider('num_return_sequences', 1, num_beams, 2)

gen_kwargs = {
    "max_length": 256,
    "length_penalty": length_penalty,
    "num_beams": num_beams,
    "num_return_sequences": num_return_sequences,
}

model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs,
)

decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
st.title('Input text')

st.write(text)

if not agree:
    st.title('Silver output')
    st.write(dataset[dataset_example]['triplets'])
    st.write(extract_triplets(dataset[dataset_example]['triplets']))

st.title('Prediction text')
decoded_preds = [text.replace('<s>', '').replace('</s>', '').replace('<pad>', '') for text in decoded_preds]
st.write(decoded_preds)

for idx, sentence in enumerate(decoded_preds):
    st.title(f'Prediction triplets sentence {idx}')
    st.write(extract_triplets(sentence))
