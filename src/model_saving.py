from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import omegaconf
config = AutoConfig.from_pretrained(
    'facebook/bart-large',
    decoder_start_token_id = 0,
    early_stopping = False,
    no_repeat_ngram_size = 0,
)

tokenizer = AutoTokenizer.from_pretrained(
    'facebook/bart-large',
    use_fast=True,
    additional_special_tokens = ['<obj>', '<subj>', '<triplet>']
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    'facebook/bart-large',
    config=config,
)
model.resize_token_embeddings(len(tokenizer))

conf = omegaconf.OmegaConf.load('outputs/XXXX-XX-XX/XX-XX-XX/.hydra/config.yaml')
pl_module = BasePLModule(conf, config, tokenizer, model)
model = pl_module.load_from_checkpoint(checkpoint_path = 'outputs/XXXX-XX-XX/XX-XX-XX/experiments/dataset/last.ckpt', config = config, tokenizer = tokenizer, model = model)

model.model.save_pretrained('../model/MODEL-NAME')
model.tokenizer.save_pretrained('../model/MODEL-NAME')