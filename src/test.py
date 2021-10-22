import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

relations = {'no_relation': 'no relation',
'org:alternate_names': 'alternate name',
'org:city_of_branch': 'city of headquarters',
'org:country_of_branch': 'country of headquarters',
'org:dissolved': 'dissolved',
'org:founded_by': 'founded by',
'org:founded': 'founded',
'org:member_of': 'member of',
'org:members': 'members',
'org:number_of_employees/members': 'number of members',
'org:political/religious_affiliation': 'affiliation',
'org:shareholders': 'shareholders',
'org:stateorprovince_of_branch': 'state of headquarters',
'org:top_members/employees': 'top members',
'org:website': 'website',
'per:age': 'age',
'per:cause_of_death': 'cause of death',
'per:charges': 'charges',
'per:children': 'children',
'per:cities_of_residence': 'city of residence',
'per:city_of_birth': 'place of birth',
'per:city_of_death': 'place of death',
'per:countries_of_residence': 'country of residence',
'per:country_of_birth': 'country of birth',
'per:country_of_death': 'country of death',
'per:date_of_birth': 'date of birth',
'per:date_of_death': 'date of death',
'per:employee_of': 'employee of',
'per:identity': 'identity',
'per:origin': 'origin',
'per:other_family': 'other family',
'per:parents': 'parents',
'per:religion': 'religion',
'per:schools_attended': 'educated at',
'per:siblings': 'siblings',
'per:spouse': 'spouse',
'per:stateorprovince_of_birth': 'state of birth',
'per:stateorprovince_of_death': 'state of death',
'per:stateorprovinces_of_residence': 'state of residence',
'per:title': 'title'}


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        # cache_dir=conf.cache_dir,
        # revision=conf.model_revision,
        # use_auth_token=True if conf.use_auth_token else None,
    )
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )
    if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
        tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
        tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
        tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)
    pl_module = pl_module.load_from_checkpoint(checkpoint_path = conf.checkpoint_path, config = config, tokenizer = tokenizer, model = model)
    # pl_module.hparams.predict_with_generate = True
    pl_module.hparams.test_file = pl_data_module.conf.test_file
    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
    )
    # Manually run prep methods on DataModule
    pl_data_module.prepare_data()
    pl_data_module.setup()

    trainer.test(pl_module, test_dataloaders=pl_data_module.test_dataloader())


@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
    main()
