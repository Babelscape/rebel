from typing import Any, Union, List, Optional

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset, set_caching_enabled
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)

class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model
        if conf.relations_file:
            self.datasets = load_dataset(conf.dataset_name, data_files={'train': conf.train_file, 'dev': conf.validation_file, 'test': conf.test_file, 'relations': conf.relations_file})
        else:
            self.datasets = load_dataset(conf.dataset_name, data_files={'train': conf.train_file, 'dev': conf.validation_file, 'test': conf.test_file})
        set_caching_enabled(True)
        self.prefix = conf.source_prefix if conf.source_prefix is not None else ""
        self.column_names = self.datasets["train"].column_names
        # self.source_lang, self.target_lang, self.text_column, self.summary_column = None, None, None, None
        self.text_column = conf.text_column
        self.summary_column = conf.target_column
        self.max_target_length = conf.max_target_length
        self.padding = "max_length" if conf.pad_to_max_length else False

        # Data collator
        label_pad_token_id = -100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if conf.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, label_pad_token_id=label_pad_token_id)

    def prepare_data(self, *args, **kwargs):
        self.train_dataset = self.datasets["train"]
        if "train" not in self.datasets:
            raise ValueError("--do_train requires a train dataset")
        if self.conf.max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(range(self.conf.max_train_samples))
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.overwrite_cache,
            cache_file_name=self.conf.train_file.replace('.jsonl', '-') + self.conf.dataset_name.split('/')[-1].replace('.py', '.cache'),
        )

        if self.conf.do_eval:
            max_target_length = self.conf.val_max_target_length
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")
            self.eval_dataset = self.datasets["validation"]
            if self.conf.max_val_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.conf.max_val_samples))
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=self.conf.validation_file.replace('.jsonl', '-') + self.conf.dataset_name.split('/')[-1].replace('.py', '.cache'),
            )

        if self.conf.do_predict:
            max_target_length = self.conf.val_max_target_length
            if "test" not in self.datasets:
                raise ValueError("--do_predict requires a test dataset")
            self.test_dataset = self.datasets["test"]
            if self.conf.max_test_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.conf.max_test_samples))
            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=self.conf.test_file.replace('.jsonl', '-') + self.conf.dataset_name.split('/')[-1].replace('.py', '.cache'),
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.conf.eval_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.eval_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     raise NotImplementedError

    def preprocess_function(self, examples):

        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.conf.max_source_length, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.conf.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        # model_inputs["decoder_input_ids"] = labels["input_ids"]
        # model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        # model_inputs["labels"] = shift_tokens_left(labels["input_ids"], self.tokenizer.pad_token_id)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs