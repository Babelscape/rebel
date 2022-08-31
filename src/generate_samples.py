from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import wandb
class GenerateTextSamplesCallback(Callback):  # pragma: no cover
    """
    PL Callback to generate triplets along training
    """

    def __init__(
        self,
        logging_batch_interval
    ):
        """
        Args:
            logging_batch_interval: How frequently to inspect/potentially plot something
        """
        super().__init__()
        self.logging_batch_interval = logging_batch_interval

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        wandb_table = wandb.Table(columns=["Source", "Pred", "Gold"])
        # pl_module.logger.info("Executing translation callback")
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
            return
        labels = batch.pop("labels")
        gen_kwargs = {
            "max_length": pl_module.hparams.val_max_target_length
            if pl_module.hparams.val_max_target_length is not None
            else pl_module.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "num_beams": pl_module.hparams.eval_beams if pl_module.hparams.eval_beams is not None else pl_module.config.num_beams,
        }
        pl_module.eval()

        decoder_inputs = torch.roll(labels, 1, 1)[:,0:2]
        decoder_inputs[:, 0] = 0
        generated_tokens = pl_module.model.generate(
            batch["input_ids"].to(pl_module.model.device),
            attention_mask=batch["attention_mask"].to(pl_module.model.device),
            decoder_input_ids=decoder_inputs.to(pl_module.model.device),
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = pl_module._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        pl_module.train()
        decoded_preds = pl_module.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        if pl_module.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = torch.where(labels != -100, labels, pl_module.tokenizer.pad_token_id)

        decoded_labels = pl_module.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_inputs = pl_module.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)

        # pl_module.logger.experiment.log_text('generated samples', '\n'.join(decoded_preds).replace('<pad>', ''))
        # pl_module.logger.experiment.log_text('original samples', '\n'.join(decoded_labels).replace('<pad>', ''))
        for source, translation, gold_output in zip(decoded_inputs, decoded_preds, decoded_labels):
            wandb_table.add_data(
                source.replace('<pad>', ''), translation.replace('<pad>', ''), gold_output.replace('<pad>', '')
            )
        pl_module.logger.experiment.log({"Triplets": wandb_table})
