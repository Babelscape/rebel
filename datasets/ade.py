# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ADE"""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
ADE dataset.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping = {'Adverse-Effect': 'has effect'}

class AdeConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AdeConfig, self).__init__(**kwargs)


class Ade(datasets.GeneratorBasedBuilder):
    """Ade Version 1.0."""

    BUILDER_CONFIGS = [
        AdeConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://www.sciencedirect.com/science/article/pii/S1532046412000615?via%3Dihub/",
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], 
                "dev": self.config.data_files["dev"], 
                "test": self.config.data_files["test"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)

        with open(filepath) as json_file:
            f = json.load(json_file)
            for id_, row in enumerate(f):
                triplets = ''
                prev_head = None
                for relation in row['relations']:
                    if prev_head == relation['head']:
                        triplets += f' <subj> ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' <obj> ' + mapping[relation['type']]
                    elif prev_head == None:
                        triplets += '<triplet> ' + ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' <subj> ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' <obj> ' + mapping[relation['type']]
                        prev_head = relation['head']
                    else:
                        triplets += ' <triplet> ' + ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' <subj> ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' <obj> ' + mapping[relation['type']]
                        prev_head = relation['head']
                text = ' '.join(row['tokens'])
                yield str(row["orig_id"]), {
                    "title": str(row["orig_id"]),
                    "context": text,
                    "id": str(row["orig_id"]),
                    "triplets": triplets,
                }