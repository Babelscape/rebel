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
"""Re-Tacred."""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
    Re-TACRED: Reimplementation of TACRED dataset.
"""
relations = {'no_relation': 'no relation',
'org:alternate_names': 'alternate name',
'org:city_of_branch': 'headquarters location', 
'org:country_of_branch': 'country of headquarters',
'org:dissolved': 'dissolved',
'org:founded_by': 'founded by',
'org:founded': 'inception',
'org:member_of': 'member of',
'org:members': 'has member',
'org:number_of_employees/members': 'member count',
'org:political/religious_affiliation': 'affiliation',
'org:shareholders': 'owned by',
'org:stateorprovince_of_branch': 'state of headquarters',
'org:top_members/employees': 'top members',
'org:website': 'website',
'per:age': 'age',
'per:cause_of_death': 'cause of death',
'per:charges': 'charge',
'per:children': 'child',
'per:cities_of_residence': 'city of residence',
'per:city_of_birth': 'place of birth',
'per:city_of_death': 'place of death',
'per:countries_of_residence': 'country of residence',
'per:country_of_birth': 'country of birth',
'per:country_of_death': 'country of death',
'per:date_of_birth': 'date of birth',
'per:date_of_death': 'date of death',
'per:employee_of': 'employer',
'per:identity': 'identity',
'per:origin': 'country of citizenship',
'per:other_family': 'relative',
'per:parents': 'father',
'per:religion': 'religion',
'per:schools_attended': 'educated at',
'per:siblings': 'sibling',
'per:spouse': 'spouse',
'per:stateorprovince_of_birth': 'state of birth',
'per:stateorprovince_of_death': 'state of death',
'per:stateorprovinces_of_residence': 'state of residence',
'per:title': 'position held'}


_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}


class TacredConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TacredConfig, self).__init__(**kwargs)


class Tacred(datasets.GeneratorBasedBuilder):
    """Tacred: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        TacredConfig(
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
            homepage="https://github.com/gstoica27/Re-TACRED",
#             citation=_CITATION,
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
            mapping = {'subj_start': '@', 'subj_end': '@', 'obj_start': '#', 'obj_end': '#'}
            for id_, row in enumerate(f):
                triplets = '<triplet> ' + ' '.join(row['token'][row['subj_start']:row['subj_end']+1]) + ' <subj> ' + ' '.join(row['token'][row['obj_start']:row['obj_end']+1]) + ' <obj> ' + relations[row['relation']]
                boundaries_entities = sorted([[index, row[index]] for index in row if index in ['subj_start', 'subj_end', 'obj_start', 'obj_end']], key=lambda tup: tup[1])
                new_tokens = []
                prev_tok = 0
                for index, boundary in boundaries_entities:
                    if index.endswith('end'):
                        boundary+=1
                    new_tokens.extend(row['token'][prev_tok:boundary])
                    new_tokens.append(mapping[index])
                    prev_tok = boundary
                new_tokens.extend(row['token'][prev_tok:])
                text = ' '.join(new_tokens)
                yield row["id"], {
                    "title": row["docid"],
                    "context": text,
                    "id": row["id"],
                    "triplets": triplets,
                }