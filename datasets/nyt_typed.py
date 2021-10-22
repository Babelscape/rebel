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
"""Conll04: The Stanford Question Answering Dataset."""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
NYT dataset
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping = {'/people/person/nationality': 'country of citizenship', '/sports/sports_team/location': 'headquarters location', 
            '/location/country/administrative_divisions': 'contains administrative territorial entity', '/business/company/major_shareholders': 'shareholders', 
            '/people/ethnicity/people': 'country of origin', '/people/ethnicity/geographic_distribution': 'denonym', 
            '/business/company_shareholder/major_shareholder_of': 'major shareholder', '/location/location/contains': 'location',
            '/business/company/founders': 'founded by', '/business/person/company': 'employer', '/business/company/advisors': 'advisors', 
            '/people/deceased_person/place_of_death': 'place of death', '/business/company/industry': 'industry', 
            '/people/person/ethnicity': 'ethnicity', '/people/person/place_of_birth': 'place of birth', 
            '/location/administrative_division/country': 'country', '/people/person/place_lived': 'residence', 
            '/sports/sports_team_location/teams': 'member of sports team', '/people/person/children': 'child', 
            '/people/person/religion': 'religion', '/location/neighborhood/neighborhood_of': 'neighborhood of', 
            '/location/country/capital': 'capital', '/business/company/place_founded': 'location of formation', 
            '/people/person/profession': 'occupation'}

mapping_types = {'LOCATION': '<loc>', 'ORGANIZATION': '<org>', 'PERSON': '<per>'}

class NYTConfig(datasets.BuilderConfig):
    """BuilderConfig for NYT."""

    def __init__(self, **kwargs):
        """BuilderConfig for NYT.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NYTConfig, self).__init__(**kwargs)


class NYT(datasets.GeneratorBasedBuilder):
    """NYT: Version 1.0."""

    BUILDER_CONFIGS = [
        NYTConfig(
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
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
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
                list_relations = zip(row['spo_list'],row['spo_details'])
                relations_sorted = sorted(list_relations, key=lambda tup: tup[1][0])
                for relation, details in relations_sorted:
                    if prev_head == relation[0]:
                        triplets += f' {mapping_types[details[2]]} ' + relation[2] + f' {mapping_types[details[-1]]} ' + mapping[relation[1]]
                    elif prev_head == None:
                        triplets += '<triplet> ' + relation[0] + f' {mapping_types[details[2]]}  ' + relation[2] + f' {mapping_types[details[-1]]} ' + mapping[relation[1]]
                        prev_head = relation[0]
                    else:
                        triplets += ' <triplet> ' + relation[0] + f' {mapping_types[details[2]]}  ' + relation[2] + f' {mapping_types[details[-1]]} ' + mapping[relation[1]]
                        prev_head = relation[0]
                text = ' '.join(row['tokens'])
                yield str(id_), {
                    "title": str(id_),
                    "context": text,
                    "id": str(id_),
                    "triplets": triplets,
                }