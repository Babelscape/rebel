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
"""DocRED"""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
DocRED for RE.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping_types = {'LOC': '<loc>', 'MISC': '<misc>', 'PER': '<per>', 'NUM': '<num>', 'TIME': '<time>', 'ORG': '<org>'}

class DocREDConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocREDConfig, self).__init__(**kwargs)


class DocRED(datasets.GeneratorBasedBuilder):
    """DocRED Version 1.0."""

    BUILDER_CONFIGS = [
        DocREDConfig(
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
            homepage="https://github.com/lavis-nlp/jerex/",
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
        relations_file = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}
        with open(filepath) as json_file:
            f = json.load(json_file)
            for id_, row in enumerate(f):
                triplets = ''
                prev_head = None
                relations_sorted = sorted(row['labels'], key=lambda tup: tup['h'])
                for relation in relations_sorted:
                    if prev_head == relation['h']:
                        triplets += f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} ' + row['vertexSet'][relation['t']][0]['name'] + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} ' + relations_file[relation['r']]
                    elif prev_head == None:
                        triplets += '<triplet> ' + row['vertexSet'][relation['h']][0]['name'] + f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} ' + row['vertexSet'][relation['t']][0]['name'] + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} ' + relations_file[relation['r']]
                        prev_head = relation['h']
                    else:
                        triplets += ' <triplet> ' + row['vertexSet'][relation['h']][0]['name'] + f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} ' + row['vertexSet'][relation['t']][0]['name'] + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} ' + relations_file[relation['r']]
                        prev_head = relation['h']
                yield str(row["title"]), {
                    "title": str(row["title"]),
                    "context": ' '.join([word for sentence in row['sents'] for word in sentence]),
                    "id": str(row["title"]),
                    "triplets": triplets,
                }