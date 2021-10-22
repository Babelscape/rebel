#!/usr/bin/env python3

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np 

NO_RELATION = "no relation"

relations = ['no_relation',
'org:alternate_names',
'org:city_of_headquarters',
'org:country_of_headquarters',
'org:dissolved',
'org:founded',
'org:founded_by',
'org:member_of',
'org:members',
'org:number_of_employees/members',
'org:parents',
'org:political/religious_affiliation',
'org:shareholders',
'org:stateorprovince_of_headquarters',
'org:subsidiaries',
'org:top_members/employees',
'org:website',
'per:age',
'per:alternate_names',
'per:cause_of_death',
'per:charges',
'per:children',
'per:cities_of_residence',
'per:city_of_birth',
'per:city_of_death',
'per:countries_of_residence',
'per:country_of_birth',
'per:country_of_death',
'per:date_of_birth',
'per:date_of_death',
'per:employee_of',
'per:origin',
'per:other_family',
'per:parents',
'per:religion',
'per:schools_attended',
'per:siblings',
'per:spouse',
'per:stateorprovince_of_birth',
'per:stateorprovince_of_death',
'per:stateorprovinces_of_residence',
'per:title']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(prediction)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

'''Adapted from: https://github.com/btaille/sincere/blob/6f5472c5aeaf7ef7765edf597ede48fdf1071712/code/utils/evaluation.py'''
def re_score(pred_relations, gt_relations, relation_types, mode="boundaries"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]
    relation_types = relations if relation_types is None else relation_types
    # relation_types = [v for v in relation_types if not v == "None"]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per relation Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    print(f"RE Evaluation in *** {mode.upper()} *** mode")

    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type][
                "fp"]))

    return scores, precision, recall, f1


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: {} in gold vs {} in prediction".format(len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

