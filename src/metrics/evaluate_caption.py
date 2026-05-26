import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def evaluate_caption(
    all_generated_sentence,
    all_references    
    ):

    references_dict = {i:list(sentences) for i, sentences in enumerate(all_references)}
    generated_dict = {i:[sentence] for i, sentence in enumerate(all_generated_sentence)}

    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(
        references_dict,
        generated_dict
    )

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(
        references_dict,
        generated_dict
    )

    metric_result = {
        "bleu1": bleu_score[0],
        "bleu2": bleu_score[1],
        "bleu3": bleu_score[2],
        "bleu4": bleu_score[3],
        "cider": cider_score,
        "generated": generated_dict,
        "references": references_dict
    }

    return metric_result