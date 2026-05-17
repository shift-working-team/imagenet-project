import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def evaluate_caption(
    decoder,
    all_feature,
    all_reference,
    w2i,
    i2w
    ):
    generated_token = decoder.generate(
        all_feature,
        torch.full((all_feature.size(0),), w2i["<sos>"], device=all_feature.device),
        w2i["<eos>"]
    )

    generated_sentence = []
    for sentence in generated_token:
        if w2i["<eos>"] in sentence:
            end_inx = sentence.index(w2i["<eos>"])
            sentence = sentence[:end_inx]

        words = [i2w[i] for i in sentence]
        
        generated_sentence.append(' '.join(words))

    generated_dict = {i:[sentence] for i, sentence in enumerate(generated_sentence)}
    references_dict = {i:list(sentences) for i, sentences in enumerate(all_reference)}

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

    return {
        "bleu1": bleu_score[0],
        "bleu2": bleu_score[1],
        "bleu3": bleu_score[2],
        "bleu4": bleu_score[3],
        "cider": cider_score,
        "generated": generated_dict,
        "references": references_dict
    }