import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def evaluate_caption(
    decoder,
    all_feature,
    all_reference,
    w2i,
    i2w,
    batch_size
    ):
    device = next(decoder.parameters()).device

    generated_token = []
    with torch.no_grad():
        for i in range(0, all_feature.size(0), batch_size):
            batch_feature = all_feature[i:i+batch_size].to(device)

            batch_generated = decoder.generate(
                batch_feature,
                torch.full((batch_feature.size(0),), w2i["<sos>"], device=device),
                w2i["<eos>"]
            )

            generated_token.extend(batch_generated)

            del batch_feature
            torch.cuda.empty_cache()


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