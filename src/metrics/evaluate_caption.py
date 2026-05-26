import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def evaluate_caption(
    decoder,
    all_feature,
    all_reference,
    w2i,
    i2w,
    batch_size,
    image_0,
    save_atten
    ):
    device = next(decoder.parameters()).device

    generated_token = []
    all_dec_atten = []
    all_enc_dec_atten = []
    with torch.no_grad():
        for i in range(0, all_feature.size(0), batch_size):
            batch_feature = all_feature[i:i+batch_size].to(device)
            
            # (B, seq_len-1), (layers, nhead, seq_len, seq_len), (layers, nhead, seq_len, 49)
            batch_generated, dec_atten, enc_dec_atten = decoder.generate(
                batch_feature, # B, 49, 512
                torch.full((batch_feature.size(0),), w2i["<sos>"], device=device), # all_B,
                w2i["<eos>"],
                save_atten
            )

            generated_token.extend(batch_generated)
            all_dec_atten.append(dec_atten) # all_B/B, layers, nhead, seq_len, seq_len
            all_enc_dec_atten.append(enc_dec_atten) # all_B/B, layers, nhead, seq_len, 49

            del batch_feature
            del dec_atten
            del enc_dec_atten

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
    if save_atten:
        all_dec_atten = torch.stack(all_dec_atten, dim=0) # all_B/B, layers, nhead, seq_len, seq_len)
        all_enc_dec_atten = torch.stack(all_enc_dec_atten, dim=0) # all_B/B, layers, nhead, seq_len, 49)

        decoder.show_dec_atten(all_dec_atten, generated_sentence[0].split(), 4, '/workspace/outputs/captioning/heatmap/dec_atten.jpg')
        decoder.show_cross_atten(all_enc_dec_atten, generated_sentence[0].split(), 4, image_0, '/workspace/outputs/captioning/heatmap/cross_atten.jpg')

    return metric_result