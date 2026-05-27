###### best val loss 지점에서 모든 생성 캡션 출력 및 반환, heatmap 저장 #####
import torch
from utils.checkpoint_manager import load_checkpoint

def make_show_all_caption(
        loader,
        encoder,
        decoder,
        optimizer,
        w2i,
        i2w,
        best_path,
        dec_atten_dir,
        enc_dec_atten_dir,
        heatmap_sample,
        layer,
        device
):
    
    _, best_val_loss = load_checkpoint(
        best_path,
        encoder,
        decoder,
        optimizer,
        device
    )

    all_references = []
    all_generated_token = []
    all_dec_atten = []
    all_enc_dec_atten = []
    all_images = []
    for images, _, batch_references in loader:
        images = images.to(device)

        features = encoder(images, return_features=True)

        generated_token, dec_atten, enc_dec_atten = decoder.generate(
                features, # B, 49, 512
                torch.full((features.size(0),), w2i["<sos>"], device=device), # B,
                w2i["<eos>"],
            )
        all_dec_atten.extend(dec_atten) # all_B, layers, nhead, seq_len, seq_len
        all_enc_dec_atten.extend(enc_dec_atten) # all_B, layers, nhead, seq_len, 49
        all_images.extend(images.cpu())
        all_references.extend(list(zip(*batch_references)))
        all_generated_token.extend(generated_token) # all_B, seq_len-1

    
    all_generated_sentence = []
    for sentence_token in all_generated_token:
        if w2i["<eos>"] in sentence_token:
            end_inx = sentence_token.index(w2i["<eos>"])
            sentence_token = sentence_token[:end_inx]

        words = [i2w[i] for i in sentence_token]
        
        all_generated_sentence.append(' '.join(words)) # all_B, 1(문장)


    decoder.show_dec_atten(all_dec_atten[heatmap_sample], all_generated_sentence[heatmap_sample].split(), layer, dec_atten_dir)
    decoder.show_cross_atten(all_enc_dec_atten[heatmap_sample], all_generated_sentence[heatmap_sample].split(), layer, all_images[heatmap_sample], enc_dec_atten_dir)

    all_B = len(all_generated_sentence)
    for i in range(0,all_B,30):
        print("-" * 60)
        print(f' Generated Sentence {i}: {all_generated_sentence[i]}')
        print("-" * 60)

        for inx, reference in enumerate(all_references[i], start=1):
            print(f'Reference {inx}: {reference}')
        print("=" * 60)

    print(f'Best Val Loss: {best_val_loss}')

    return all_generated_sentence, all_references

