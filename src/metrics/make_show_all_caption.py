###### best val loss 지점에서 모든 생성 캡션 출력 및 반환, heatmap 저장 #####
import os
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
        save_prefix,
        sample,
        layer,
        device,
        use_subword,
        sp_model_path
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
    all_file_name = []
    for images, _, batch_references, file_name in loader:
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
        all_file_name.extend(file_name)

    
    all_generated_sentence = []
    for sentence_token in all_generated_token:
        if w2i["<eos>"] in sentence_token:
            end_inx = sentence_token.index(w2i["<eos>"])
            sentence_token = sentence_token[:end_inx]
        
        # ==================================
        # SentencePiece tokenizer
        # ==================================
        if use_subword:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)
            # special token 제거
            sentence_token = [token for token in sentence_token
                if token not in [
                    w2i["<pad>"],
                    w2i["<sos>"],
                    w2i["<eos>"]
                ]
            ]
            sentence = sp.decode(sentence_token)

        else:
            words = [i2w[i] for i in sentence_token]
            sentence = ' '.join(words)
        
        all_generated_sentence.append(sentence) # all_B, 1(문장)


    for i in sample:
        dec_atten_name = os.path.join(dec_atten_dir, f"{save_prefix}_{i}")
        cross_atten_name = os.path.join(enc_dec_atten_dir, f"{save_prefix}_{i}")
        decoder.show_dec_atten(all_dec_atten[i], all_generated_sentence[i].split(), layer, dec_atten_name)
        decoder.show_cross_atten(all_enc_dec_atten[i], all_generated_sentence[i].split(), layer, all_images[i], cross_atten_name)

        print("-" * 60)
        print(f' {all_file_name[i]}: {all_generated_sentence[i]}')
        print("-" * 60)

        for inx, reference in enumerate(all_references[i], start=1):
            print(f'Reference {inx}: {reference}')
        print("=" * 60)

    print(f'Best Val Loss: {best_val_loss}')

    return all_generated_sentence, all_references

