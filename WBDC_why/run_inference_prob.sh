# CUDA_VISIBLE_DEVICES='2, 3' python -u inference.py \
#     --model_type='dualbert'\
#     --savedmodel_path='save/dualbert_roberta_base_fd6_fh12/fgm_ema_mask_sl_384_lr5e-5/' \
#     --ckpt_file='dualbert_best_mean_f1' \
#     --bert_seq_length=384 \
#     --bert_dir='hfl/chinese-roberta-wwm-ext' \
#     --bert_tokenizer_dir='hfl/chinese-roberta-wwm-ext' \
#     --fusion_num_hidden_layers=6 \
#     --fusion_num_attention_heads=12 \

CUDA_VISIBLE_DEVICES='0, 1' python -u inference_prob.py \
    --model_type='unibert'\
    --savedmodel_path='save/unibert_roberta_base_fd6_fh12/fgm_ema_mask_sl_384_lr5e-5/' \
    --ckpt_file='unibert_best_mean_f1' \
    --final_ckpt_file='unibert_final' \
    --bert_seq_length=384 \
    --bert_dir='hfl/chinese-roberta-wwm-ext' \
    --bert_tokenizer_dir='hfl/chinese-roberta-wwm-ext' \