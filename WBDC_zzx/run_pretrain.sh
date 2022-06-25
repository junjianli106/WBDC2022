CUDA_VISIBLE_DEVICES=2,4 python pretrain.py --savedmodel_path=save/chinese-roberta-wwm-ext/pretrain/len128_bs128_maxepoch40_itm5 \
    --max_epochs=10 --warmup_steps=-1 --warmup_ratio=0.06 --learning_rate=6e-5 --batch_size=128 --val_batch_size=128 \
    --fgm=0 --img_fgm_eps=0 --text_fgm_eps=0.5 --ema=0 --val_ratio=0.1 --bert_seq_length=128 \
    --bert_dir=hfl/chinese-roberta-wwm-ext --model_type=vbert --only_train=0 --train_annotation=./data/annotations/all.json \
    --train_zip_feats=./data/zip_feats/all.zip --pretrain=1 --lr_decay=cos \
    --mlm_co=1.0 --itm_co=5.0 --mfm_co=1.0
