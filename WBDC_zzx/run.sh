# hfl/chinese-roberta-wwm-ext
# hfl/chinese-macbert-base
# bert-base-chinese
# hfl/chinese-bert-wwm-ext

if [ $1 == 0 ]; then
echo 'train'
CUDA_VISIBLE_DEVICES=6 python main.py --savedmodel_path=save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/lr6e-5_warmup0.1_bs32_len384_pooldrop0.1_fgm0.15_ema_seed1026 \
    --max_epochs=5 --warmup_steps=-1 --warmup_ratio=0.1 --learning_rate=6e-5 --batch_size=32 --val_batch_size=128 \
    --fgm=1 --img_fgm_eps=0 --text_fgm_eps=0.15 --ema=0.9995 --val_ratio=0.1 --bert_seq_length=384 \
    --ema_start_epoch=0 \
    --pgd=0 --img_pgd_alpha=0 --text_pgd_alpha=0.5 --pgd_epsilon=0.5 \
    --bert_dir=hfl/chinese-roberta-wwm-ext --model_type=vbert --only_train=1 \
    --pretrain_path=/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrain/len128_bs128/model_epoch_14_loss_3.046991698706652_mlm_1.532_itm_0.225_mfm_7.384 \
    --val_annotation=./data/annotations/labeled_val0.1.json --seed=1026 \
    --train_annotation=./data/annotations/labeled.json --lr_decay=linear --pool_dropout=0.1 --pool=cls

elif [ $1 == 1 ]; then
echo 'tune'
CUDA_VISIBLE_DEVICES=0 python main.py --savedmodel_path=save/chinese-roberta-wwm-ext/tune/labeled_val0.1/pretrainbs128_len128_itm5_epoch9 \
    --max_epochs=5 --warmup_steps=-1 --warmup_ratio=0.1 --learning_rate=6e-5 --batch_size=32 --val_batch_size=128 \
    --fgm=1 --img_fgm_eps=0 --text_fgm_eps=0.5 --ema=0.9995 --val_ratio=0.1 --bert_seq_length=384 \
    --pgd=0 --img_pgd_alpha=0 --text_pgd_alpha=0.5 --pgd_epsilon=0.5 \
    --pretrain_path=/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrain/len128_bs128_maxepoch10_itm5/model_epoch_9_loss_3.3682066051626367_mlm_1.667_itm_1.059_mfm_7.379 \
    --bert_dir=hfl/chinese-roberta-wwm-ext --model_type=vbert --print_steps=1000 --tune=1 --val_annotation=./data/annotations/labeled_val0.1.json \
    --train_annotation=./data/annotations/labeled_train0.9.json

elif [ $1 == 2 ]; then
k=10
echo $k 'fold'
for((i=5;i<10;i++));
do
echo ${i}-th fold
CUDA_VISIBLE_DEVICES=4 python main.py --savedmodel_path=save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/${k}fold_seed2022/${i} \
    --max_epochs=5 --warmup_steps=-1 --warmup_ratio=0.1 --learning_rate=6e-5 --batch_size=32 --val_batch_size=128 \
    --fgm=1 --img_fgm_eps=0 --text_fgm_eps=0.15 --ema=0.9995 --val_ratio=0.1 --bert_seq_length=384 \
    --ema_start_epoch=0 \
    --pgd=0 --img_pgd_alpha=0 --text_pgd_alpha=0.5 --pgd_epsilon=0.5 \
    --bert_dir=hfl/chinese-roberta-wwm-ext --model_type=vbert --only_train=0 \
    --pretrain_path=/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrain/len128_bs128/model_epoch_14_loss_3.046991698706652_mlm_1.532_itm_0.225_mfm_7.384 \
    --val_annotation=./data/annotations/${k}fold_seed2022/val${i}.json --seed=2022 \
    --train_annotation=./data/annotations/${k}fold_seed2022/train${i}.json --lr_decay=linear --pool_dropout=0.1 --pool=cls

done

else
echo 'predict'
CUDA_VISIBLE_DEVICES=2 python inference.py --ckpt_file=/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/10fold_seed2022/0/model_epoch_2_mean_f1_0.6958_a1_0.7667_i1_0.7864_a2_0.5515_i2_0.6787.bin \
    --bert_dir=hfl/chinese-roberta-wwm-ext --bert_seq_length=384 --test_output_csv=./result/result.csv --pool=cls --outraw=0
fi