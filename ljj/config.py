import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='../challenge/data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='../challenge/data/annotations/test_a.json')
    parser.add_argument('--text_path', type=str, default='../challenge/data/annotations')
    parser.add_argument('--frame_fea', type=str, default='../challenge/data/zip_feats')
    parser.add_argument('--train_zip_feats', type=str, default='../challenge/data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='../challenge/data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=96, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=512, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=512, type=int, help="use for testing duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    parser.add_argument('--data_cache_path', type=str, default='..data/data_cache_path')
    # ======================== SavedModel Configs =========================
    #parser.add_argument('--savedmodel_path', type=str, default='save/VisualBertMultiModal_base//pretrained5_seqlen256_ema_fgm1_itm1_mlm10_mask_modal_')
    parser.add_argument('--savedmodel_path', type=str, default='save/VisualBertMultiModal_base_2022610//pretrained3_mlm_110W_w_fp16_seq_len256_ema_fgm1.5_learning_rate-cosine-9e-5')
    parser.add_argument('--ckpt_file', type=str, default='visual_roberta_best_mean_f1_0.6744.bin')
    parser.add_argument('--best_score', default=0, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=9e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--patience", default=1, type=float, help="Early Stop.")
    parser.add_argument("--scheduler", default='linear', type=str, help="scheduler") # cosine

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='new_self_pretrained_model_record/roberta-base-pretrain-mlm-itm-10-1')
    #parser.add_argument('--bert_dir', type=str, default='../challenge/prev_pretrained_models/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='../challenge/prev_pretrained_models/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_embeddding_size', type=int, default=768)
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    parser.add_argument('--do_fgm', type=bool, default=True, help='use the fgm')
    parser.add_argument('--fgm_epsilon', type=float, default=0.5)
    parser.add_argument('--do_pgd', type=bool, default=False, help='use the pgd')
    parser.add_argument('--do_ema', type=bool, default=True, help='use the pgd')
    parser.add_argument('--clear_text', type=str, default=False, help='clear_text')
    parser.add_argument('--mask_modal', type=str, default=True, help='mask_modal')
    
    # ========================== text masked pro =============================
    parser.add_argument('--masked_title_pro', type=float, default=0.15)
    parser.add_argument('--masked_asr_pro', type=float, default=0.17)
    parser.add_argument('--masked_ocr_pro', type=float, default=0.25)
    
    parser.add_argument('--hmc_loss', type=bool, default=False, help='use the hmc loss')

    parser.add_argument('--hmc_lambda', type=float, default=0.5)
    parser.add_argument('--hmc_beta', type=float, default=0.5)
    
    parser.add_argument('--fp16', type=bool, default=True, help='use the fp16')
    
    return parser.parse_args()
