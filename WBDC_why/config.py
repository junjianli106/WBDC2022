import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split some training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    
    parser.add_argument('--start_fold', default=0, type=int, help="resume training")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--savedmodel_path', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--final_ckpt_file')
    
    parser.add_argument('--best_score', default=0, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--patience", default=1, type=float, help="Early Stop.")

    # ========================== Text BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_learning_rate', default=5e-5, type=float, help='initial bert learning rate')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    
    # ========================== Video ViT =============================
    parser.add_argument('--vit_depth', type=int, default=3)
    parser.add_argument('--vit_num_heads', type=int, default=12)
    parser.add_argument('--vit_mlp_ratio', type=int, default=4)
    parser.add_argument('--vit_learning_rate', default=1e-3, type=float, help='initial vit learning rate')
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    
    # ========================== Fusion Transformer =============================
    parser.add_argument('--fusion_num_hidden_layers', type=int, default=6)
    parser.add_argument('--fusion_num_attention_heads', type=int, default=12)
    parser.add_argument('--fusion_learning_rate', default=1e-4, type=float, help='initial fusion bert learning rate')
    parser.add_argument('--fusion_proj_size', type=int, default=768)

    parser.add_argument('--fp16', type=bool, default=True, help='use fp16')
    parser.add_argument('--do_fgm', type=bool, default=True, help='use the fgm')
    parser.add_argument('--do_pgd', type=bool, default=False, help='use the pgd')
    parser.add_argument('--do_ema', type=bool, default=True, help='use the ema')
    return parser.parse_args()
