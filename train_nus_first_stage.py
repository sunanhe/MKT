import os
import argparse
import datetime


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import clip
from utils.misc import *
from utils.dataset import build_dataloader
from utils.optimizer import build_optimizer
from models.clip_vit import CLIPVIT
from engine_nus_first_stage import train, test

def main(args):

    setup_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    # Init Recoder
    record_name = datetime.datetime.now().strftime('%m-%d-%H:%M:%S') + "_" + "MKT"
    args.record_path = os.path.join("outputs", "first_stage", record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    # Init DataLoader
    train_dataset, val_dataset = build_dataloader(args)
    len_val_dataset = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                    args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.workers, 
                                    drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                args.test_batch_size,
                                shuffle=False, 
                                num_workers=args.workers, 
                                drop_last=False)

    # Load Label Embedding
    label_emd_path = os.path.join(args.data_path, 'label_emb.pt')
    label_emb = torch.load(label_emd_path, map_location=args.device).to(torch.float32)

    # Build Model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    model = CLIPVIT(args, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)

    # Load CLIP
    clip_model, _ = clip.load(args.clip_path, jit=False)
    clip_model.eval()
    clip_model = clip_model.to(args.device)

    # Build Optimizer
    optimizer = build_optimizer(args, model)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train(model, clip_model, args, optimizer, train_dataloader, logger, label_emb, epoch)
        model.eval()
        test(model, args, val_dataloader, logger, label_emb, len_val_dataset, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",                   type=int,   default=42  )
    parser.add_argument("--record_path",            type=str,   default=None)

    parser.add_argument("--clip-path",              type=str,   default=None)
    parser.add_argument("--data-path",              type=str,   default=None)
    
    parser.add_argument("--batch-size",             type=int,   default=64,     )
    parser.add_argument("--test-batch-size",        type=int,   default=471,    )
    parser.add_argument("--epochs",                 type=int,   default=20,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=2,      )
    parser.add_argument("--lr",                     type=float, default=1e-3,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-6,   )
    parser.add_argument("--weight_decay",           type=float, default=0.05,   )
    parser.add_argument("--workers",                type=int,   default=8,      )
    parser.add_argument("--momentum",               type=float, default=0.95,   )

    parser.add_argument("--input_size",             type=int,   default=224     )
    
    parser.add_argument("--layer_decay",            type=float, default=0.65    )
    parser.add_argument("--fix_layer",              type=int,   default=10      )
    parser.add_argument("--topk",                   type=int,   default=16      )

    args = parser.parse_args()

    main(args)
    
