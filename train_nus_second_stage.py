#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import clip

from models.prompt_model import PromptLearner
from models.clip_vit import CLIPVIT
from utils.misc import *
from utils.dataset import build_dataloader
from engine_nus_second_stage import train, test, eval


def main(args):

    if not args.eval:

        init_distributed_mode(args)

        setup_seed(args.seed)

        args.is_master = is_main_process()
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        #########  RECORD SETTING ###########

        time = datetime.datetime.now().strftime('%m-%d-%H:%M:%S')
        record_name = time + "_" + args.ckpt_path.split("/")[-2][len("mm-dd-hh:mm:ss"):]
        args.record_path = os.path.join("outputs", "second_stage", record_name)

        if not os.path.exists(args.record_path):
            os.makedirs(args.record_path, exist_ok=True)
        
        if args.is_master:
            logger = init_log(args, args.record_path)
        else:
            logger = None

        cudnn.benchmark = True  # For speed i.e, cudnn autotuner

        # Build Dataloader
        train_dataset, test_dataset = build_dataloader(args)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, 
                                    args.batch_size, 
                                    pin_memory=True, 
                                    sampler=train_sampler, 
                                    num_workers=args.workers, 
                                    drop_last=True)
        test_dataloader = DataLoader(test_dataset, 
                                    args.test_batch_size,  
                                    shuffle=False, 
                                    pin_memory=True, 
                                    num_workers=args.workers, 
                                    drop_last=False)

        # Init Vision Backbone
        clip_model, _ = clip.load(args.clip_path, jit=False)
        image_encoder = CLIPVIT(args, clip_model)
        convert_models_to_fp32(image_encoder)

        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location="cuda")
            msg = image_encoder.load_state_dict(ckpt, strict=False)
            print(msg)

        image_encoder = image_encoder.to(args.gpu)

        # Init Language Backbone
        text_encoder = PromptLearner(args)
        text_encoder = text_encoder.to(args.gpu)

        # Generate Label Embedding
        unseen_labels_path = os.path.join(args.data_path, "Concepts81.txt")
        seen_labels_path = os.path.join(args.data_path, "Concepts925.txt")
        text_encoder.init_label_emb(seen_labels_path, unseen_labels_path)

        text_encoder = torch.nn.parallel.DistributedDataParallel(text_encoder, device_ids=[args.local_rank], find_unused_parameters=True)
        convert_models_to_fp32(text_encoder)

        image_encoder = image_encoder.eval()
        text_encoder = text_encoder.train()
        
        # Build Optimizer
        train_param = []
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                train_param.append(param)
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(train_param, args.lr, weight_decay=args.weight_decay)

        # Dump Params
        if is_main_process():

            logger.info("------------------------------------------------------------------")
            logger.info("USING LR SCHEDULER")
            logger.info("------------------------------------------------------------------")
            logger.info(("initial learning rate {}".format(args.lr)))
            logger.info(optimizer)

            write_description_to_folder(os.path.join(args.record_path, "params.txt"), args)
            
        scaler = GradScaler()

        for epoch in range(args.epochs): 
            train(text_encoder, image_encoder, optimizer, train_dataloader, logger, scaler, args, epoch)
            text_encoder.eval()
            test(text_encoder, image_encoder, args, test_dataloader, logger, len(test_dataset))
            text_encoder.train()
            dist.barrier()

    else:

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        cudnn.benchmark = True  # For speed i.e, cudnn autotuner

        # Build Dataloader
        _, test_dataset = build_dataloader(args)
        test_dataloader = DataLoader(test_dataset, 
                                    args.test_batch_size,  
                                    shuffle=False, 
                                    pin_memory=True, 
                                    num_workers=args.workers, 
                                    drop_last=False)

        # Init Vision Backbone
        clip_model, _ = clip.load(args.clip_path, device=args.device, jit=False)
        image_encoder = CLIPVIT(args, clip_model)
        convert_models_to_fp32(image_encoder)
        ckpt = torch.load(args.ckpt_path, map_location="cuda")
        msg = image_encoder.load_state_dict(ckpt, strict=True)
        print("Image Encoder Load Info: ", msg)
        image_encoder = image_encoder.to(args.device)
        

        # Init Language Backbone
        text_encoder = PromptLearner(args)
        text_encoder = text_encoder.to(args.device)

        unseen_labels_path = os.path.join(args.data_path, "Concepts81.txt")
        seen_labels_path = os.path.join(args.data_path, "Concepts925.txt")
        text_encoder.init_label_emb(seen_labels_path, unseen_labels_path)

        txt_ckpt = torch.load(args.eval_ckpt, map_location="cuda")
        if next(iter(txt_ckpt.items()))[0].startswith('module'):
            txt_ckpt = {k[len('module.'):]: v for k, v in txt_ckpt.items()}
        msg = text_encoder.load_state_dict(txt_ckpt, strict=True)
        print("Text Encoder Load Info: ", msg)

        image_encoder = image_encoder.eval()
        text_encoder = text_encoder.eval()
        
        scaler = GradScaler()

        eval(text_encoder, image_encoder, args, test_dataloader, len(test_dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--name",                   type=str,   default=None    )
    parser.add_argument("--seed",                   type=int,   default=42      )
    parser.add_argument("--record_path",            type=str,   default=None    )
    parser.add_argument("--eval",                   action="store_true"         )

    parser.add_argument("--ckpt-path",              type=str,   default=None    )
    parser.add_argument("--clip-path",              type=str,   default=None    )
    parser.add_argument("--eval-ckpt",              type=str,   default=None    )
    parser.add_argument("--data-path",              type=str,   default=None    )
    
    parser.add_argument("--batch-size",             type=int,   default=64,     )
    parser.add_argument("--test-batch-size",        type=int,   default=471,    )
    parser.add_argument("--epochs",                 type=int,   default=4,      )
    parser.add_argument("--warmup_epochs",          type=int,   default=0,      )
    parser.add_argument("--lr",                     type=float, default=1e-3,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-8,   )
    parser.add_argument("--weight_decay",           type=float, default=0.05,   )
    parser.add_argument("--workers",                type=int,   default=8,      )
    parser.add_argument("--momentum",               type=float, default=0.95,   )

    parser.add_argument("--input_size",             type=int,   default=224     )
    
    parser.add_argument("--bert-embed-dim",         type=int,   default=512,    )
    parser.add_argument("--context-length",         type=int,   default=77,     )
    parser.add_argument("--vocab-size",             type=int,   default=49408,  )
    parser.add_argument("--transformer-width",      type=int,   default=512,    )
    parser.add_argument("--transformer-heads",      type=int,   default=8,      )
    parser.add_argument("--transformer-layers",     type=int,   default=12,     )
    parser.add_argument("--topk",                   type=int,   default=16      )

    parser.add_argument("--local_rank",             type=int,   default=-1      )
    parser.add_argument("--world_size",             type=int,   default=1       )
    parser.add_argument("--dist_url",               type=str,   default='env://')

    args = parser.parse_args()

    main(args)

    