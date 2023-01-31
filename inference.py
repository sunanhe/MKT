import os
import argparse
from PIL import Image

import torch

import clip

from models.prompt_model import PromptLearner
from models.clip_vit import CLIPVIT
from utils.misc import convert_models_to_fp32
from utils.transforms import build_transform


def main(args):

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Image Encoder
    clip_model, _ = clip.load(args.clip_path, device=args.device, jit=False)
    image_encoder = CLIPVIT(args, clip_model)
    convert_models_to_fp32(image_encoder)
    ckpt = torch.load(args.img_ckpt, map_location="cuda")
    msg = image_encoder.load_state_dict(ckpt, strict=True)
    print("Image Encoder Load Info: ", msg)
    image_encoder = image_encoder.eval().to(args.device)

    # Load Text Encoder
    text_encoder = PromptLearner(args)
    text_encoder = text_encoder.to(args.device)
    txt_ckpt = torch.load(args.txt_ckpt, map_location="cuda")
    if next(iter(txt_ckpt.items()))[0].startswith('module'):
        txt_ckpt = {k[len('module.'):]: v for k, v in txt_ckpt.items()}
    msg = text_encoder.load_state_dict(txt_ckpt, strict=True)
    print("Text Encoder Load Info: ", msg)

    unseen_labels = open(os.path.join(args.data_path, "Concepts81.txt")).readlines()
    seen_labels = open(os.path.join(args.data_path, "Concepts925.txt")).readlines()

    label_nus = seen_labels + unseen_labels

    text_encoder.load_label_emb(label_nus)
    text_encoder = text_encoder.eval()

    with torch.no_grad():
        txt_feat = text_encoder("all").float()

    # Preprocess Image
    transforms = build_transform(False, args) 
    img = Image.open(args.image_path).convert('RGB')
    img = transforms(img).unsqueeze(0)

    # Infer 
    with torch.no_grad():
        pred_feat, dist_feat = image_encoder.encode_img(img.to(args.device))

    score1 = torch.topk(pred_feat @ txt_feat.t(), k=image_encoder.topk, dim=1)[0].mean(dim=1)
    score2 = dist_feat @ txt_feat.t()
    score1 = score1 / score1.norm(dim=-1, keepdim=True)
    score2 = score2 / score2.norm(dim=-1, keepdim=True)
    logits = (score1 + score2) / 2

    _, topk_preds = logits.topk(10)

    print("Top-10 Predictions: ")
    for idx in topk_preds[0]:
        print(label_nus[idx].strip().ljust(20) + str(float(logits[:, idx].data))[:6])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--txt-ckpt",               type=str,   default=None    )
    parser.add_argument("--img-ckpt",               type=str,   default=None    )
    parser.add_argument("--clip-path",              type=str,   default=None    )
    parser.add_argument("--image-path",             type=str,   default=None    )
    parser.add_argument("--data-path",              type=str,   default=None    )

    parser.add_argument("--input_size",             type=int,   default=224     )
    
    parser.add_argument("--bert-embed-dim",         type=int,   default=512,    )
    parser.add_argument("--context-length",         type=int,   default=77,     )
    parser.add_argument("--vocab-size",             type=int,   default=49408,  )
    parser.add_argument("--transformer-width",      type=int,   default=512,    )
    parser.add_argument("--transformer-heads",      type=int,   default=8,      )
    parser.add_argument("--transformer-layers",     type=int,   default=12,     )
    parser.add_argument("--topk",                   type=int,   default=18      )

    args = parser.parse_args()

    main(args)