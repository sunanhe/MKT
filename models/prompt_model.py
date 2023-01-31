import clip
import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import Transformer, LayerNorm
import torch.nn as nn
import torch
import json

_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.transformer_width = args.transformer_width 
        self.context_length = args.context_length
        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size, args.transformer_width)

        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.bert_embed_dim))

        self.load_from_openai_model(pretrained_model=args.clip_path)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_label_emb(self, seen_labels_path, unseen_labels_path):
        
        label925 = open(seen_labels_path, 'r').readlines()
        label81 = open(unseen_labels_path, 'r').readlines()
        label1006 = label925 + label81
        self.name_lens = [len(_tokenizer.encode(name)) for name in label1006]
        self.label_token = torch.zeros((len(self.name_lens), self.context_length), dtype=torch.long).to(self.device)
        for i, c in enumerate(label1006):
            self.label_token[i] = clip.tokenize(f"There is a {c.strip()} in the scene")
        self.label_emb = torch.zeros((len(self.name_lens), max(self.name_lens), self.transformer_width)).to(self.device)
        for i, embed in enumerate(self.token_embedding(self.label_token)):
            self.label_emb[i][:self.name_lens[i]] = embed[4:4+self.name_lens[i]].clone().detach()
        
    def load_from_openai_model(self, pretrained_model):
        state_dict = clip.load(pretrained_model, jit=False)[0].state_dict()
        load_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("visual") and (k not in ["logit_scale", "input_resolution", "context_length", "vocab_size"]):
                load_dict[k] = v
        msg = self.load_state_dict(load_dict)
        
    def load_label_emb(self, label=None):
        
        self.name_lens = [len(_tokenizer.encode(name.split("\t")[-1])) for name in label]
        self.label_token = torch.zeros((len(self.name_lens), self.context_length), dtype=torch.long).to(self.device)
        for i, c in enumerate(label):
            name = c.split("\t")[-1]
            self.label_token[i] = clip.tokenize(f"There is a {name.strip()} in the scene")
        self.label_emb = torch.zeros((len(self.name_lens), max(self.name_lens), self.transformer_width)).to(self.device)
        for i, embed in enumerate(self.token_embedding(self.label_token)):
            self.label_emb[i][:self.name_lens[i]] = embed[4:4+self.name_lens[i]].clone().detach()
  
    def forward(self, partion):

        if partion == "all":
            start_idx = 0
            end_idx = 1006
        elif partion == "unseen":
            start_idx = 925
            end_idx = 1006
        else:
            start_idx = 0
            end_idx = 925
        
        label_embeds = self.token_embedding(self.label_token[start_idx:end_idx])

        for i in range(label_embeds.shape[0]):
            label_embeds[i, 4:4+self.name_lens[start_idx:end_idx][i], :] = self.label_emb[start_idx:end_idx][i][:self.name_lens[start_idx:end_idx][i]]

        x = label_embeds + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        res = x[torch.arange(x.shape[0]), self.label_token[start_idx:end_idx].argmax(dim=-1)] @ self.text_projection

        return res
