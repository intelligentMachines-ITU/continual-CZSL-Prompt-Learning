import argparse

import torch
from clip.model import CLIP

from .text_encoder import CustomTextEncoder


class CLIPInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        token_ids: torch.tensor,
        soft_embeddings: torch.nn.Parameter = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
       
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("CLIP INTERFACE FILE FOR LOADING MODELS")
        self.token_ids = token_ids
        print("self.token_ids")
        print(self.token_ids)
        self.soft_embeddings = soft_embeddings
        # if (self.soft_embeddings):
        print("self.soft_embeddings.shape")
        print(self.soft_embeddings.shape)

    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se):
        if se.shape == self.soft_embeddings.shape:
            self.state_dict()['soft_embeddings'].copy_(se)
            print("self.state_dict()['soft_embeddings'].copy_(se)")
           
        else:
            print("Error")
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!")

    def construct_token_tensors(self, idx):
       
        if self.soft_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError

    def forward(self, batch_img, idx):
        batch_img = batch_img.to(self.device)

        token_tensors = self.construct_token_tensors(idx)

        text_features = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )

        #_text_features = text_features[idx, :]
        _text_features = text_features

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * normalized_img
            @ idx_text_features.t()
        )

        return logits