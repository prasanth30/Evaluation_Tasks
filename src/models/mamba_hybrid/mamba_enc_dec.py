import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams

from transformers import PreTrainedTokenizerFast
from transformers.optimization import get_inverse_sqrt_schedule

from .utils.helpers.ffn import FeedForwardWrapper
from .utils.helpers.flash_cross_attention import FlashCrossAttentionWrapper
from .mamba import MambaDecoder, MixerModel

# TODO: change max seq len to arg
class MambaEncDec(pl.LightningModule):
    is_encoder_decoder = True
    is_concat = False  # FIXME remove
    model_name = "mamba_encdec"
    configs = {
        "default": {
            "enc_n_layer": 5,
            # mamba config
            "d_model": 512,
            "n_layer": 5,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": False,
            "learning_rate": 5e-4,
            "warmup_steps": 4000,
            "weight_decay": 0.001,
            "devices": 'cuda:0'
        }
    }

    def __init__(
        self,
        config=None,
        tokenizer=PreTrainedTokenizerFast,
        src_vocab_size=459,
        tgt_vocab_size=59,
        d_model=None,
        n_layer=None,
        enc_n_layer=None,
        rms_norm=None,
        fused_add_norm=None,
        use_fast_path=None,
        dropout=None,
        use_padding=None,
        precision="32-true",
        test_per_sample=True,
        test=False,
        test_suffix="",
        **kwargs,
    ):
        super().__init__()

        self.config = MambaConfig(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            # use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
        )

        self.encoder = MixerModel(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layer=enc_n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
            layer_dict={},
        )

        self.layers = (0, 3, 6, 9, 12, 15)
        x_attention_layers = [
            (i, FlashCrossAttentionWrapper) for i in (1, 4, 7, 10, 13, 16)
        ]
        ffn_layers = [(i, FeedForwardWrapper) for i in (2, 5, 8, 11, 14, 17)]

        layer_dict = dict(x_attention_layers + ffn_layers)

        self.decoder = MambaDecoder(
            config=self.config,
            layer_dict=layer_dict,
            layer_kwargs={"dropout":0.1}
        )

        self.tokenizer = tokenizer
        self.config = config
        self.use_padding = use_padding
        dtype_map = {
            "bf16-mixed": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self.precision = dtype_map[precision]

        if test:
            # self.comet = load_comet()
            self.test_per_sample = test_per_sample
            self.test_res = []
            self.test_suffix = test_suffix

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.decoder.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        context_tokens,
        input_ids,
        source_attention_mask=None,
        target_attention_mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
    ):
        
        b, _, _, l = source_attention_mask.shape
        source_attention_mask = source_attention_mask.reshape(b,l).to(torch.bool)
        target_attention_mask = target_attention_mask.to(torch.bool)

        source_vec = self.encoder.forward(
            input_ids=context_tokens,
            mask=source_attention_mask,
        )
        # print(source_vec.dtype, source_attention_mask.dtype)
        cache = self.allocate_inference_cache(
            batch_size=b,
            max_seqlen=300 + l + 1,  # source + BOS
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=300 + l + 1,
            max_batch_size=b,
            key_value_memory_dict=cache,
        )
        
        logits = self.decoder.forward(
            input_ids,
            context=source_vec,
            context_mask=source_attention_mask,
            attention_mask=target_attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )
        return logits

    def training_step(self, batch, batch_idx):

        source, target, _, source_attention_mask, _ = batch
        # source, target, source_attention_mask, 

        target_attention_mask = (
            (target != self.tokenizer.pad_token_id).to(torch.bool).to(source.device)
        )
        # print(source.type(), source_attention_mask.type())
        
        lm_logits = self.forward(
            context_tokens=source,
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
            input_ids=target,
        )
        
        logits = lm_logits[:, :-1].contiguous()
        labels = target[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        return loss

    def validation_step(self, batch, batch_idx):

        src_tokens, target, _, source_attention_mask, _ = batch

        target_attention_mask = (
            (target != self.tokenizer.pad_token_id).to(torch.bool).to(src_tokens.device)
        )
        
        lm_logits = self.forward(
            context_tokens=src_tokens,
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
            input_ids=target,
        )
        
        logits = lm_logits[:, :-1].contiguous()
        labels = target[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss
        
    def test_step(self, batch, batch_idx):
        """autoregressive generation"""

        src_tokens, _, labels, source_attention_mask, _ = batch
        batch_size, seq_len = src_tokens.shape
        max_length = 350
        cache = self.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_length + seq_len + 1,  # source + BOS
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=max_length + seq_len + 1,
            max_batch_size=batch_size,
            key_value_memory_dict=cache,
        )

        done = torch.tensor([False] * batch_size).to(src_tokens.device)
        preds = (
            torch.ones((batch_size, 1), dtype=torch.long).to(src_tokens.device)
            * self.tokenizer.bos_token_id
        )

        source_vec = self.encoder.forward(
            input_ids=src_tokens,
            mask=source_attention_mask,
        )

        position_ids = None

        for idx in range(labels.size(1)):

            if idx > 0:
                last_tokens = preds[:, -1:]  # (B, 1)
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=src_tokens.device,
                )

            logits = self.decoder.forward(
                input_ids=preds if idx == 0 else last_tokens,
                context=source_vec,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            )

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            preds = torch.cat((preds, next_token), dim=-1)
            inference_params.seqlen_offset += 1
            # print(next_token.dtype)
            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)

            if done.all():
                break

        # Create a cumulative sum mask where positions after EOS become True
        eos_token_id = self.tokenizer.eos_token_id
        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0
        preds[eos_mask] = self.tokenizer.pad_token_id
        
        preds = preds.cpu()
        labels = labels.cpu()

        # Clear GPU
        import gc
        del cache, logits, next_token_logits, inference_params
        if position_ids is not None:
            position_ids = position_ids.cpu()
            del position_ids

        eos_mask = eos_mask.cpu()
        del eos_mask
        gc.collect()
        torch.cuda.empty_cache()

        return preds, labels

    def on_test_epoch_end(self):
        # TODO save results
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            fused=True,
        )

        scheduler = {
            "scheduler": get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.config["warmup_steps"],
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
