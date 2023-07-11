from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor
from torch.nn.modules.transformer import TransformerDecoder

from bttr.datamodule import vocab, vocab_size
from bttr.model.pos_enc import WordPosEnc, WordRotaryEmbed
from bttr.utils import Hypothesis, to_tgt_output


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerDecoder:
    """build transformer decoder with params
    Parameters
    ----------
    d_model : int
    nhead : int
    num_decoder_layers : int
    dim_feedforward : int
    dropout : float
    Returns
    -------
    nn.TransformerDecoder
    """
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder


class Decoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask: LongTensor
            [b, t]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]

        src = rearrange(src, "b t d -> t b d")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def _beam_search(
        self,
        src: FloatTensor,
        mask: LongTensor,
        direction: str,
        beam_size: int,
        max_len: int,
    ) -> List[Hypothesis]:
        """run beam search for one direction

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        direction : str
            one of "l2r" and "r2l"
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        assert direction in {"l2r", "r2l"}
        assert (
            src.size(0) == 1 and mask.size(0) == 1
        ), f"beam search should only have single source, encounter with batch_size: {src.size(0)}"

        if direction == "l2r":
            start_w = vocab.SOS_IDX
            stop_w = vocab.EOS_IDX
        else:
            start_w = vocab.EOS_IDX
            stop_w = vocab.SOS_IDX

        hypotheses = torch.full(
            (1, max_len + 1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=self.device,
        )
        hypotheses[:, 0] = start_w

        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)

            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]
            log_p_t = F.log_softmax(decode_outputs, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=vocab_size)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                continuous_hyp_scores, k=live_hyp_num
            )

            prev_hyp_ids = top_cand_hyp_pos // vocab_size
            hyp_word_ids = top_cand_hyp_pos % vocab_size

            t += 1
            new_hypotheses = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                hypotheses[prev_hyp_id, t] = hyp_word_id

                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            seq_tensor=hypotheses[prev_hyp_id, 1:t]
                            .detach()
                            .clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction=direction,
                        )
                    )
                else:
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),
                    score=hyp_scores[0].detach().item(),
                    direction=direction,
                )
            )

        return completed_hypotheses

    def _cross_rate_score(
        self,
        src: FloatTensor,
        mask: LongTensor,
        hypotheses: List[Hypothesis],
        direction: str,
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask : LongTensor
            [1, l]
        hypotheses : List[Hypothesis]
        direction : str
        """
        assert direction in {"l2r", "r2l"}
        indices = [h.seq for h in hypotheses]
        tgt, output = to_tgt_output(indices, direction, self.device)

        b = tgt.size(0)
        exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
        exp_mask = repeat(mask.squeeze(0), "s -> b s", b=b)

        output_hat = self(exp_src, exp_mask, tgt)

        flat_hat = rearrange(output_hat, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        loss = F.cross_entropy(
            flat_hat, flat, ignore_index=vocab.PAD_IDX, reduction="none"
        )

        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, l in enumerate(loss):
            score = -l
            hypotheses[i].score += score

    def beam_search(
        self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run beam search for src img

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        l2r_hypos = self._beam_search(src, mask, "l2r", beam_size, max_len)
        self._cross_rate_score(src, mask, l2r_hypos, direction="r2l")

        r2l_hypos = self._beam_search(src, mask, "r2l", beam_size, max_len)
        self._cross_rate_score(src, mask, r2l_hypos, direction="l2r")
        return l2r_hypos + r2l_hypos
