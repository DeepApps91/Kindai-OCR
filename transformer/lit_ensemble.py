import zipfile
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from bttr.datamodule import Batch, vocab, vocab_size
from bttr.lit_bttr import LitBTTR
from bttr.utils import ExpRateRecorder, Hypothesis, to_tgt_output


class LitEnsemble(pl.LightningModule):
    def __init__(self, paths: List[str]):
        super(LitEnsemble, self).__init__()

        self.models = nn.ModuleList()
        for p in paths:
            model = LitBTTR.load_from_checkpoint(checkpoint_path=p)
            model = model.eval()
            self.models.append(model)

        self.beam_size = self.models[0].hparams.beam_size
        self.max_len = self.models[0].hparams.max_len
        self.alpha = self.models[0].hparams.alpha
        self.recorder = ExpRateRecorder()

    def test_step(self, batch: Batch, _):
        hypotheses = self.beam_search(batch.imgs, batch.mask)
        best_hypo = max(hypotheses, key=lambda h: h.score / (len(h) ** self.alpha))

        indices = batch.indices[0]
        indices_hat = best_hypo.seq

        self.recorder(indices_hat, indices)

        return {
            "fname": batch.img_bases[0],
            "pred": vocab.indices2label(indices_hat),
        }

    def test_epoch_end(self, outputs) -> None:
        exp_rate = self.recorder.compute()

        print(f"ExpRate: {exp_rate}")
        print(f"length of total file: {len(outputs)}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for d in outputs:
                content = f"%{d['fname']}\n${d['pred']}$".encode()
                with zip_f.open(f"{d['fname']}.txt", "w") as f:
                    f.write(content)

    def beam_search(self, img, mask):
        src_mask_list = [m.bttr.encoder(img, mask) for m in self.models]

        l2r_hyps = self.ensemble_beam_search(
            src_mask_list=src_mask_list,
            direction="l2r",
            beam_size=self.beam_size,
            max_len=self.max_len,
        )
        self.ensemble_cross_rate_score(
            src_mask_list=src_mask_list,
            hypotheses=l2r_hyps,
            direction="r2l",
        )

        r2l_hyps = self.ensemble_beam_search(
            src_mask_list=src_mask_list,
            direction="r2l",
            beam_size=self.beam_size,
            max_len=self.max_len,
        )
        self.ensemble_cross_rate_score(
            src_mask_list=src_mask_list,
            hypotheses=r2l_hyps,
            direction="l2r",
        )

        return l2r_hyps + r2l_hyps

    def ensemble_beam_search(
        self,
        src_mask_list: List[Tuple[torch.Tensor, torch.Tensor]],
        direction: str,
        beam_size: int,
        max_len: int,
    ) -> List[Hypothesis]:
        """search result for single image with beam strategy

        Args:
            src_mask_list: [([1, len, d_model], [1, len])]
            direction (str):
            beam_size (int): beam size
            max_len (int): max length for decode result

        Returns:
            List[Hypothesis(seq: [max_len])]: list of hypotheses(no order)
        """
        assert direction in {"l2r", "r2l"}

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

            prob_sum = torch.zeros(
                (hyp_num, vocab_size),
                dtype=torch.float,
                device=self.device,
            )
            for i, m in enumerate(self.models):
                src, src_mask = src_mask_list[i]
                exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
                exp_src_mask = repeat(src_mask.squeeze(0), "s -> b s", b=hyp_num)

                decode_outputs = m.bttr.decoder(exp_src, exp_src_mask, hypotheses)[
                    :, t, :
                ]
                prob_sum = prob_sum + torch.softmax(decode_outputs, dim=-1)
            log_p_t = torch.log(prob_sum / len(self.models))

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

    def ensemble_cross_rate_score(
        self,
        src_mask_list: List[Tuple[torch.Tensor, torch.Tensor]],
        hypotheses: List[Hypothesis],
        direction: str,
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace

        Args:
            src_mask_list: [([1, len, d_model], [1, len])]
            hypotheses (List[Hypothesis]):
            direction (str): one of {"l2r", "r2l"}
        """
        indices = [h.seq for h in hypotheses]
        tgt, output = to_tgt_output(indices, direction, self.device)

        b, length = tgt.size()
        prob_sum = torch.zeros(
            (b, length, vocab_size), dtype=torch.float, device=self.device
        )
        for i, m in enumerate(self.models):
            src, src_mask = src_mask_list[i]
            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
            exp_src_mask = repeat(src_mask.squeeze(0), "s -> b s", b=b)

            output_hat = m.bttr.decoder(exp_src, exp_src_mask, tgt)
            prob_sum = prob_sum + torch.softmax(output_hat, dim=-1)
        log_p = torch.log(prob_sum / len(self.models))

        flat_hat = rearrange(log_p, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        loss = F.nll_loss(flat_hat, flat, ignore_index=vocab.PAD_IDX, reduction="none")

        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, length in enumerate(loss):
            score = -length
            hypotheses[i].score += score
