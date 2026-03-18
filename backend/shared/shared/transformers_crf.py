# -*- coding: utf-8 -*-
# Adapted from https://github.com/lonePatient/BERT-NER-Pytorch/tree/master

from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    DebertaConfig,
    DebertaModel,
    PretrainedConfig,
    PreTrainedModel,
    RobertaConfig,
    RobertaModel,
)


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
        nbest: Optional[int] = None,
        pad_tag: Optional[int] = None,
    ) -> List[List[List[int]]]:
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}"
            )
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        seq_length, batch_size = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        pad_tag: Optional[int] = None,
    ) -> List[List[int]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_idx = torch.zeros((batch_size, self.num_tags), dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
            end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size), dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        nbest: int,
        pad_tag: Optional[int] = None,
    ) -> List[List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags, nbest), dtype=torch.long, device=device
        )
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full(
            (seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device
        )
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
            end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros(
            (seq_length, batch_size, nbest), dtype=torch.long, device=device
        )
        best_tags = (
            torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        )
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest
        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


class AutoModelForCrfPretrainedConfig(PretrainedConfig):
    model_type = "auto_model_for_crf"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "auto_model_for_crf"


class AutoModelCrfForNer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.model_type == "BertCrfForNer":
            self.model = BertCrfForNer(config)
        elif config.model_type == "RobertaCrfForNer":
            self.model = RobertaCrfForNer(config)
        elif config.model_type == "DebertaCrfForNer":
            self.model = DebertaCrfForNer(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        config = PretrainedConfig.from_pretrained(model_path)
        model_type = config.model_type
        if model_type == "BertCrfForNer":
            return BertCrfForNer.from_pretrained(model_path, *args, **kwargs)
        elif model_type == "RobertaCrfForNer":
            return RobertaCrfForNer.from_pretrained(model_path, *args, **kwargs)
        elif model_type == "DebertaCrfForNer":
            return DebertaCrfForNer.from_pretrained(model_path, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class BERT_CRF_Config(PretrainedConfig):
    model_type = "BertCrfForNer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "BertCrfForNer"


class BertCrfForNer(PreTrainedModel):
    config_class = BERT_CRF_Config

    def __init__(self, config):
        super().__init__(config)
        bert_config = BertConfig.from_pretrained(config.name_or_path)
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(config.name_or_path, config=bert_config)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, special_tokens_mask=None):
        last_hidden_layer = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        last_hidden_layer = self.dropout(last_hidden_layer)
        logits = self.linear(last_hidden_layer)
        outputs = (logits,)
        if special_tokens_mask is not None:
            special_tokens_mask = 1 - special_tokens_mask
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=special_tokens_mask)
            return (-1 * loss,) + outputs
        else:
            tags = self.crf.decode(logits, mask=attention_mask)
            return (logits, tags)


class ROBERTA_CRF_Config(PretrainedConfig):
    model_type = "RobertaCrfForNer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "RobertaCrfForNer"


class RobertaCrfForNer(PreTrainedModel):
    config_class = ROBERTA_CRF_Config

    def __init__(self, config):
        super().__init__(config)
        roberta_config = RobertaConfig.from_pretrained(config.name_or_path)
        roberta_config.output_attentions = True
        roberta_config.output_hidden_states = True
        self.roberta = RobertaModel.from_pretrained(
            config.name_or_path, config=roberta_config, add_pooling_layer=False
        )
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.roberta.config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, special_tokens_mask=None):
        last_hidden_layer = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        last_hidden_layer = self.dropout(last_hidden_layer)
        logits = self.linear(last_hidden_layer)
        outputs = (logits,)
        if special_tokens_mask is not None:
            special_tokens_mask = 1 - special_tokens_mask
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=special_tokens_mask)
            return (-1 * loss,) + outputs
        else:
            tags = self.crf.decode(logits, mask=attention_mask)
            return (logits, tags)


class DEBERTA_CRF_Config(PretrainedConfig):
    model_type = "DebertaCrfForNer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "DebertaCrfForNer"


class DebertaCrfForNer(PreTrainedModel):
    config_class = DEBERTA_CRF_Config

    def __init__(self, config):
        super().__init__(config)
        deberta_config = DebertaConfig.from_pretrained(config.name_or_path)
        deberta_config.output_attentions = True
        deberta_config.output_hidden_states = True
        self.deberta = DebertaModel.from_pretrained(config.name_or_path, config=deberta_config)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.deberta.config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, special_tokens_mask=None):
        last_hidden_layer = self.deberta(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        last_hidden_layer = self.dropout(last_hidden_layer)
        logits = self.linear(last_hidden_layer)
        outputs = (logits,)
        if special_tokens_mask is not None:
            special_tokens_mask = 1 - special_tokens_mask
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=special_tokens_mask)
            return (-1 * loss,) + outputs
        else:
            tags = self.crf.decode(logits, mask=attention_mask)
            return (logits, tags)
