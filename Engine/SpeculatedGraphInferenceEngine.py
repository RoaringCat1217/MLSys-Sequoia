from typing import Optional, List

import torch
from utils import _make_causal_mask

from .Engine import GraphInferenceEngine


class SpeculatedGraphInferenceEngine:
    def __init__(self,
                 tiny_model_engine: GraphInferenceEngine,
                 spec_steps: int,
                 *args, **kwargs):
        self.model = GraphInferenceEngine(*args, **kwargs)
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.tiny_model_engine = tiny_model_engine
        self.spec_steps = spec_steps
        self.num_nodes = 0
        self.n_read = 0
        self.attn_mask = _make_causal_mask((1, self.model.max_length), dtype=self.dtype, device=self.device)
        self.cached_logits = None

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        n_needed = len(input_ids[0])
        self.num_nodes = self.n_read = n_needed
        self.tiny_model_engine.inference(input_ids,
                                         storage_ids,
                                         position_ids,
                                         attn_mask)
        return self.model.inference(input_ids,
                                    storage_ids,
                                    position_ids,
                                    attn_mask)

    @torch.inference_mode()
    def graph_inference(self,
            input_ids: torch.LongTensor,
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            debug :bool=False):
        n_needed = len(input_ids[0])
        if n_needed <= self.num_nodes - self.n_read:
            self.n_read += n_needed
            logits = self.cached_logits[:n_needed]
            self.cached_logits = self.cached_logits[n_needed:]
            return logits[None, ...]
        assert self.num_nodes == self.n_read

        # draft the tiny model for spec_steps steps
        logits = self.tiny_model_engine.graph_inference(input_ids,
                                                        storage_ids,
                                                        position_ids,
                                                        attn_mask)
        speculated_tokens = [logits[0, -1].argmax().item()]
        full = False
        for step in range(1, self.spec_steps + 1):
            if not self.draft_step(step, n_needed, speculated_tokens):
                full = True
                break
        if not full:
            speculated_tokens.pop()
        # if self.model.max_length - n_needed < len(speculated_tokens):
        #     speculated_tokens = speculated_tokens[:self.model.max_length - n_needed]

        # verify
        input_ids = input_ids[0].cpu().tolist() + speculated_tokens

        if self.num_nodes + len(input_ids) > self.model.max_length:
            diff = self.num_nodes + len(input_ids) - self.model.max_length
            input_ids = input_ids[:len(input_ids) - diff]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        storage_ids = position_ids = torch.arange(self.num_nodes, self.num_nodes + len(input_ids), device=self.device).long()
        attn_mask = self.attn_mask[self.num_nodes:self.num_nodes + len(input_ids)]
        logits = self.model.graph_inference(input_ids.unsqueeze(0),
                                            storage_ids,
                                            position_ids.unsqueeze(0),
                                            attn_mask[None, None])[0]
        ground_truth = logits.argmax(dim=1).cpu().tolist()[n_needed - 1:-1]
        n_accept = 0
        for i in range(len(ground_truth)):
            if speculated_tokens[i] == ground_truth[i]:
                n_accept += 1
            else:
                break
        self.model.set_kv_len(self.num_nodes + n_needed + n_accept)
        self.tiny_model_engine.set_kv_len(self.num_nodes + n_needed + n_accept)
        self.num_nodes += n_needed + n_accept
        self.n_read += n_needed
        self.cached_logits = logits[n_needed:]
        return logits[None, :n_needed]

    @torch.inference_mode()
    def draft_step(self, step: int, n_needed: int, speculated_tokens: List[int]) -> bool:
        offset = self.num_nodes + n_needed + step - 1
        if offset >= self.model.max_length:
            return False
        input_ids = torch.tensor([speculated_tokens[-1]], device=self.device).long()
        storage_ids = position_ids = torch.tensor([offset], device=self.device).long()
        attn_mask = self.attn_mask[offset:offset + 1]
        logits = self.tiny_model_engine.graph_inference(input_ids.unsqueeze(0),
                                                        storage_ids,
                                                        position_ids.unsqueeze(0),
                                                        attn_mask[None, None])
        speculated_tokens.append(logits[0, -1].argmax().item())
        return True

    def set_kv_len(self, kv_len: int):
        self.tiny_model_engine.set_kv_len(kv_len)
        self.num_nodes = kv_len
        self.n_read = kv_len
        self.model.set_kv_len(kv_len)

    def initialize_cuda_graph(self, *args, **kwargs):
        return self.model.initialize_cuda_graph(*args, **kwargs)

    def clear_kv(self):
        return self.model.clear_kv()
