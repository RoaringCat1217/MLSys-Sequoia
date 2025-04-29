import torch
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from utils import _make_causal_mask
from utils import _merge_lists
import numpy as np


class OPTTree:
    def __init__(self,
                 draft_model_engine: GraphInferenceEngine,
                 target_model_engine: GraphInferenceEngineTG,
                 prefix: torch.LongTensor,
                 max_length=256,
                 device: str = 'cpu',
                 dtype = torch.float16,
                 vocab_size=32000,
                 sampling_callables=None,
                 n_spec=128,
                 eps=0.1):
        self.num_nodes = len(prefix)  # prefix + accepted tokens
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        assert self.max_length == draft_model_engine.engine.max_length
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.prefill_target = True
        self.sampling_callables = sampling_callables
        self.n_spec = n_spec
        self.eps = eps

        # prefilled and generated tokens
        self.tokens = torch.zeros(self.max_length, device=self.device).long()
        self.tokens[:len(prefix)] = prefix.to(self.device)
        # a token's offset in a sequence
        self.position_ids = torch.zeros(self.max_length, device=self.device).long()
        self.position_ids[:len(prefix)] = torch.arange(len(prefix))
        # which slot in kv cache does a token go to
        self.storage_ids = torch.arange(self.max_length, device=self.device)
        # store a full attention mask to avoid redundant mask creation
        self.attn_mask = _make_causal_mask((1, self.max_length), dtype=self.dtype, device=self.device)

        # drafted probabilities
        self.drafted_probs = np.zeros((self.n_spec, self.n_spec), dtype=np.float32)
        # drafted tokens
        self.drafted_tokens = np.zeros((self.n_spec, self.n_spec), dtype=np.int64)

        # prefill
        draft_model_outputs = self.draft_model_engine.inference(
            input_ids=self.tokens[:self.num_nodes].unsqueeze(0),
            storage_ids=self.storage_ids[:self.num_nodes],
            position_ids=self.position_ids[:self.num_nodes].unsqueeze(0),
            attn_mask=self.attn_mask[None, None, :self.num_nodes]
        ) # (1, num_nodes, vocab_size)

        probs, tokens = self.sampling_callables[self.n_spec](draft_model_outputs[0, -1])
        self.drafted_probs[0] = probs.cpu().numpy()
        self.drafted_tokens[0] = tokens.cpu().numpy()
        self.selected_tokens = [(self.drafted_probs[0, j].item(), (0, j)) for j in range(self.n_spec)]
        self.E = sum([x[0] for x in self.selected_tokens])

    @torch.inference_mode()
    def draft_step(self, step: int) -> float:
        num_nodes = self.num_nodes + step
        tokens = torch.tensor([self.drafted_tokens[step - 1, 0]], device=self.device).long()
        position_ids = torch.tensor([num_nodes - 1], device=self.device).long()
        storage_ids = torch.tensor([num_nodes - 1], device=self.device).long()
        attn_mask = self.attn_mask[num_nodes - 1: num_nodes]

        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids=tokens.unsqueeze(0),
            storage_ids=storage_ids,
            position_ids=position_ids.unsqueeze(0),
            attn_mask=attn_mask[None, None, ...]
        )
        new_probs, new_tokens = self.sampling_callables[self.n_spec](draft_model_outputs[0, -1])
        self.drafted_probs[step] = new_probs.cpu().numpy() * self.drafted_probs[step - 1, 0]
        self.drafted_tokens[step] = new_tokens.cpu().numpy()
        selected_tokens = [(self.drafted_probs[step, j].item(), (step, j)) for j in range(self.n_spec)]
        self.selected_tokens = _merge_lists(self.selected_tokens, selected_tokens)
        E = sum([x[0] for x in self.selected_tokens])
        delta = E - self.E
        self.E = E
        return delta

    def draft(self):
        for step in range(1, self.n_spec):
            if self.num_nodes + step + 1 >= self.max_length:
                break
            delta = self.draft_step(step)
            if delta < self.eps:
                break
            # </s> token
            if self.drafted_tokens[step, 0] == 2:
                break

    @torch.inference_mode()
    def verify(self):
        # create spec_tokens, tree_mask and position_ids
        coords = [x[1] for x in self.selected_tokens]
        coords.sort() # sorted by the depth in the tree
        k = 0
        tree = [] # map (i, j) to linearized index k
        for i, j in coords:
            if i == len(tree):
                tree.append([])
            tree[-1].append(k)
            k += 1
        spec_tokens = np.zeros(self.n_spec, dtype=np.int64)
        tree_mask = np.zeros((self.n_spec, self.n_spec), dtype=np.bool_)
        position_ids = np.zeros(self.n_spec, dtype=np.int64)
        expand_indices = [] # the k index of all expanding nodes
        k = 0
        for i, j in coords:
            if self.num_nodes + k >= self.max_length:
                break
            spec_tokens[k] = self.drafted_tokens[i, j]
            tree_mask[k, expand_indices[:i]] = True
            tree_mask[k, k] = True
            position_ids[k] = i
            if j == 0:
                expand_indices.append(k)
            k += 1
        position_ids += self.num_nodes
        # now k is the number of nodes

        # move ndarray to tensor
        self.tokens[self.num_nodes:self.num_nodes + k] = torch.tensor(spec_tokens[:k], dtype=torch.long, device=self.device)
        self.position_ids[self.num_nodes:self.num_nodes + k] = torch.tensor(position_ids[:k], dtype=torch.long, device=self.device)
        tree_mask = torch.tensor(tree_mask[:k, :k], dtype=self.dtype, device=self.device)
        tree_mask = torch.where(tree_mask > 0, 0, torch.finfo(self.dtype).min)
        attn_mask = torch.full((k + 1, self.num_nodes + k), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        attn_mask[:, :self.num_nodes] = 0
        attn_mask[1:, self.num_nodes:] = tree_mask

        if self.prefill_target:
            # the first call to target
            self.prefill_target = False
            full_attn_mask = torch.full((self.num_nodes + k, self.num_nodes + k), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
            full_attn_mask[:self.num_nodes, :self.num_nodes] = self.attn_mask[:self.num_nodes, :self.num_nodes]
            full_attn_mask[self.num_nodes:] = attn_mask[1:]
            target_model_outputs = self.target_model_engine.inference(
                input_ids=self.tokens[:self.num_nodes + k].unsqueeze(0),
                position_ids=self.position_ids[:self.num_nodes + k].unsqueeze(0),
                attn_mask=full_attn_mask[None, None, ...],
                storage_ids=self.storage_ids[:self.num_nodes + k]
            )
            target_model_outputs = target_model_outputs[0, self.num_nodes - 1:]
        else:
            target_model_outputs = self.target_model_engine.inference(
                input_ids=self.tokens[self.num_nodes - 1:self.num_nodes + k].unsqueeze(0),
                position_ids=self.position_ids[self.num_nodes - 1:self.num_nodes + k].unsqueeze(0),
                attn_mask=attn_mask[None, None, ...],
                storage_ids=self.storage_ids[self.num_nodes - 1:self.num_nodes + k]
            )
            target_model_outputs = target_model_outputs[0]
        target_tokens = target_model_outputs.argmax(dim=-1).cpu().numpy()

        # traverse the tree layer by layer
        terminal = False
        target_token = target_tokens[0]
        accept_indices = []
        accept_tokens = []
        for layer in tree:
            if target_token in [0, 2]:
                # <unk> or </s>
                terminal = True
                break
            j, = np.where(spec_tokens[layer] == target_token)
            if len(j) == 0:
                break
            j = j[0]
            k = layer[j]
            accept_indices.append(k)
            accept_tokens.append(target_token)
            target_token = target_tokens[k + 1]
            if j != 0:
                break
        accept_tokens.append(target_token)

        # update
        if self.max_length - self.num_nodes < len(accept_tokens):
            accept_tokens = accept_tokens[:self.max_length - self.num_nodes]
            terminal = True
        self.tokens[self.num_nodes:self.num_nodes + len(accept_tokens)] = torch.tensor(accept_tokens, dtype=torch.long, device=self.device)
        self.position_ids[self.num_nodes:self.num_nodes + len(accept_tokens)] = torch.arange(self.num_nodes, self.num_nodes + len(accept_tokens), dtype=torch.long, device=self.device)
        if terminal:
            self.num_nodes += len(accept_tokens)
            return len(accept_tokens), terminal

        # consolidate caches
        from_indices = np.array(accept_indices) + self.num_nodes
        to_indices = np.arange(self.num_nodes, self.num_nodes + len(accept_indices))
        self.target_model_engine.engine.kv_cache.move_kv_indices(from_indices, to_indices, self.num_nodes + len(accept_indices))
        matched = 0
        for k in accept_indices:
            if k == expand_indices[matched]:
                matched += 1
            else:
                break
        assert len(accept_indices) - matched in [0, 1]
        self.draft_model_engine.set_kv_len(self.num_nodes + matched)
        self.num_nodes += len(accept_tokens)

        # prepare for the next draft
        # feed the remaining accepted tokens to draft model
        remain = len(accept_indices) - matched + 1
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids=self.tokens[self.num_nodes - remain:self.num_nodes].unsqueeze(0),
            storage_ids=self.storage_ids[self.num_nodes - remain:self.num_nodes],
            position_ids=self.position_ids[self.num_nodes - remain:self.num_nodes].unsqueeze(0),
            attn_mask=self.attn_mask[None, None, self.num_nodes - remain:self.num_nodes]
        )
        if draft_model_outputs.shape[1] == 0:
            return len(accept_tokens), True
        probs, tokens = self.sampling_callables[self.n_spec](draft_model_outputs[0, -1])
        self.drafted_probs[0] = probs.cpu().numpy()
        self.drafted_tokens[0] = tokens.cpu().numpy()
        self.selected_tokens = [(self.drafted_probs[0, j], (0, j)) for j in range(self.n_spec)]
        self.E = sum([x[0] for x in self.selected_tokens])

        return len(accept_tokens), terminal