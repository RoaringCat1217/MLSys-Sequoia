import sys

from Engine.SpeculatedGraphInferenceEngine import SpeculatedGraphInferenceEngine
from Tree.OPTTree import OPTTree

sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval, convert_wikimqa_dataset, \
    convert_qasper_dataset
import argparse
import time
from utils import _make_causal_mask, cuda_graph_for_residual, cuda_graph_for_sampling_argmax, graph_for_residual, \
    graph_for_sampling_argmax, graph_for_sampling_opttree
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from Engine.offload_engine import OffloadEngine
import random

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='tiny model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--seed', type=int, default=17, help='random seed')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--S', type=int, default=128, help='prefill length')
parser.add_argument('--spec_steps', type=int, default=5, help='size of draft tree')
args = parser.parse_args()
print(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
eval_list = list(range(0, 200))

random.shuffle(eval_list)

if args.dataset == 'openwebtext':
    tokenized_dataset_eval = load_from_disk("../dataset/openwebtext_eval").select(eval_list[args.start:args.end])
elif args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer, seq_len=args.S + 128).select(
        eval_list[args.start:args.end])
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer, seq_len=args.S + 128).select(
        eval_list[args.start:args.end])
elif args.dataset == 'wikimqa':
    tokenized_dataset_eval = convert_wikimqa_dataset(tokenizer=tokenizer, seq_len=args.S + 128).select(
        eval_list[args.start:args.end])
elif args.dataset == 'qasper':
    tokenized_dataset_eval = convert_qasper_dataset(tokenizer=tokenizer, seq_len=args.S + 128).select(
        eval_list[args.start:args.end])
else:
    tokenized_dataset_eval = convert_c4_dataset_eval(tokenizer=tokenizer, seq_len=args.S + 128).select(
        eval_list[args.start:args.end])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)

tiny_model = GraphInferenceEngine(max_length=args.M, model_name_or_path=args.model, dtype=torch.float16, device="cpu")
model = SpeculatedGraphInferenceEngine(
    tiny_model_engine=tiny_model,
    spec_steps=args.spec_steps,
    max_length=args.M, model_name_or_path=args.target, dtype=torch.float16, device="cpu"
)

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

num_eval_steps = len(dataloader)
num_decoding_steps = 0
num_large_model_steps = 0
total_time = 0.0
dtype = torch.float16

with torch.no_grad():
    for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        input_ids = torch.zeros(args.M, dtype=torch.long)
        storage_ids = torch.arange(args.M, dtype=torch.long)
        position_ids = torch.arange(args.M, dtype=torch.long)
        attn_mask = _make_causal_mask((1, args.M), dtype=dtype, device='cpu')
        # input_ids and labels are the same here. labels are typically used for training target
        input_ids[:args.S] = batch['input_ids'][0, :args.S]
        labels = batch['labels'][..., :args.S]

        length = args.S
        logits = model.inference(input_ids[None, :length],
                                 storage_ids[:length],
                                 position_ids[None, :length],
                                 attn_mask[None, None, :length])
        next_token = logits[0, -1].argmax()
        input_ids[length] = next_token
        length += 1

        while length < args.M and next_token != 2:
            logits = model.graph_inference(input_ids[None, length - 1:length],
                                           storage_ids[length - 1:length],
                                           position_ids[None, length - 1:length],
                                           attn_mask[None, None, length - 1:length])
            next_token = logits[0, -1].argmax()
            input_ids[length] = next_token
            length += 1
