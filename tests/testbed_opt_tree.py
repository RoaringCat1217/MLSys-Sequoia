import sys
sys.path.append("/home/ubuntu/MLSys-Sequoia")

from Tree.OPTTree import OPTTree

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
from utils import cuda_graph_for_sampling_opttree
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from Engine.offload_engine import OffloadEngine
import random

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--seed', type=int, default=17, help='random seed')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--S', type=int, default=128, help='prefill length')
parser.add_argument('--offloading', action='store_true')
parser.add_argument('--n_spec', type=int, default=128, help='size of draft tree')
parser.add_argument('--eps', type=float, default=0.1, help='eps of opt tree algorithm')
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

draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path=args.model, dtype=torch.float16,
                                   device="cuda:0")
if args.offloading:
    target_model = OffloadEngine(max_length=args.M, model_name_or_path=args.target, dtype=torch.float16,
                                 device="cuda:0")
else:
    target_model = GraphInferenceEngineTG(max_length=args.M, model_name_or_path=args.target, dtype=torch.float16,
                                          device="cuda:0", offloading=args.offloading)
graph_capture_list = [1, 2]
draft_model.initialize_cuda_graph(graph_capture_list)
sampling_callables = {args.n_spec: cuda_graph_for_sampling_opttree(num_samples=args.n_spec)}

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

num_eval_steps = len(dataloader)
num_decoding_steps = 0
num_large_model_steps = 0
total_time = 0.0
dtype = torch.float16

with torch.no_grad():
    for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        # input_ids and labels are the same here. labels are typically used for training target
        input_ids = batch['input_ids'][..., :args.S]
        labels = batch['labels'][..., :args.S]
        terminate = False
        if labels[0][-1] == -100:
            terminate = True  # -100 is ignore_index in DataCollatorForLanguageModeling. Not necessary here.
        draft_kv_len = 0
        target_kv_len = 0

        spectree = OPTTree(draft_model_engine=draft_model,
                           target_model_engine=target_model,
                           prefix=input_ids[0],
                           max_length=args.M,
                           device='cuda:0',
                           dtype=dtype,
                           sampling_callables=sampling_callables,
                           n_spec=args.n_spec,
                           eps=args.eps)

        # torch.cuda.synchronize()
        t1 = time.time()
        while input_ids.shape[1] < args.S + 128 and terminate == False:
            spectree.draft()
            n_accepted, terminate = spectree.verify()
            num_decoding_steps += n_accepted
            num_large_model_steps += 1

        # torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        draft_model.clear_kv()
        target_model.clear_kv()
print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(total_time,
                                                                                                  total_time / num_decoding_steps,
                                                                                                  num_decoding_steps,
                                                                                                  num_large_model_steps,
                                                                                                  num_decoding_steps / num_large_model_steps))
