CUDA_VISIBLE_DEVICES=0 python testbed_opt_tree.py --model JackFram/llama-68m --target meta-llama/Llama-2-7b-hf --start 0 --end 200 --M 384 --dataset openwebtext | tee -a /workspace/opt_tree_68_7_open
echo "opt 68 7 open"

CUDA_VISIBLE_DEVICES=0 python testbed_opt_tree.py --model JackFram/llama-68m --target meta-llama/Llama-2-7b-hf --start 0 --end 200 --M 384 --dataset cnn | tee -a /workspace/opt_tree_68_7_cnn
echo "opt 68 7 cnn"


CUDA_VISIBLE_DEVICES=0 python testbed_opt_tree.py --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset cnn | tee -a /workspace/opt_tree_7_13_cnn
echo "opt 7 13 cnn"

CUDA_VISIBLE_DEVICES=0 python testbed_opt_tree.py --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset openwebtext | tee -a /workspace/opt_tree_7_13_openwebtext
echo "opt 7 13 open"

CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/68m_13b/growmaps/L40-CNN-68m-13b-greedy.pt  --Mode greedy --dataset openwebtext | tee -a /workspace/sequoia_greedy_7_13_open
echo "seq 7 13 open"

CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/68m_13b/growmaps/L40-CNN-68m-13b-greedy.pt  --Mode greedy --dataset cnn | tee -a /workspace/sequoia_greedy_7_13_cnn
echo "seq 7 13 cnn"

python3 testbed_double_draft.py --tiny_model JackFram/llama-68m --draft_model TinyLlama/TinyLlama_v1.1 --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset openwebtext --n_spec 128 --spec_steps 6 --eps 0.2 | tee -a /workspace/double_draft_openwebtext_68_1_13
echo "dd 68 1.1 13 open"

python3 testbed_double_draft.py --tiny_model JackFram/llama-68m --draft_model TinyLlama/TinyLlama_v1.1 --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset cnn --n_spec 128 --spec_steps 6 --eps 0.2 | tee -a /workspace/double_draft_cnn_68_1_13
echo "dd 68 1.1 13 cnn"

python3 testbed_double_draft.py --tiny_model JackFram/llama-68m --draft_model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset openwebtext --n_spec 128 --spec_steps 6 --eps 0.2 | tee -a /workspace/double_draft_openwebtext_68_1_13
echo "dd 68 7 13 open"

python3 testbed_double_draft.py --tiny_model JackFram/llama-68m --draft_model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-13b-hf --start 0 --end 200 --M 384 --dataset cnn --n_spec 128 --spec_steps 6 --eps 0.2 | tee -a /workspace/double_draft_cnn_68_1_13
echo "dd 68 7 13 cnn"

