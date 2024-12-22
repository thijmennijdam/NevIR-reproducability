
# Official repo of "Reproducing NevIR: Negation in Neural Information Retrieval:
As there are lots of different models we are benchmarking and most libraries that we use all have their own dependencies, we need different environments for different models. All models that we evaluated in our paper can be reproduced. Simply follow the commands described in the section "How to reproduce". For some models you have to ask permission on huggingface first, this usually is accepted pretty quickly.

This repository is organized as follows:

```
src  
├── evaluate/      # Contains scripts for evaluating models, primarily on the NevIR dataset.  
├── finetune/      # Includes code for finetuning models and evaluating them on various development datasets (e.g., NevIR, MS MARCO).  
├── external/      # Holds external repositories such as ColBERT, pygaggle, MTEB, and RankGPT used in the project.  

data/              # Stores all datasets. Use `data.py` to download the required data.  
requirements/      # Contains requirement files needed to set up different environments.  
models/            # Contains model checkpoints and weights used during training and evaluation.  
```

To manage the different virual environtments it is highly recommended to use uv: https://docs.astral.sh/uv/. It is an extremely fast (10-100x faster than pip) package manager and can easily be used to create vens with different python version which we will need.  


# Evaluate on NevIR

To evaluate the following models:
-  cross-encoder/qnli-electra-base
-  cross-encoder/stsb-roberta-large
-  cross-encoder/nli-deberta-v3-base
-  msmarco-bert-base-dot-v5
-  multi-qa-mpnet-base-dot-v1
-  DPR
-  msmarco-bert-co-condensor
-  DRAGON 

Run the following commands:

```bash
uv venv .envs/dense_env --python 3.10
source .envs/dense_env/bin/activate
uv pip install -r requirements/requirements_dense.txt
uv pip install -e .
uv run python src/evaluate/evaluate_dense.py
```

------

To evaluatue the following models:

- TF-IDF
- SPLADEv2 ensemble-distill
- SPLADEv2 self-distill
- SPLADEv3

Run the following commands:

```bash
uv venv .envs/sparse_env --python 3.10
source .envs/sparse_env/bin/activate
uv pip install -r requirements/requirements_sparse.txt
uv pip install transformers==4.29.0
uv pip install -e .
uv run python src/evaluate/evaluate_sparse.py
```

To evaluate the following models:
-   MonoT5 small (Nogueira et al., 2020)
-   MonoT5 base (default) (Nogueira et al., 2020)
-   MonoT5 large (Nogueira et al., 2020)
-   MonoT5 3B (Nogueira et al., 2020)

Run the following commands:

```bash
module load 2023
module load Anaconda3/2023.07-2
conda env create -f requirements/rerank.yml
source activate rerank
pip install -r requirements/requirements_rerank.txt
pip uninstall spacy thinc pydantic
pip install spacy thinc pydantic
pip install -e .
python src/evaluate/evaluate_rerankers.py
```

------

To evaluate the following models:
- ColBERTv1
- ColBERTv2

Run the following commands to download the ColBERT weights:
```bash
mkdir -p ./models/colbert_weights
chmod +x ./requirements/install_git_lfs.sh
./requirements/install_git_lfs.sh
git clone https://huggingface.co/orionweller/ColBERTv1 models/colbert_weights/ColBERTv1
git clone https://huggingface.co/colbert-ir/colbertv2.0 models/colbert_weights/colbertv2.0
```

And the following commands to evaluate them:
```bash
uv venv .envs/colbert_env --python 3.10
source .envs/colbert_env/bin/activate
uv pip install -r requirements/requirements_colbert.txt
uv pip install src/external/ColBERT
uv pip install -e .
uv run python src/evaluate/evaluate_colbert.py
```

------

To evaluate the following models:
- GritLM-7B
- RepLlama
- promptriever-llama3.1-8b-v1
- promptriever-mistral-v0.1-7b-v1
- OpenAI text-embedding-3-small
- OpenAI text-embedding-3-large
- gte-Qwen2-1.5B-instruct 
- gte-Qwen2-7B-instruct
- bge-reranker-base
- bge-reranker-base
- bge-reranker-v2-m3
- jina-reranker-v2-base-multilingual

We are going to use a fork of the [MTEB GitHub repository](https://github.com/thijmennijdam/mteb) in which we fixed some minor bugs, added a few custom rerankers and added our custom code for running the models. First clone this fork:

```bash
git clone --branch old_version_v2 https://github.com/thijmennijdam/mteb.git src/external/mteb
cd src/external/mteb
```

Now create environment and install additional libraries. 
```bash
uv sync
source .venv/bin/activate
uv pip install 'mteb[peft]'
uv pip install 'mteb[jina]'
uv pip install mteb[flagembedding]
uv pip install gritlm
uv pip install --upgrade setuptools
uv pip uninstall triton
uv pip install triton
uv pip install -e ./../../..
```

For these models, you need to download and log into hugginface:

```bash
uv pip install hugginface
huggingface-cli login
```

Evaluate the bi-encoders with the following commands (Note that you need to set the environment variable 'OPENAI_API_KEY' before being able to run the OpenAI models):

```bash
uv run python eval_nevir.py --model "castorini/repllama-v1-7b-lora-passage"
uv run python eval_nevir.py --model "GritLM/GritLM-7B"
uv run python eval_nevir.py --model "text-embedding-3-small"
uv run python eval_nevir.py --model "text-embedding-3-large"
uv run python eval_nevir.py --model "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
uv run python eval_nevir.py --model "Alibaba-NLP/gte-Qwen2-7B-instruct"
uv run python eval_nevir.py --model "samaya-ai/promptriever-llama2-7b-v1"
uv run python eval_nevir.py --model "samaya-ai/promptriever-mistral-v0.1-7b-v1"
```

And the rerankers, make sure to add a path to any previous result from bi-encoders. Example: ```--previous_results "results/castorini_repllama-v1-7b-lora-passage/NevIR_default_predictions.json"```. In our case which bi-encoder does not matter as we are only reranking two documents.:

```bash
uv run python eval_nevir.py --model "jinaai/jina-reranker-v2-base-multilingual" --previous_results "path/to/results"
uv run python eval_nevir.py --model "BAAI/bge-reranker-base" --previous_results "path/to/results"
uv run python eval_nevir.py --model "BAAI/bge-reranker-base" --previous_results "path/to/results"
uv run python eval_nevir.py --model "BAAI/bge-reranker-v2-m3" --previous_results "path/to/results"
```

The RankLlama reranker is not integrated yet in MTEB, so you need to run a seperate script for this. We run it in the mteb environment, so simply:

```bash
uv run python ../../evaluate/evaluate_rankllama.py
```

----------

To evaluate the RankGPT models:
- RankGPT 4o-mini
- RankGPT 4o

Use any environment, make sure to install openai library and pass your OpenAI key as argument:

```bash
uv pip install openai==1.56.1
uv run python src/evaluate/evaluate_rankgpt.py --api_key "your_api_key"
```

To evaluate the following models:
- Qwen2-1.5B-Instruct 
- Qwen2-7B-Instruct 
- Llama-3.2-3B-Instruct 
- Llama-3.1-7B-Instruct 
- Mistral-7B-Instruct-v0.3 

Run the following commands:
```bash
uv venv .envs/llm_env --python 3.10
source .envs/llm_env/bin/activate
uv pip install -r requirements/requirements_llms.txt 
uv pip install -e .
uv run python src/evaluate/evaluate_llms.py 
```

# Finetune experiments
In this section we describe how to reproduce the finetuning experiments. First the necessary data needs to be downloaded. Simply use any of the provided environments and make sure to install gdown, polars and sentence_transformers. 

```bash
uv pip install gdown sentence_transformers polars
uv run python src/data.py
```

The following data is now installed:
- NevIR data train, validation, and test triplets in tsv files
- MSMarco collection and top1000 documents, including a custom file with only the 500 queries.
- ExcluIR data
- A merged dataset of MSMarco and Nevir under the merged_dataset folder
- A merged dataset of NevIR and ExcluIR under the Exclu_NevIR_data folder

NOTE: In the following sections we added example commands for the experiments of finetuning on NevIR and subsequently evaluating these checkpoints on NevIR and MSMarco data. One can reproduce all our experiments simply by change on which data a model is finetuned or evaluated with by providing a different path to the arguments. For example, ```--triples data/NevIR_data/train_triplets.tsv``` finetunes on NevIR data, while ```data/merged_dataset/train_samples.tsv``` finetunes on the merged dataset.


## ColBERTv1

To finetune the ColBERTv1 model activate the colbert env:

```bash
source .envs/colbert_env/bin/activate
```

Then run the following command to finetune on NevIR data:
```bash
uv run python src/finetune/colbert/finetune_colbert.py \
--amp \
--doc_maxlen 180 \
--mask-punctuation \
--bsize 32 \
--accum 1 \
--triples data/NevIR_data/train_triplets.tsv \
--root models/checkpoints/colbert/finetune_nevir \
--experiment NevIR \
--similarity l2 \
--run nevir \
--lr 3e-06 \
--checkpoint models/colbert_weights/ColBERTv1/colbert-v1.dnn
```

Or change the --triples argument to train on other data. Make sure to store the checkpoints in a different folder with the --root argument. The tarting weights are gathered from the --checkpoint argument, these should not be changed.

Afterwards, run the following command to evaluate each checkpoint on NevIR:
  
```bash 
uv run python src/finetune/colbert/dev_eval_nevir.py \
--num_epochs 20 \
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/ \
--output_path results/colbert/finetune_nevir/NevIR_performance
```

and on MSMarco:

```bash
yes | uv run python src/finetune/colbert/dev_eval_msmarco.py \
--num_epochs 20 \
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/ \
--output_path results/colbert/finetune_nevir/MSMarco_performance \
--experiment NevIR_ft
```

Again, change the checkpoint path and output path based on the experiment.

To evaluate the best model on test NevIR and ExcluIR run:

```bash
yes | uv run python src/finetune/colbert/test_eval_nevir_excluir.py \ 
--check_point_path models/checkpoints/colbert/finetune_nevir/NevIR/finetune_colbert.py/nevir/checkpoints/colbert-19.dnn \
--output_path_nevir results/colbert/finetune_nevir/best_model_performance/NevIR \
--output_path_excluIR results/colbert/finetune_nevir/best_model_performance/excluIR \
--excluIR_testdata data/ExcluIR_data/excluIR_test.tsv 
```

The --output_path arguments are for storage of the results. 

## MultiQA-mpnet-base
To finetune MultiQA model we can reuse the colbert environment.

```bash
source .envs/colbert_env/bin/activate
```

This file finetunes the model on NevIR data and evaluates on NevIR and MSMarco after each epoch, so simply run:

```bash
uv run python src/finetune/multiqa/nevir_finetune.py
```

To finetune on the merged dataset, run:
```bash
uv run python src/finetune/multiqa/merged_finetune.py
```


To evaluate the best model on test NevIR and ExcluIR run:
```bash
uv run python src/finetune/multiqa/evaluate_excluIR_multiqa.py
uv run python src/finetune/multiqa/evaluate_nevIR_multiqa.py
```

## MonoT5-base-MSMarco-10k

We need to setup a seperate environment to finetune monot5, as we need python 3.8: 

```bash
uv venv .envs/finetune_monot5_env --python 3.8
source .envs/finetune_monot5_env/bin/activate
uv pip install -r requirements/requirements_finetune_monot5.txt
```

To finetune on NevIR, run:
```bash
uv run python src/finetune/monot5/finetune_monot5.py
```

To evaluate monot5, we will go back to the rerank environment 
```bash
module load 2023 && module load Anaconda3/2023.07-2 && source activate rerank
```

To evaluate on NevIR, run:
```bash
python src/finetune/monot5/dev_eval_nevir.py
```

and to evaluate on MSMarco, run:
```bash
python src/finetune/monot5/dev_eval_msmarco.py \
--model_name_or_path models/checkpoints/monot5/finetune_nevir/checkpoint-29 \
--initial_run data/MSmarco_data/top1000.txt \
--corpus data/MSmarco_data/collection.tsv \
--queries data/MSmarco_data/queries.dev.small.tsv \
--output_run results/monot5/eval_msmarco \
--qrels data/MSmarco_data/qrels.dev.small.tsv
```

----------