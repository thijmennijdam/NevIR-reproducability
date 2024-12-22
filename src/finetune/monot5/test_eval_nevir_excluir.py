import os
import random
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from sentence_transformers import InputExample
from transformers import T5ForConditionalGeneration
from external.pygaggle.rerank.transformer import MonoT5
from finetune.monot5.pairwise_evaluator import PairwiseEvaluator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def convert_dataset_to_triplets(df, evaluation: bool = False):
    """
    Convert DataFrame into triplets for evaluation.
    """
    all_instances = []
    for _, row in df.iterrows():
        if evaluation:
            all_instances.append(
                InputExample(texts=[row["q1"], row["q2"], row["doc1"], row["doc2"]])
            )
    print("Number of instances:", len(all_instances))
    return all_instances

def eval_monoT5(check_point_path, output_path_nevir, output_path_excluIR, excluIR_path):
    """
    Evaluate MonoT5 on NevIR and ExcluIR test sets.
    """
    _set_seed(42)

    print("LOADING MODEL FROM CHECKPOINT")
    model = T5ForConditionalGeneration.from_pretrained(check_point_path).to(DEVICE)
    reranker = MonoT5(model=model)

    # Evaluation on ExcluIR
    if output_path_excluIR:
        print("Loading ExcluIR test data...")
        excluIR_test_df = pd.read_csv(excluIR_path, sep="\t", header=None, names=["q1", "q2", "doc1", "doc2"])
        excl_test_data = convert_dataset_to_triplets(excluIR_test_df, evaluation=True)
        evaluator_test_excluIR = PairwiseEvaluator.from_input_examples(excl_test_data, name="test", model_name='monot5')

        print("Evaluating the model on the ExcluIR test set...")
        evaluation_accuracy_test = evaluator_test_excluIR(model=reranker, output_path=output_path_excluIR)
        print(f"ExcluIR Test Accuracy: {evaluation_accuracy_test:.4f}")
        print(f"Saved to: {output_path_excluIR}")
    else:
        print('No evaluation on ExcluIR')

    # Evaluation on NevIR
    if output_path_nevir:
        print("Loading NevIR test data...")
        dataset_test = load_dataset("orionweller/NevIR", split="test")
        test_df = dataset_test.to_pandas()
        test_data = convert_dataset_to_triplets(test_df, evaluation=True)
        evaluator_test = PairwiseEvaluator.from_input_examples(test_data, name="test", model_name='monot5')

        print("Evaluating the model on the NevIR test set...")
        evaluation_accuracy_test = evaluator_test(model=reranker, output_path=output_path_nevir)
        print(f"NevIR Test Accuracy: {evaluation_accuracy_test:.4f}")
        print(f"Saved to: {output_path_nevir}")
    else:
        print('No evaluation on NevIR')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MonoT5 model on ExcluIR and NevIR test sets")

    parser.add_argument('--check_point_path', type=str, required=True, 
                        help='Path to the checkpoint directory for MonoT5')
    parser.add_argument('--output_path_nevir', type=str, required=False, default=None,
                        help='Path to save evaluation results for NevIR')
    parser.add_argument('--output_path_excluIR', type=str, required=False, default=None,
                        help='Path to save evaluation results for ExcluIR')
    parser.add_argument('--excluIR_testdata', type=str, required=True, 
                        help='Path to the ExcluIR test data')

    args = parser.parse_args()

    if args.output_path_nevir:
        os.makedirs(args.output_path_nevir, exist_ok=True)
    if args.output_path_excluIR:
        os.makedirs(args.output_path_excluIR, exist_ok=True)
        
    eval_monoT5(
        check_point_path=args.check_point_path, 
        output_path_nevir=args.output_path_nevir, 
        output_path_excluIR=args.output_path_excluIR, 
        excluIR_path=args.excluIR_testdata
    )
