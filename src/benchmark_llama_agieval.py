import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import create_logger, LocalDecoder, answer_cleansing, InstructionTunedDecoder
from custom_dataset_agieval import BenchmarkDataset
from transformers import set_seed
import re
@torch.inference_mode
def main():
    args = parse_arguments()
    output_directory = os.path.join("experiments", 'benchmark-agieval',
                                    args.model.split("/")[-2], args.model.split("/")[-1], args.dataset)
    os.makedirs(output_directory, exist_ok=True)
    csv_file = os.path.join(output_directory, "result.csv")
    logger = create_logger(output_directory)
    
    logger.info('*****************************')
    logger.info(args)
    logger.info('*****************************')

    set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("................................................")
    print(args.model)

    if "instruction_tunning" in args.model:
        print()
        decoder = InstructionTunedDecoder(model_name_or_path=args.model,
                                          batch_size=args.batch_size, device=device)
    else:
        decoder = LocalDecoder(model_name_or_path=args.model,
                               batch_size=args.batch_size, device=device)

    logger.info("setup data loader ...")
    dataset = BenchmarkDataset(args)
 
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size)

    total = 0
    correct_list = []
    csv_data = {
        "input": [],
        "pred_before": [],
        "pred_after": [],
        "GT": [],
    }
    for data in tqdm(dataloader):
        x, y = data

        x = list(x)
        z = decoder.decode(x)
        z2 = [temp + "\n" + temp_out + args.direct_answer_trigger for temp, temp_out in zip(x, z)]
        pred = decoder.decode(z2)
        
        csv_data["input"] += z2
        csv_data["pred_before"] += pred

        pred = answer_cleansing(args, pred)
        # pred = clean_pred(pred)

        csv_data["pred_after"] += pred
        csv_data["GT"] += y
        
        correct = (np.array(pred) == np.array(y)).sum().item()
        correct_list.append(correct)
        total += len(y)

        if (args.limit_dataset_size != 0) and ((total+1) >= args.limit_dataset_size):
            break

    accuracy = (sum(correct_list) * 1.0 / total) * 100
    logger.info(f"accuracy : {accuracy}")

    data = pd.DataFrame(csv_data)
    #save the data to a csv file    
    data.to_csv(csv_file, index=False)
    logger.info(f"data saved to {csv_file}")


def clean_pred(preds):
    clean_preds = []
    print("in function clean_pred")
    print(preds)
    for pred in preds:
        print(len(pred))
        print(type(pred))
        if len(pred) > 0:
            print(pred)
            pred = pred.replace("\\boxed{", "")
            #just replace the last occurance of } not all of them
            pred = pred.rstrip("}")
            print(pred)
            print("--------------------------------")
        else:
            pred = ""
        clean_preds.append(pred)
    return clean_preds

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int,
                        default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["agieval"],
        help="dataset used for experiment"
    )

    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size.")

    parser.add_argument("--max_num_worker", type=int, default=3,
                        help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot",
        choices=["zero_shot", "role_play"], help="method"
    )

    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )

    args = parser.parse_args()
    
    if args.dataset == "agieval":
        args.dataset_path = "./data/benchmark/agieval/output.json"
        args.direct_answer_trigger = "\nGive the final answer answer in the format of: The final answer is: \$ \\boxed{...} \$"
    
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
