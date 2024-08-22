import logging
import numpy as np
import random
import torch
import datetime
import os
import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import TrainerCallback
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


dataset2label = {
    "personal": ["tpa", "oa", "ra"],
    "ucc": ["antagonize", "condescending", "hostile"],
    "ghc":     ["vo", "hd", "cv"],
    "imdb":     ["sentiment"],
    "agnews":     ["category"],
}

label2dataset = {label: dataset for dataset,
                 labels in dataset2label.items() for label in labels}


# def get_trainer(trainer_type=None):
#     if trainer_type == "constant_noise_matrix":
#         return CustomTrainerWithConstantNoiseMatrix
#     elif trainer_type == "withfocalloss":
#         return CustomTrainerFocalLoss
#     else:
#         return CustomTrainer


def add_label_noise(example, noise_chance, label_col="label"):
    # Add a column to record whether the label was flipped or not
    example['label_flipped'] = np.random.rand() < noise_chance

    # Flip the label based on the noise_chance
    example[label_col] = 1 - \
        example[label_col] if example['label_flipped'] else example[label_col]

    return example


def introduce_noise(df, label_col, noise_ratio):
    """
    Introduce noise to the label column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - label_col (str): Name of the label column to introduce noise.
    - noise_ratio (float): Ratio of rows to introduce noise to (value between 0 and 1).

    Returns:
    - pd.DataFrame: DataFrame with introduced noise in the label column.
    """

    # Validate the input parameters
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in the DataFrame.")

    if not (0 <= noise_ratio <= 1):
        raise ValueError("Noise ratio must be a value between 0 and 1.")

    # Determine the number of rows to introduce noise to
    num_rows = int(noise_ratio * len(df))

    # Randomly select rows to introduce noise
    noisy_rows = np.random.choice(df.index, size=num_rows, replace=False)

    # Flip the labels in the selected rows
    df.loc[noisy_rows, label_col] = 1 - df.loc[noisy_rows, label_col]

    return df


class EvalOnTrainCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="")
            return control_copy


class LogPredicitonsCallback(TrainerCallback):
    def __init__(self, logger, trainer, output_dir) -> None:
        super().__init__()
        self._logger = logger
        self._trainer = trainer
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        print("in here logging predictions")

        for split in ["train", "val"]:
            if split == "train":
                dataset = self._trainer.train_dataset
            else:
                dataset = self._trainer.eval_dataset

            predictions = self._trainer.predict(
                test_dataset=dataset).predictions

            preds = np.argmax(predictions, axis=1)
            if os.path.exists(os.path.join(self.output_dir, f"{split}_preds.csv")):
                self._logger.info(f"Loading prev preds!!")
                predictions_df = pd.read_csv(os.path.join(
                    self.output_dir, f"{split}_preds.csv"))
                pred_columns_ids = [int(c.split("_")[-1])
                                    for c in predictions_df.columns if "pred" in c]
                pred_col = f"pred_{max(pred_columns_ids)+1}"
                self._logger.info(
                    f"Adding predictions to column {pred_col} to {split}_preds.csv")
                predictions_df[pred_col] = preds
            else:
                self._logger.info(f"First epoch, no prev preds")
                labels = dataset['label']
                try:
                    flipped_labels = dataset['label_flipped']
                    ids = dataset['__index_level_0__']
                except:
                    self._logger.info(
                        f"No label_flipped column for split {split}")
                    self._logger.info(
                        f"No index_level_0 ids column for split {split}")
                    flipped_labels = [False]*len(labels)
                    ids = [i for i in range(len(labels))]

                pred_col = "pred_1"
                predictions_df = pd.DataFrame(
                    {"id": ids, "label_flipped": flipped_labels, "label": labels,  pred_col: preds})
            predictions_df.to_csv(os.path.join(
                self.output_dir, f"{split}_preds.csv"), index=False)


def compute_metrics(p):
    # Convert probabilities to predictions
    predictions = np.argmax(p.predictions, axis=1)

    # Assuming the labels are 0 and 1
    labels = p.label_ids

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Use probabilities of class 1 for AUC-ROC
    auc_roc = roc_auc_score(labels, p.predictions[:, 1])

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
    }


def load_ghc_data():
    data_path = "./data/GHC/"

    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).dropna()
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv")).dropna()
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).dropna()
    return {"train": train_df, "val": val_df, "test": test_df}
    # return pd.concat([train_df, val_df, test_df])


def load_personal_attack_data():
    data_path = "./data/personal_attack/"

    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).rename(
        columns={"comment": "text"}).dropna()
    val_df = pd.read_csv(os.path.join(data_path, "dev.csv")).rename(
        columns={"comment": "text"}).dropna()
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).rename(
        columns={"comment": "text"}).dropna()
    # return pd.concat([train_df, val_df, test_df])
    return {"train": train_df, "val": val_df, "test": test_df}


def load_ucc_data():
    data_path = "./data/ucc/"

    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).rename(
        columns={"comment": "text"}).dropna()
    val_df = pd.read_csv(os.path.join(data_path, "dev.csv")).rename(
        columns={"comment": "text"}).dropna()
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).rename(
        columns={"comment": "text"}).dropna()
    return {"train": train_df, "val": val_df, "test": test_df}
    # return pd.concat([train_df, val_df, test_df])


def load_jigsaw_data():
    data_dir = "../../Data/Jigsaw Annotations/8670-items(re-annotated one VO questions version)"
    dfs = []
    for i in range(1, 6):
        annotator_file = f"annotator{i}.json"
        annotator_path = os.path.join(data_dir, annotator_file)
        annotator_df = pd.read_json(annotator_path, lines=True)
        annotator_df['annotator'] = i
        dfs.append(annotator_df)

    all_annotations_df = pd.concat(dfs)
    all_annotations_df
    LABELS = ['CV', 'HD', 'VO', 'NH', 'RAE', 'NAT', 'GEN', 'REL',
              'SXO', 'IDL', 'POL', 'MPH', 'Explicit', 'Implicit']
    all_annotations_df['accept'] = all_annotations_df['accept'].apply(
        lambda x: x if isinstance(x, list) else [])
    # Create new columns for each value in LABELS
    for label in LABELS:
        all_annotations_df[label] = all_annotations_df['accept'].apply(
            lambda x: 1 if label in x else 0)

    return all_annotations_df


def load_imdb_data():
    data_path = "./data/IMDB"

    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).rename(
        columns={'review': 'text'})
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv")
                         ).rename(columns={'review': 'text'})
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).rename(
        columns={'review': 'text'})

    sentiment_mapping = {'positive': 1, 'negative': 0}
    for df in [train_df, test_df, val_df]:
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    return {"train": train_df, "val": val_df, "test": test_df}


def load_agnews_data():
    data_path = "./data/AGNEWS"

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    return {"train": train_df, "val": val_df, "test": test_df}


def load_system12_data():
    data_path = "./data/system12"
    label_mapping = {'System 1': 0, 'System 2': 1}
    df = pd.read_csv(os.path.join(data_path, "Cognitive_Biases_Dataset.csv"))
    df['labels'] = df['Strategy'].map(label_mapping)
    return df
    # return pd.concat([train_df, val_df, test_df])


def load_system12_questions_data():
    data_path = "./data/system12"
    df = pd.read_csv(os.path.join(data_path, "Cognitive_Biases_Dataset.csv"))
    df = pd.DataFrame(df['Question'].unique(), columns=['Question'])
    return df


def load_system12_10k_questions_data():
    data_path = "./data/system12"
    df = pd.read_csv(os.path.join(
        data_path, "Cognitive_Biases_Dataset_10000_Examples.csv"))
    df = pd.DataFrame(df['Question'].unique(), columns=['Question'])
    return df


def load_system12_gpt_questions_data():
    data_path = "./data/system12"
    df = pd.read_csv(os.path.join(
        data_path, "gpt_Cognitive_Biases_Dataset.csv"))
    df = pd.DataFrame(df['Question'].unique(), columns=['Question'])
    return df


def load_system12_combined_questions_data():
    data_path = "./data/system12"
    df = pd.read_csv(os.path.join(
        data_path, "combined_cognitive_biases_dataset.csv"))
    df = pd.DataFrame(df['Question'].unique(), columns=['Question'])
    return df


def load_system12_combined_data():
    data_path = "./data/system12"
    label_mapping = {'System 1': 0, 'System 2': 1}
    df = pd.read_csv(os.path.join(
        data_path, "combined_cognitive_biases_dataset.csv"))
    df['labels'] = df['Strategy'].map(label_mapping)
    return df


def load_model_response(data_path):
    df = pd.read_csv(data_path)
    return df


def get_dataset_loader_func(dataset_name):

    if dataset_name == 'system12':
        return load_system12_data()
    elif dataset_name == 'system12_questions':
        return load_system12_data()
    elif dataset_name == 'system12_combined':
        return load_system12_combined_data()
    elif dataset_name == 'system12_combined_questions':
        return load_system12_combined_questions_data()
    elif dataset_name == 'system12_gpt_questions':
        return load_system12_gpt_questions_data()
    elif dataset_name == 'system12_10k_questions':
        return load_system12_10k_questions_data()
    # elif dataset_name == "personal_attack":
    #     return load_personal_attack_data()
    # elif dataset_name == "imdb":
    #     return load_imdb_data()
    # elif dataset_name == "agnews":
    #     return load_agnews_data()
    # elif dataset_name == "jigsaw_mola":
    #     return load_jigsaw_data()
    # elif dataset_name == "ghc":
    #     return load_ghc_data()
    # elif dataset_name == "ucc":
    #     return load_ucc_data()
    else:
        return load_model_response(dataset_name)


def create_logger(save_path, log_level=logging.INFO, prefix=""):
    EXPERIMENT_DIRECTORY = save_path
# Configure the logging settings
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Create a log file with the current date and time in its name
    current_datetime = datetime.datetime.now()
    log_file = current_datetime.strftime(prefix+"%Y-%m-%d_%H-%M-%S.log")

    # Create a file handler to write log messages to the specified log file
    file_handler = logging.FileHandler(
        os.path.join(EXPERIMENT_DIRECTORY, log_file))

    # Set the log level for the file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages (if you want a different format for the log file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def add_pad_token_id(tokenizer, model):
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id") is None:
        model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model


def get_pipeline(model_name_or_path, device):
    # make sure that text generation pipeline is using AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if "Phi" in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
        )

    tokenizer, model = add_pad_token_id(tokenizer, model)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return pipe


class LocalDecoder():
    def __init__(self, model_name_or_path, device, batch_size, MAX_LEN=256):
        self.pipeline = get_pipeline(model_name_or_path, device)
        # self.batch_size = batch_size
        self.MAX_LEN = MAX_LEN

    def decode(self, inputs):
        conversations = []
        for input in inputs:
            conversation = [{"role": "user", "content": input}]
            conversations.append(conversation)
        # conversation = [{"role": "user", "content": input}]
        responses = self.pipeline(conversations,
                                  max_new_tokens=self.MAX_LEN,
                                  #  batch_size=self.batch_size,
                                  #  padding='longest'
                                  )
        content = []
        for response in responses:
            content.append(response[0]['generated_text'][-1]['content'])
        return content


def answer_cleansing(args, preds):

    # print("pred_before : " + pred)

    # if args.method in ("few_shot", "few_shot_cot"):
    #     preds = pred.split(args.direct_answer_trigger_for_fewshot)
    #     answer_flag = True if len(preds) > 1 else False
    #     pred = preds[-1]
    clean_preds = []
    for pred in preds:
        if args.dataset in ("aqua", "commonsensqa"):
            pred = re.findall(r'A|B|C|D|E', pred)
        elif args.dataset == "bigbench_date":
            pred = re.findall(r'A|B|C|D|E|F', pred)
        elif args.dataset in ("object_tracking"):
            pred = re.findall(r'A|B|C', pred)
        elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        elif args.dataset in ("strategyqa", "coin_flip"):
            pred = pred.lower()
            pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
            pred = pred.split(" ")
            pred = [i for i in pred if i in ("yes", "no")]
        elif args.dataset == "last_letters":
            right_index = pred.rfind('"')
            if right_index != -1:
                left_index = pred[:right_index].rfind('"')
                pred = pred[left_index:right_index+1].lower()
            pred = re.sub("\"|\'|\n|\.|\s", "", pred)
            pred = [pred]
        else:
            raise ValueError("dataset is not properly defined ...")

        # If there is no candidate in list, null is set.
        if len(pred) == 0:
            pred = ""
        else:
            if args.method in ("few_shot", "few_shot_cot"):
                if answer_flag:
                    # choose the first element in list ...
                    pred = pred[0]
                else:
                    # choose the last element in list ...
                    pred = pred[-1]
            elif args.method in ("zero_shot", "role_play"):
                # choose the first element in list ...
                pred = pred[0]
            else:
                raise ValueError("method is not properly defined ...")

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

        # print("pred_after : " + pred)
        clean_preds.append(pred)

    return clean_preds


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []

    # example sentences ...
    if args.dataset in ("multiarith", "gsm8k"):

        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")

        x.append(
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")

    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + \
                " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + \
                " " + y[i] + ".\n\n"

    return demo_text


def create_logger(save_path, log_level=logging.INFO, prefix=""):
    EXPERIMENT_DIRECTORY = save_path
    # Configure the logging settings
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Create a log file with the current date and time in its name
    current_datetime = datetime.datetime.now()
    log_file = current_datetime.strftime(prefix+"%Y-%m-%d_%H-%M-%S.log")

    # Create a file handler to write log messages to the specified log file
    file_handler = logging.FileHandler(
        os.path.join(EXPERIMENT_DIRECTORY, log_file))

    # Set the log level for the file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages (if you want a different format for the log file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def get_tokenizer(model_name_or_path):
    if any(k in model_name_or_path.lower() for k in ("gemma", "llama", "gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    print(f"Padding side: {padding_side}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side=padding_side)

    return tokenizer
