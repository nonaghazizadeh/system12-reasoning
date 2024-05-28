import logging
import numpy as np
import random
import torch
import datetime
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import TrainerCallback
from copy import deepcopy
# from custom_trainer import CustomTrainer, CustomTrainerWithConstantNoiseMatrix, CustomTrainerFocalLoss

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
