import pandas as pd
import json
from utils import create_logger

algorithm = "dpo"
system = "1"
benchamrk= "sexual_orientation"
prompt = "bias_benchmark_newprompt"

with open(f'./data/bias_benchmark/{benchamrk}.json', 'r') as json_file:
    json_data = json.load(json_file)

context_conditions = [entry['context_condition'] for entry in json_data]

csv_data = pd.read_csv(f'./experiments/{prompt}/{algorithm}/lora-Meta-Llama-3-8B-Instruct-system{system}/{benchamrk}/result.csv')

if len(context_conditions) != len(csv_data):
    raise ValueError("The length of the JSON data and CSV data do not match!")

csv_data['context_condition'] = context_conditions

csv_data.to_csv(f'./experiments/{prompt}/{algorithm}/lora-Meta-Llama-3-8B-Instruct-system{system}/{benchamrk}/cat_result.csv', index=False)

def calculate_accuracy(group):
    correct = (group['pred_after'] == group['GT']).sum()
    total = len(group)
    accuracy = correct / total if total > 0 else 0
    return accuracy

accuracy_by_condition = (
    csv_data.groupby('context_condition')
            .apply(calculate_accuracy)
            .rename('accuracy')
)

logger = create_logger(f'./experiments/{prompt}/{algorithm}/lora-Meta-Llama-3-8B-Instruct-system{system}/{benchamrk}')
logger.info('*****************************')
logger.info(accuracy_by_condition)