import os
import json

# def merge_json_files(root_folder):
#     for subdir, _, files in os.walk(root_folder):
#         json_data = []
        
#         for file in files:
#             if file.endswith(".json"):
#                 file_path = os.path.join(subdir, file)
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                     json_data.extend(data)
        
#         merged_file_path = os.path.join("./experiments/probabilityv2/system1/commonsensqa", "merged.json")
#         with open(merged_file_path, "w", encoding="utf-8") as f:
#             json.dump(json_data, f, indent=4)
#         print(f"Merged JSON saved at: {merged_file_path}")


# root_directory = "./experiments/prob_benchmarkv2/dpo/lora-Meta-Llama-3-8B-Instruct-system1/commonsensqa" 
# merge_json_files(root_directory)


with open('./experiments/probabilityv2/system1/commonsensqa/merged.json', 'r') as f:
    data1 = json.load(f)
    
with open('./experiments/probabilityv2/system2/commonsensqa/merged.json', 'r') as f:
    data2 = json.load(f)
    
print(len(data1))
print(len(data2))