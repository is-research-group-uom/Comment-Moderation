from API import get_data
from llm.llama import llama
from llm.claude import claude3_5
from llm.deepseek import deepseek
from evaluation import eval_dictionary
import re

data = get_data()
dict_eval = []
for i in data['rows']:
    response = llama(i['row']['prompt'])
    print("Response: ", response)

    # Match all label groups following -> or Label:, with or without parentheses, including "and"/"," separation
    label_section = re.findall(r'->(?:\s*Label:)?\s*([^.\n]+)', response)

    # Flatten and extract individual labels (S, H, V, HR, SH)
    labels = []
    for section in label_section:
        matches = re.findall(r'\b(SH|HR|S|H|V|H2|S3|V2)\b', section)
        labels.extend(matches)

    # Deduplicate
    unique_labels = list(set(labels))

    print("Detected labels:", unique_labels)
    # print(i['row'])
    h_labels = [label for label in ['S', 'H', 'V', 'HR', 'SH', 'H2', 'S3', 'V2'] if i['row'][label] == 1]
    # print("Labels: ", h_labels)
    print("Correct labels:", h_labels)

    dict = {
        "Model Logic: ": response,
        "Ai Labels: ": unique_labels,
        "Human Labels: ": h_labels
    }

    dict_eval.append(dict)

eval_dictionary(dict_eval)
