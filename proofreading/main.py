#校对系统入口，运行该文件，将读取测试集，输出预测结果到结果文件
import argparse
import os
import pickle
import json
import xml.etree.ElementTree as ET
from lxml import etree

def create_dataset(args, input_file):
    input_file = os.path.join(args.data_dir, input_file)
    dataset = pickle.load(open(input_file, 'rb'))
    return dataset

# def create_dataset(args, input_file):
#     input_file = os.path.join(args.data_dir, input_file)
#     file_extension = os.path.splitext(input_file)[1].lower()

#     if file_extension == '.json':
#         with open(input_file, 'r', encoding='utf-8') as f:
#             dataset = json.load(f)
#     elif file_extension == '.sgml':
#         dataset = []
#         try:
#             tree = etree.parse(input_file)
#             root = tree.getroot()
#             for sample in root.findall('sample'):
#                 dataset.append({
#                     'input_ids': [int(token) for token in sample.find('input_ids').text.split()],
#                     'attention_mask': [int(mask) for mask in sample.find('attention_mask').text.split()],
#                     'label_ids': [int(label) for label in sample.find('label_ids').text.split()]
#                 })
#         except etree.XMLSyntaxError as e:
#             print(f"XML parsing error: {e}")
#     else:
#         raise ValueError(f"Unsupported file format: {file_extension}")

#     return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="pycode/project/data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    args = parser.parse_args()
    
    dataset = create_dataset(args, 'train_data/test.sighan15.pkl')
    # dataset = create_dataset(args, 'train_data/LabeledData.json')
    print(dataset)




