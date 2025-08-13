from extract_embedding import extract_embedding,Seed_everything
from prepare import prepare_feature
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='./input/example.json')
parser.add_argument("--feature_path", type=str, default='./data/',help="Prepare input features path")
parser.add_argument("--output", type=str, default='./output/',help="Results path")
args = parser.parse_args()


with open('./models/setting.json', 'r', encoding='utf-8') as file:
    nn_config = json.load(file)

nn_config["input"]=args.input
nn_config['feature_path']=args.feature_path
nn_config['output']=args.output

Seed_everything(seed=nn_config['seed'])

if __name__ == '__main__':
    prepare_feature(nn_config)
    extract_embedding(nn_config)
