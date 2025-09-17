from prepare import Inference_prepare_feature
import json
import argparse
import datetime
from tqdm import tqdm
from model import MAGE
from torch_geometric.loader import DataLoader
from data import ProteinGraphDataset

import numpy as np
import os, random, torch
import pandas as pd


def Inference(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = config['output']
    os.makedirs(output_path, exist_ok = True)
    
    artifacts_path = config['artifacts_path']

    num_workers = 8
    dropout=0.0
    node_input_dim = 1473
    edge_input_dim = 450
    hidden_dim = 128
    GNN_layers = 2
    batch_size = 12
    folds = 5
    kcat_num_att_layers= 2
    km_num_att_layers= 4


    test_dataset = ProteinGraphDataset(nn_config['dms_path'],nn_config['feature_path'])
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    kcat_random_models = []
    km_random_models = []
    kcat_seq_models = []
    km_seq_models = []
    for fold in range(folds):
        state_dict = torch.load(artifacts_path + 'kcat_random/finetune_fold%s.ckpt'%fold, device)
        model = MAGE(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "kcat", kcat_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        kcat_random_models.append(model)

        state_dict = torch.load(artifacts_path + 'km_random/finetune_fold%s.ckpt'%fold, device)
        model = MAGE(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "km", km_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        km_random_models.append(model)

        state_dict = torch.load(artifacts_path + 'kcat_seq/finetune_fold%s.ckpt'%fold, device)
        model = MAGE(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "kcat", kcat_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        kcat_seq_models.append(model)

        state_dict = torch.load(artifacts_path + 'km_seq/finetune_fold%s.ckpt'%fold, device)
        model = MAGE(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "km", km_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        km_seq_models.append(model)


    test_random_pred_dict = {} 
    test_seq_pred_dict = {} 
    for data in tqdm(test_dataloader):
        wt_data,mut_data,smiles = data[0].to(device),data[1].to(device),data[2].to(device)
        with torch.no_grad():
            mut_data.mut_graph_mask=mut_data.mut_graph_mask10
            kcat_random_outputs = [model(wt_data,mut_data,smiles) for model in kcat_random_models] # [[b,2]*5]
            km_random_outputs = [model(wt_data,mut_data,smiles) for model in km_random_models] # [[b,2]*5]
            km_seq_outputs = [model(wt_data,mut_data,smiles) for model in km_seq_models] # [[b,2]*5]        
            
            mut_data.mut_graph_mask=mut_data.mut_graph_mask12
            kcat_seq_outputs = [model(wt_data,mut_data,smiles) for model in kcat_seq_models] # [[b,2]*5]
            
            
            kcat_random_preds = [outputs[0] for outputs in kcat_random_outputs] # [[b,2]*5]
            kcat_random_preds = torch.stack(kcat_random_preds,0).mean(0).detach().cpu().numpy() 
            kcat_seq_preds = [outputs[0] for outputs in kcat_seq_outputs] # [[b,2]*5]
            kcat_seq_preds = torch.stack(kcat_seq_preds,0).mean(0).detach().cpu().numpy() 
            km_random_preds = [outputs[0] for outputs in km_random_outputs] # [[b,2]*5]
            km_random_preds = torch.stack(km_random_preds,0).mean(0).detach().cpu().numpy() 
            km_seq_preds = [outputs[0] for outputs in km_seq_outputs] # [[b,2]*5]
            km_seq_preds = torch.stack(km_seq_preds,0).mean(0).detach().cpu().numpy() 
            
                
        names = wt_data.name 
        for i, name in enumerate(names):
            test_random_pred_dict[name] = [kcat_random_preds[i],km_random_preds[i],kcat_random_preds[i]-km_random_preds[i]]
            test_seq_pred_dict[name] = [kcat_seq_preds[i],km_seq_preds[i],kcat_seq_preds[i]-km_seq_preds[i]]
        
    dms_df = pd.read_csv(config['dms_path'])

    kcat_random_preds = []
    km_random_preds = []
    kcatOverkm_random_preds = []
    kcat_seq_preds = []
    km_seq_preds = []
    kcatOverkm_seq_preds = []

    kcatOverkm_aveLevel_preds = []

    for _,row in dms_df.iterrows():
        index = row['index']
        kcat_random_preds.append(test_random_pred_dict[index][0])
        km_random_preds.append(test_random_pred_dict[index][1])
        kcatOverkm_random_preds.append(test_random_pred_dict[index][2])

        kcat_seq_preds.append(test_seq_pred_dict[index][0])
        km_seq_preds.append(test_seq_pred_dict[index][1])
        kcatOverkm_seq_preds.append(test_seq_pred_dict[index][2])

        kcatOverkm_aveLevel_preds.append((test_random_pred_dict[index][2]+test_seq_pred_dict[index][2])/2)
    
    dms_df['kcat_random_preds']=kcat_random_preds
    dms_df['km_random_preds']=km_random_preds
    dms_df['kcat_seq_preds']=kcat_seq_preds
    dms_df['km_seq_preds']=km_seq_preds
    dms_df['kcatOverkm_aveLevel_preds']=kcatOverkm_aveLevel_preds

    dms_df.to_csv(config['output']+config['job_name']+".csv",index=False)


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='./input_json/example.json')
parser.add_argument("--feature_path", type=str, default='./features/',help="Save prepared features path")
parser.add_argument("--output", type=str, default='./output/',help="Results path")
args = parser.parse_args()


with open(args.input, 'r', encoding='utf-8') as file:
    nn_config = json.load(file)
    
nn_config['input']=args.input
nn_config['job_name'] = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
nn_config['feature_path']=args.feature_path + nn_config['job_name'] + "/"
os.makedirs(nn_config['feature_path'],exist_ok=True)
nn_config['output']=args.output


if __name__ == '__main__':
    Seed_everything(seed=nn_config['seed'])
    Inference_prepare_feature(nn_config)
    Inference(nn_config)
