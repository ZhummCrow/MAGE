import numpy as np
import os, random, pickle, torch
import pandas as pd
from tqdm import tqdm
from model import GPSite
from torch_geometric.loader import DataLoader
from data import *

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')


def extract_embedding(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    output_path = config['output']
    os.makedirs(output_path, exist_ok = True)
    
    model_path = config['model_path']

    num_workers = config['num_workers']
    node_input_dim = config['node_input_dim']
    edge_input_dim = config['edge_input_dim']
    hidden_dim = config['hidden_dim']
    GNN_layers = config['GNN_layers']
    dropout = config['dropout']
    batch_size = config['batch_size']
    folds = config['folds']
    kcat_num_att_layers= config['attention_layers']['kcat']
    km_num_att_layers= config['attention_layers']['km']


    test_dataset = ProteinGraphDataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    kcat_random_models = []
    km_random_models = []
    kcat_seq_models = []
    km_seq_models = []
    for fold in range(folds):
        state_dict = torch.load(model_path + 'kcat_random/finetune_fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "kcat", kcat_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        kcat_random_models.append(model)

        state_dict = torch.load(model_path + 'km_random/finetune_fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "km", km_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        km_random_models.append(model)

        state_dict = torch.load(model_path + 'kcat_seq/finetune_fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "kcat", kcat_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        kcat_seq_models.append(model)

        state_dict = torch.load(model_path + 'km_seq/finetune_fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, "km", km_num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        km_seq_models.append(model)

    kcat_random_emb_dict = {}
    kcat_seq_emb_dict = {}
    km_random_emb_dict = {}
    km_seq_emb_dict = {}
    test_random_pred_dict = {} # 导出测试结果
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
            kcat_random_preds = torch.stack(kcat_random_preds,0).mean(0).detach().cpu().numpy() # 5个模型预测结果求平均
            kcat_seq_preds = [outputs[0] for outputs in kcat_seq_outputs] # [[b,2]*5]
            kcat_seq_preds = torch.stack(kcat_seq_preds,0).mean(0).detach().cpu().numpy() # 5个模型预测结果求平均
            km_random_preds = [outputs[0] for outputs in km_random_outputs] # [[b,2]*5]
            km_random_preds = torch.stack(km_random_preds,0).mean(0).detach().cpu().numpy() # 5个模型预测结果求平均
            km_seq_preds = [outputs[0] for outputs in km_seq_outputs] # [[b,2]*5]
            km_seq_preds = torch.stack(km_seq_preds,0).mean(0).detach().cpu().numpy() # 5个模型预测结果求平均
            
            
            kcat_random_embs = [outputs[1].detach().cpu().numpy() for outputs in kcat_random_outputs] # [[b,128]*5]
            kcat_seq_embs = [outputs[1].detach().cpu().numpy() for outputs in kcat_seq_outputs] # [[b,128]*5]
            km_random_embs = [outputs[1].detach().cpu().numpy() for outputs in km_random_outputs] # [[b,128]*5]
            km_seq_embs = [outputs[1].detach().cpu().numpy() for outputs in km_seq_outputs] # [[b,128]*5]
            
                
        mutants = wt_data.name # name是突变点
        for i, mut in enumerate(mutants):
            kcat_random_emb_dict[mut] = [batch_emb[i] for batch_emb in kcat_random_embs]
            kcat_seq_emb_dict[mut] = [batch_emb[i] for batch_emb in kcat_seq_embs]
            km_random_emb_dict[mut] = [batch_emb[i] for batch_emb in km_random_embs]
            km_seq_emb_dict[mut] = [batch_emb[i] for batch_emb in km_seq_embs]
            
            test_random_pred_dict[mut] = [kcat_random_preds[i],km_random_preds[i],kcat_random_preds[i]-km_random_preds[i]]
            test_seq_pred_dict[mut] = [kcat_seq_preds[i],km_seq_preds[i],kcat_seq_preds[i]-km_seq_preds[i]]
        

    # 是否保存embedding 用于后续的active learning
    pickle.dump(kcat_random_emb_dict,open(config['output']+"embeddings/"+config['job_name']+"_kcat_random.pkl",'wb'))
    pickle.dump(kcat_seq_emb_dict,open(config['output']+"embeddings/"+config['job_name']+"_kcat_seq.pkl",'wb'))
    pickle.dump(km_random_emb_dict,open(config['output']+"embeddings/"+config['job_name']+"_km_random.pkl",'wb'))
    pickle.dump(km_seq_emb_dict,open(config['output']+"embeddings/"+config['job_name']+"_km_seq.pkl",'wb'))
    
    
    dms_df = pd.read_csv(config['dms_path'])

    kcat_random_preds = []
    km_random_preds = []
    kcatOverkm_random_preds = []
    kcat_seq_preds = []
    km_seq_preds = []
    kcatOverkm_seq_preds = []

    kcatOverkm_aveLevel_preds = []

    for _,row in dms_df.iterrows():
        mut = row['mutant']
        kcat_random_preds.append(test_random_pred_dict[mut][0])
        km_random_preds.append(test_random_pred_dict[mut][1])
        kcatOverkm_random_preds.append(test_random_pred_dict[mut][2])

        kcat_seq_preds.append(test_seq_pred_dict[mut][0])
        km_seq_preds.append(test_seq_pred_dict[mut][1])
        kcatOverkm_seq_preds.append(test_seq_pred_dict[mut][2])

        kcatOverkm_aveLevel_preds.append((test_random_pred_dict[mut][2]+test_seq_pred_dict[mut][2])/2)
    
    dms_df['kcat_random_preds']=kcat_random_preds
    dms_df['km_random_preds']=km_random_preds
    dms_df['kcat_seq_preds']=kcat_seq_preds
    dms_df['km_seq_preds']=km_seq_preds
    dms_df['kcatOverkm_aveLevel_preds']=kcatOverkm_aveLevel_preds

    dms_df.to_csv(config['output']+config['job_name']+".csv",index=False)
