import json
import pandas as pd
import torch
import esm
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
import os
AMINO_ACIDS = 'CDSQKIPTFAGHELRWVNYM'

def prepare_dms(nn_config):
    if os.path.exists(nn_config['dms_path']):
        return pd.read_csv(nn_config['dms_path'])
    
    sequence = nn_config['sequence']
    protein_name = nn_config['protein_name']
    mutants = []
    mutant_seqs = []
    for i in range(len(sequence)):  
        original_aa = sequence[i] 
        for aa in AMINO_ACIDS: 
            if aa != original_aa:
                mutant = original_aa+str(i+1)+aa
                mutant_sequence = sequence[:i] + aa + sequence[i+1:]
                mutants.append(mutant)  # mutant (start from 0)
                mutant_seqs.append(mutant_sequence)  # mutant sequence 
    data = {
        "protein_name":[protein_name for _ in mutants],
        "smiles":nn_config['smiles'],
        "mutant":mutants,
        "mutant_sequence":mutant_seqs,
        "seq_id":[protein_name+"_"+mut for mut in mutants],}
    dms_df = pd.DataFrame(data)
    dms_df.to_csv(nn_config['dms_path'],index=False)
    return dms_df

def prepare_esm_embedding(dms_df,nn_config):
    sequence = nn_config['sequence']
    protein_name = nn_config['protein_name']
    save_path = nn_config['ESM_path']
    Max_esm = np.load("./models/esm_t33_Max_esm.npy")
    Min_esm = np.load("./models/esm_t33_Min_esm.npy")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model = model.to(device)
    
    
    # Prepare data
    
    data = [(row['seq_id'],row['mutant_sequence']) for _,row in dms_df.iterrows()]
    data.append((protein_name,sequence))   # wildtype
    
    # 保证一口气生成
    if len([(seq_id,seq) for seq_id,seq in data if not os.path.isfile(save_path + seq_id + '.tensor')])==0:
        # 没有任何一条esm emb缺失
        return
        
    with torch.no_grad():
        batch_size = 10
        for batch_idx,i in enumerate(range(0, len(data), batch_size)):
            batch_data = data[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            print(f"ESM batch idx:{batch_idx}")

            batch_tokens=batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            # Generate and save per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, ((seq_id,_),tokens_len) in enumerate(zip(batch_data,batch_lens)):
                raw_esm = token_representations[i, 1 : tokens_len - 1].detach().cpu().numpy()
                esm_norm = (raw_esm - Min_esm) / (Max_esm - Min_esm)
                torch.save(torch.tensor(esm_norm, dtype = torch.float32), save_path + seq_id + '.tensor')



def pdb2tensor(nn_config):
    def get_pdb_xyz(pdb_file):
        current_pos = -1000
        X = []
        current_aa = {} # N, CA, C, O, R
        for line in pdb_file:
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    R_group = []
                    for atom in current_aa:
                        if atom not in ["N", "CA", "C", "O"]:
                            R_group.append(current_aa[atom])
                    if R_group == []:
                        R_group = [current_aa["CA"]]
                    R_group = np.array(R_group).mean(0)
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom != "H":
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
        return np.array(X)
    
    origin_pdb_path = nn_config['pdb_path']
    with open(origin_pdb_path, "r") as f:
        X = get_pdb_xyz(f.readlines())
        
    save_path = nn_config['structs_path']
    protein_name = nn_config['protein_name']
    torch.save(torch.tensor(X, dtype = torch.float32), save_path + protein_name + '.tensor')



def get_DSSP(nn_config):
    ########## Get DSSP ##########
    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(8)
            SS_vec[SS_type.find(SS)] = 1
            ACC = float(lines[i][34:38].strip())
            ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
            dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

        return seq, dssp_feature

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        padded_item = np.zeros(9)

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    ref_seq = nn_config['sequence']
    dssp_path = nn_config['dssp_path']
    pdb_path = nn_config['pdb_path']
    ID = nn_config['protein_name']
    save_path = nn_config['structs_path']
    
    os.system("{}mkdssp -i {} -o {}{}.dssp".format(dssp_path, pdb_path, save_path,ID))
    dssp_seq, dssp_matrix = process_dssp("{}{}.dssp".format(save_path,ID))
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
    dssp_matrix = np.array(dssp_matrix)

    torch.save(torch.tensor(dssp_matrix, dtype = torch.float32), save_path + ID + '_dssp.tensor')
    os.system("rm {}{}.dssp".format(save_path,ID))



def prepare_feature(nn_config):
    
    # 后续补充抛出异常，如缺失、输入格式异常等
    with open(nn_config['input'], 'r', encoding='utf-8') as file:
        data = json.load(file)
    nn_config['job_name'] = data['job_name']
    nn_config['protein_name'] = data['protein']['name']
    nn_config['sequence'] = data['protein']['sequence']
    nn_config['smiles'] = data['ligand']['SMILES']
    nn_config['pdb_path'] = data['pdb_path']
    
    # step 0 makedirs
    feature_path = nn_config['feature_path']
    ESM_path = feature_path+"esm/"+nn_config['job_name']+"/"
    os.makedirs(ESM_path,exist_ok=True)
    nn_config['ESM_path'] = ESM_path
    
    structs_path = feature_path+"structs/"
    os.makedirs(structs_path,exist_ok=True)
    nn_config['structs_path'] = structs_path
    
    
    # step 1 prepare dms file
    print("step 1 prepare dms file")
    nn_config['dms_path'] = feature_path+nn_config['job_name']+".csv"
    dms_df = prepare_dms(nn_config)

    # step2 get ESM embeddings
    print("step2 prepare ESM embeddings")
    prepare_esm_embedding(dms_df,nn_config)
    
    # step3 get DSSP
    print("step3 prepare DSSP")
    pdb2tensor(nn_config)
    get_DSSP(nn_config)
    
    return nn_config