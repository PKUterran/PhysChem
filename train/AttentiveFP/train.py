import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

import time
import numpy as np
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import QED
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)


task_name = 'SARS-COV-2'
tasks = [
   '3CL_enzymatic_activity', 'ACE2_enzymatic_activity',
    'HEK293_cell_line_toxicity_', 'Human_fibroblast_toxicity',
    'MERS_Pseudotyped_particle_entry',
    'MERS_Pseudotyped_particle_entry_(Huh7_tox_counterscreen)',
    'SARS-CoV_Pseudotyped_particle_entry',
    'SARS-CoV_Pseudotyped_particle_entry_(VeroE6_tox_counterscreen)',
    'SARS-CoV-2_cytopathic_effect_(CPE)',
    'SARS-CoV-2_cytopathic_effect_(host_tox_counterscreen)',
    'Spike-ACE2_protein-protein_interaction_(AlphaLISA)',
    'Spike-ACE2_protein-protein_interaction_(TruHit_Counterscreen)',
    'TMPRSS2_enzymatic_activity'
]
raw_filename = "./data/SARS-COV-2.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.SMILES.values
print("number of all smiles: ",len(smilesList))
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print("not successfully processed smiles: ", smiles)
        smilesList = np.delete(smilesList, np.where(smilesList == smiles))
        pass
print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["SMILES"].isin(remained_smiles)]
# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] = canonical_smiles_list

plt.figure(figsize=(5, 3))
sns.set(font_scale=1.5)
ax = sns.distplot(atom_num_dist, bins=28, kde=False)
plt.tight_layout()
# plt.savefig("atom_num_dist_"+prefix_filename+".png",dpi=200)
plt.show()
plt.close()

# print(len([i for i in atom_num_dist if i<51]),len([i for i in atom_num_dist if i>50]))

random_seed = 880
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()

batch_size = 100
epochs = 100
p_dropout = 0.5
fingerprint_dim = 200

radius = 3
T = 3
weight_decay = 3
learning_rate = 3.5
per_task_output_units_num = 4
output_units_num = len(tasks) * per_task_output_units_num

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

weights = []
for i,task in enumerate(tasks):
    df0 = remained_df[remained_df[task] == 0][["SMILES",task]]
    df1 = remained_df[remained_df[task] == 1][["SMILES",task]]
    df2 = remained_df[remained_df[task] == 2][["SMILES",task]]
    df3 = remained_df[remained_df[task] == 3][["SMILES",task]]
    weights.append([(df0.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0])/df0.shape[0],\
                    (df0.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0])/df1.shape[0],\
                    (df0.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0])/df2.shape[0],\
                    (df0.shape[0]+df1.shape[0]+df2.shape[0]+df3.shape[0])/df3.shape[0]])

folds = 5

f_roc = []

for fold in range(folds):
    print("=============================")
    print("Fold {}".format(fold))
    print()
    test_df = remained_df.sample(frac=1/10)
    while [(list(test_df[task].values).count(3) >= 1) for task in tasks].count(0):
        test_df = remained_df.sample(frac=1/10)
    training_data = remained_df.drop(test_df.index)

    valid_df = training_data.sample(frac=1/9)
    while [(list(valid_df[task].values).count(3) >= 1) for task in tasks].count(0):
        valid_df = training_data.sample(frac=1/9)
    train_df = training_data.drop(valid_df.index)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]

    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight),reduction='mean') for weight in weights]
    model = Fingerprint(radius, T, num_atom_features,num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
    model.cuda()
    
    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    label = {}
    for task in tasks:
        y = train_df[task].values
        label[task] = {}
        for i in range(4):
            label[task][i] = np.where(y == i)[0]

    def train(model, dataset, optimizer, loss_function):
        model.train()
        batch_list = []
        for i in range(5):
            for task in tasks:
                n3 = min(25, len(label[task][3]))
                n2 = min((100 - n3) // 3, len(label[task][2]))
                n1 = (100 - n3 - n2) // 2
                n0 = 100 - n1 - n2 - n3
                l0 = np.random.choice(label[task][0], n0)
                l1 = np.random.choice(label[task][1], n1)
                l2 = np.random.choice(label[task][2], n2)
                l3 = np.random.choice(label[task][3], n3)
                batch_list.append(np.concatenate([l0, l1, l2]))
        for counter, train_batch in enumerate(batch_list):
            batch_df = dataset.loc[train_batch, :]
            smiles_list = batch_df.cano_smiles.values

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                         feature_dicts)
            atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                     torch.cuda.LongTensor(x_atom_index),
                                                     torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))

            optimizer.zero_grad()
            loss = 0.0
            for i, task in enumerate(tasks):
                y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                         per_task_output_units_num]
                y_val = batch_df[task].values

                validInds = np.where((y_val == 0) | (y_val == 1) | (y_val == 2) | (y_val == 3))[0]
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.cuda.LongTensor(validInds).squeeze()
                y_pred_adjust = y_pred[validInds]
                if len(y_pred_adjust.shape) == 1:
                    y_pred_adjust = y_pred_adjust.unsqueeze(0)
                loss += loss_function[i](
                    y_pred_adjust,
                    torch.cuda.LongTensor(y_val_adjust))
            
            loss.backward()
            optimizer.step()


    def eval(model, dataset):
        model.eval()
        y_val_list = {}
        y_pred_list = {}
        losses_list = []
        valList = np.arange(0, dataset.shape[0])
        batch_list = []
        for i in range(0, dataset.shape[0], batch_size):
            batch = valList[i:i + batch_size]
            batch_list.append(batch)
        for counter, eval_batch in enumerate(batch_list):
            batch_df = dataset.loc[eval_batch, :]
            smiles_list = batch_df.cano_smiles.values

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                         feature_dicts)
            atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                     torch.cuda.LongTensor(x_atom_index),
                                                     torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))
            atom_pred = atoms_prediction.data[:, :, 1].unsqueeze(2).cpu().numpy()
            for i, task in enumerate(tasks):
                y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                         per_task_output_units_num]
                y_val = batch_df[task].values

                validInds = np.where((y_val == 0) | (y_val == 1) | (y_val == 2) | (y_val == 3))[0]
                
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.cuda.LongTensor(validInds).squeeze()
                y_pred_adjust = y_pred[validInds]
                
                if len(y_pred_adjust.shape) == 1:
                    y_pred_adjust = y_pred_adjust.unsqueeze(0)
                loss = loss_function[i](
                    y_pred_adjust,
                    torch.cuda.LongTensor(y_val_adjust))
                y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()
                
                losses_list.append(loss.cpu().detach().numpy())
                try:
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)
                except:
                    y_val_list[i] = []
                    y_pred_list[i] = []
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)

        eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i], multi_class='ovo') for i in range(len(tasks))]

        eval_loss = np.array(losses_list).mean()

        return eval_roc, eval_loss

    best_param = {}
    best_param["roc_epoch"] = 0
    best_param["loss_epoch"] = 0
    best_param["valid_roc"] = 0
    best_param["valid_loss"] = 9e8

    for epoch in range(epochs):
        train_roc, train_loss = eval(model, train_df)
        valid_roc, valid_loss = eval(model, valid_df)
        train_roc = [float('{:.4f}'.format(i)) for i in train_roc]
        valid_roc = [float('{:.4f}'.format(i)) for i in valid_roc]
        train_roc_mean = np.array(train_roc).mean()
        valid_roc_mean = np.array(valid_roc).mean()


        if valid_roc_mean > best_param["valid_roc"]:
            best_param["roc_epoch"] = epoch
            best_param["valid_roc"] = valid_roc_mean
            if valid_roc_mean > 0.65:
                torch.save(model, 'saved_models/model_' + prefix_filename + '_' + start_time + '_' + str(epoch) + '.pt')
        if valid_loss < best_param["valid_loss"]:
            best_param["loss_epoch"] = epoch
            best_param["valid_loss"] = valid_loss

        print("EPOCH:\t" + str(epoch) + '\n' \
              + "train_roc_mean" + ":" + str(train_roc_mean) + '\n' \
              + "valid_roc_mean" + ":" + str(valid_roc_mean) + '\n')
        if (epoch - best_param["roc_epoch"] > 10) and (epoch - best_param["loss_epoch"] > 20):
            break

        train(model, train_df, optimizer, loss_function)

    best_model = torch.load('saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["roc_epoch"])+'.pt')

    best_model_dict = best_model.state_dict()
    best_model_wts = copy.deepcopy(best_model_dict)

    model.load_state_dict(best_model_wts)
    (best_model.align[0].weight == model.align[0].weight).all()
    test_roc, test_losses = eval(model, test_df)

    print("best epoch:"+str(best_param["roc_epoch"])
          +"\n"+"test_roc:"+str(test_roc)
          +"\n"+"test_roc_mean:",str(np.array(test_roc).mean()))

    f_roc.append(test_roc)

    print("=============================")
    print()

f_roc = np.array(f_roc)
print('Test ROC: {:.4f}+/-{:.4f}'.format(np.mean(f_roc), np.std(f_roc)))
