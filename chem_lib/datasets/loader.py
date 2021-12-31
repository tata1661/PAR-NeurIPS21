import os
import torch
import json
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    allowable_features = {
        'possible_atomic_num_list' : list(range(1, 119)),
        'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'possible_chirality_list' : [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list' : [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'possible_bonds' : [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'possible_bond_dirs' : [ # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    }
except:
    print('Error rdkit:')
    Chem, AllChem, allowable_features=None,None, None

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir,'smiles.csv'),
            header=None).to_numpy()[:,0]



    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            #smi = self.smiles_list[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        data['smiles']=self.smiles_list[idx]
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)  # ['tox21.json'] #
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #Chem.SanitizeMol(rdkit_mol,
                                 #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                #print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    #    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # # convert nan to 0
    # labels = labels.fillna(0)
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels

def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
    #    'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
    #    'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # # convert nan to 0
    # labels = labels.fillna(0)
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 
    
    return smiles_list, rdkit_mol_objs_list, labels

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 
    # print(smiles_list)
    # print(labels)
    # raise TypeError

    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # tasks = ['Hepatobiliary disorders',
    #    'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
    #    'Investigations', 'Musculoskeletal and connective tissue disorders',
    #    'Gastrointestinal disorders', 'Social circumstances',
    #    'Immune system disorders', 'Reproductive system and breast disorders',
    #    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
    #    'General disorders and administration site conditions',
    #    'Endocrine disorders', 'Surgical and medical procedures',
    #    'Vascular disorders', 'Blood and lymphatic system disorders',
    #    'Skin and subcutaneous tissue disorders',
    #    'Congenital, familial and genetic disorders',
    #    'Infections and infestations',
    #    'Respiratory, thoracic and mediastinal disorders',
    #    'Psychiatric disorders', 'Renal and urinary disorders',
    #    'Pregnancy, puerperium and perinatal conditions',
    #    'Ear and labyrinth disorders', 'Cardiac disorders',
    #    'Nervous system disorders',
    #    'Injury, poisoning and procedural complications']
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels
    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # # Some smiles could not be successfully converted
    # # to rdkit mol object so them to None
    # preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
    #                                     rdkit_mol_objs_list]
    # preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
    #                             None for m in preprocessed_rdkit_mol_objs_list]
    # tasks = list(input_df.columns)[1:]
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # # convert nan to 0
    # labels = labels.fillna(0)
    # assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    # assert len(smiles_list) == len(preprocessed_smiles_list)
    # assert len(smiles_list) == len(labels)
    # return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
    #        labels.values
