import argparse
import json
import torch
from BioMiner.MolScribe.molscribe import MolScribe
from tqdm import tqdm
import os 
import warnings 
from collections import defaultdict
warnings.filterwarnings('ignore')
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='BioMiner/MolScribe/ckpts/swin_base_char_aux_1m680k.pth')
    parser.add_argument('--image_dir', type=str, default='BioVista/ocsr/data')
    parser.add_argument('--return_confidence', action='store_true')
    parser.add_argument('--return_atoms_bonds', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:2')
    model = MolScribe(args.model_path, device)
    ocsr_imgs = os.listdir(args.image_dir)
    
    res = {'pdf_name':[],
           'index':[], 
           'smiles':[]}
    
    name2imgpaths = defaultdict(list)
    for img in ocsr_imgs:
        img_items = img.split('.')[0].split('_')
        pdb_idx, pdb_name, page, global_index = img_items[0], img_items[1], img_items[-1].split('-')[0], img_items[-1].split('-')[1]
        name = f'{pdb_idx}_{pdb_name}'
        name2imgpaths[name].append(os.path.join(args.image_dir, img))

    for name in tqdm(name2imgpaths.keys()):
        image_paths = name2imgpaths[name]
        # print(image_paths)
        for image_path in image_paths:
            file_name = os.path.basename(image_path)

            img_items = file_name.split('.')[0].split('_')
            pdb_idx, pdb_name, page, global_index = img_items[0], img_items[1], img_items[-1].split('-')[0], img_items[-1].split('-')[1]
            name = f'{pdb_idx}_{pdb_name}'
            res['pdf_name'].append(name)
            res['index'].append(int(global_index))

        output = model.predict_image_files(
            image_paths, return_atoms_bonds=args.return_atoms_bonds, return_confidence=args.return_confidence)
        
        for pred_res_item, image_path in zip(output, image_paths):
            smiles = pred_res_item['smiles']
            # print(f'{image_path}:{smiles}')
            res['smiles'].append(smiles)

df = pd.DataFrame(res).sort_values(by=['pdf_name', 'index'])
df.to_csv('BioVista/component_result/molscribe_ocsr_smiles.csv')