import os
import json
import argparse
from tqdm import tqdm
from BioMiner.commons.ocsr import draw_bbox_xywh
from BioMiner.commons.process_pdf import image_segment_given_box_xywh, pdf_load_pypdf_images, load_pdf_pages_contain_tables_and_figures
from BioMiner.commons.utils import pmap_multi
from BioMiner.commons.mineru_pdf import run_mineru, get_mineru_table_body_bbox, get_mineru_complete_figure_table_bbox
from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict
from typing import Optional
from functools import reduce
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm

fix_structures = {
    '[N;!+:1](=[O:2])[O;!-:3]': '[N+:1](=[O:2])[O-:3]',
    '[C:1][N:2]=[N;!+:3]=[N;!-:4]': '[C:1][N:2]=[N+:3]=[N-:4]',
    '[C:1]=[O:2][C:3]': '[C:1][O:2][C:3]',
    '[C:1]=[C:2]([O:3])=[O:4]': '[C:1][C:2]([O:3])=[O:4]',
    '[C:1]=[C:2]1[N:3][C:4]=[C:5][N:6]=1': '[C:1][C:2]1[N:3][C:4]=[C:5][N:6]=1',
    '[*:1][N;!+:2]([*:3])=[*:4]': '[*:1][N+:2]([*:3])=[*:4]',
    '[*:1][N;!+:2]([*:3])([*:4])[*:5]': '[*:1][N+:2]([*:3])([*:4])[*:5]',
}


def check_and_fix_bug_structures(
        bug_smiles: Optional[str] = None,
        bug_mol=None,
):
    if bug_mol is None:
        fragments = [Chem.MolFromSmiles(_bug_smiles, sanitize=False) for _bug_smiles in bug_smiles.split('.')]
    else:
        fragments = [bug_mol]
    combs = []
    for bug_mol in fragments:
        for k, v in fix_structures.items():
            for inds in bug_mol.GetSubstructMatches(Chem.MolFromSmarts(k)):
                # substructure = Chem.MolFromSmarts(k)#Chem.MolFromSmiles(k, sanitize = False)
                # replacement = Chem.MolFromSmiles(v)

                # new_smi = Chem.MolToSmiles(
                #     Chem.ReplaceSubstructs(bug_mol, substructure, replacement, replaceAll=True, replacementConnectionPoint = 0, useChirality=True)[0]
                # )
                rxn = rdChemReactions.ReactionFromSmarts(f'{k}>>{v}')
                products = rxn.RunReactants((bug_mol,))
                if len(products) == 0:
                    continue
                new_smi = Chem.MolToSmiles(products[0][0])
                bug_mol = Chem.MolFromSmiles(new_smi, sanitize=False)

        combs.append(bug_mol)

    combo = reduce(Chem.CombineMols, combs)
    return Chem.MolToSmiles(combo)

def read_molminer_sdf(sdf: str) -> str:
    lines = sdf.splitlines()

    first_n_lines_pass = 3
    graph_attrs_line = 4
    atom_num_position = (0, 3)
    bond_num_position = (3, 6)
    graph_attrs = lines[graph_attrs_line - 1]  # .split()
    n_atoms = int(graph_attrs[slice(*atom_num_position)].strip())
    n_bonds = int(graph_attrs[slice(*bond_num_position)].strip())
    mol_attrs_1st_line = graph_attrs_line + n_atoms + n_bonds + 1

    rgroups = {}
    for l1, l2 in zip(lines[(mol_attrs_1st_line - 1)::2], lines[(mol_attrs_1st_line)::2]):
        if l1.startswith('M  '): break
        _, atom_index = l1.split()
        atom_index = int(atom_index)
        label = l2.strip()
        rgroups[atom_index] = label

    n_rgroups = len(rgroups)
    new_atom_lines = []
    for lid, l in enumerate(lines[graph_attrs_line:graph_attrs_line + n_atoms], start=1):
        # x_coords = l[:10]
        # y_coords = l[10:20]
        # z_coords = l[20:30]
        symbol = l[30:32].strip()
        all_zeros = []
        for i in range(33, 70, 3):
            all_zeros.append(l[i: i + 3])
        if (symbol == '') or (lid in rgroups):
            # label = rgroups[lid]
            all_zeros[5] = '  1'
            label = '*'
            symbol = f'{label:>2} '
            new_atom_lines.append(l[:30] + symbol + ''.join(all_zeros))
        else:
            new_atom_lines.append(l)

    if n_rgroups > 1:
        ISO_str = f'M  ISO {n_rgroups:>2}'
        rgroups = dict(sorted(rgroups.items(), key=lambda item: item[1]))
        for sindex, (lid, label) in enumerate(rgroups.items(), start=1):
            ISO_str += f' {lid:>3} {sindex:>3}'

        new_lines = lines[:graph_attrs_line] + new_atom_lines + lines[
                    graph_attrs_line + n_atoms:mol_attrs_1st_line - 1] + [
                    ISO_str] + lines[mol_attrs_1st_line - 1 + n_rgroups * 2:]
    else:
        new_lines = lines[:graph_attrs_line] + new_atom_lines + lines[
                    graph_attrs_line + n_atoms:mol_attrs_1st_line - 1] + lines[
                    mol_attrs_1st_line - 1 + n_rgroups * 2:]

    new_sdf = '\n'.join(new_lines)
    # print(new_sdf)
    return new_sdf

def convert_sdf_to_smiles(name, sdf_file):
    with open(sdf_file, 'r') as f:
        sdfs = f.read().split('$$$$')[:-1]

    res = {'pdf_name':[],
           'index':[], 
           'molminer_id':[],
           'smiles':[]}
    
    for index, sdf in enumerate(sdfs):
        temp_sdf = sdf.strip() + '\n$$$$\n'
        temp_sdf = temp_sdf.replace('G', 'X')
        lines = temp_sdf.split('\n')
        molminer_id = lines[0]
        
        res['pdf_name'].append(name)
        res['index'].append(index)
        res['molminer_id'].append(molminer_id)
        # print(sdf_file)
        # print(temp_sdf)
        if sdf is None:
            smiles = None
            res['smiles'].append(smiles)
            continue
        
        new_sdf = read_molminer_sdf(temp_sdf)
        try:
            mol = Chem.MolFromMolBlock(new_sdf)
            if mol is None:
                mol = Chem.MolFromMolBlock(new_sdf, sanitize=False)
                smiles = check_and_fix_bug_structures(bug_mol=mol)
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            # print(f'ID (With Rgroup): {molminer_id}', smiles)
        except Exception as e:
            print(f'Error: {molminer_id}')
            # print(e)
            # print('original:\n', sdf)
            # print('convert:\n', new_sdf)
            rdkit_valid = False
            smiles = None

        res['smiles'].append(smiles)

    df = pd.DataFrame(res)
    return df

def determine_mol_type(smi):
    if smi is None or '.' in smi or smi =='C':
        return 'invalid'
    
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return 'invalid'
    if mol is None:
        return 'invalid'

    if '*' in smi:
        return 'part'

    return 'full'

def draw_augmented_images_with_molminer(bbox_path, sdf_smiles_path, parsed_path, file_name, full_path, markush_path):

    page2bbox = {'full':defaultdict(list),
                 'part':defaultdict(list),
                 'invalid':defaultdict(list)}
    
    image2bboxindex = defaultdict(dict)

    df_gd = convert_sdf_to_smiles(file_name, sdf_smiles_path)

    with open(bbox_path, 'r') as file:
        bboxes = json.load(file)

    for bbox, smi in zip(bboxes, df_gd['smiles']):
        index = bbox['index']
        bbox_line = bbox['bbox']
        page = bbox['page'] + 1
        
        mol_type = determine_mol_type(smi)

        page2bbox[mol_type][page].append({'index': index, 
                                          'bbox': bbox_line, 
                                          'smiles': smi
                                        })
        
    for page in page2bbox['full'].keys():
        page_bboxs = page2bbox['full'][page]
        page_image_path = os.path.join(parsed_path, f'{file_name}_image_{page-1}.png')
        augmented_full_image_file_name = f'{file_name}_image_molminer_modify_bbox_molminer_full_{page}.png'

        if augmented_full_image_file_name not in full_augmented_image_names:
            continue

        augmented_full_image_path = os.path.join(full_path, augmented_full_image_file_name)
        image = Image.open(page_image_path)

        for bbox in page_bboxs:
            index = bbox['index']
            smi = bbox['smiles']
            x, y, w, h = bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]
            draw_bbox_xywh(image, x, y, w, h, index)       
            image2bboxindex[augmented_full_image_file_name][index] = (x,y,w,h,page_image_path)

        image.save(augmented_full_image_path)


    for page in page2bbox['part'].keys():
        page_bboxs = page2bbox['part'][page]
        page_image_path = os.path.join(parsed_path, f'{file_name}_image_{page-1}.png')
        augmented_part_image_file_name = f'{file_name}_image_molminer_modify_bbox_molminer_part_{page}.png'

        if augmented_part_image_file_name in markush_augmented_image_names['text']:
            augmented_part_image_path = os.path.join(markush_path, 'text', augmented_part_image_file_name)
        elif augmented_part_image_file_name in markush_augmented_image_names['image']:
            augmented_part_image_path = os.path.join(markush_path, 'image', augmented_part_image_file_name)
        elif augmented_part_image_file_name in markush_augmented_image_names['hybrid']:
            augmented_part_image_path = os.path.join(markush_path, 'hybrid', augmented_part_image_file_name)
        else:
            continue
        
        image = Image.open(page_image_path)

        for bbox in page_bboxs:
            index = bbox['index']
            smi = bbox['smiles']
            x, y, w, h = bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]
            draw_bbox_xywh(image, x, y, w, h, index)
            image2bboxindex[augmented_part_image_file_name][index] = (x,y,w,h,page_image_path)

        image.save(augmented_part_image_path)

    return image2bboxindex

def process_mineru_single(mineru_layout):
    complete_figure_table_bbox = get_mineru_complete_figure_table_bbox(mineru_layout)
    return complete_figure_table_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', type=str, default='example/pdfs')
    parser.add_argument('--file_name', type=str, default='BioVista/full_coreference/prepare/file_names.txt')
    parser.add_argument('--augmented_image_prepare_dir', type=str, default='BioVista/full_coreference/prepare')
    parser.add_argument('--temp_page_image_dir', type=str, default='BioVista/temp_page_images')
    parser.add_argument('--full_augmented_image_names', type=str, default='BioVista/full_coreference/augmented_image_names.json')
    parser.add_argument('--markush_augmented_image_names', type=str, default='BioVista/markush_enumeration/augmented_image_names.json')
    parser.add_argument('--full_save_dir', type=str, default='BioVista/full_coreference/data')
    parser.add_argument('--markush_save_dir', type=str, default='BioVista/markush_enumeration/data')
    parser.add_argument('--mineru_path', type=str, default='BioMiner/example_log/mineru')

    args = parser.parse_args()

    with open(args.file_name, 'r') as f:
        names = f.read().strip().split('\n')

    with open(args.full_augmented_image_names, 'r') as f:
        full_augmented_image_names = json.load(f)

    with open(args.markush_augmented_image_names, 'r') as f:
        markush_augmented_image_names = json.load(f)

    # convert pdf pages to images
    os.makedirs(args.temp_page_image_dir, exist_ok=True)
    os.makedirs(args.full_save_dir, exist_ok=True)
    os.makedirs(args.markush_save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.markush_save_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(args.markush_save_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.markush_save_dir, 'hybrid'), exist_ok=True)

    pdf_paths = []
    for name in names:
        pdf_paths.append(os.path.join(args.pdf_dir, f'{name}.pdf'))

    page_image_pathss = pmap_multi(pdf_load_pypdf_images, 
                                    zip(names, pdf_paths),
                                    save_path=args.temp_page_image_dir,
                                    n_jobs=32,
                                    desc='converting pdf pages to images')

    # MinerU layout analysis and reading order determination
    mineru_layouts = []
    for name, pdf_path in tqdm(zip(names, pdf_paths), desc='run mineru'):
        mineru_layout, mineru_text = run_mineru(name, pdf_path, args.mineru_path, 'cuda:0')
        mineru_layouts.append(mineru_layout)

    # process mineru layout
    res = pmap_multi(process_mineru_single,
                     zip(mineru_layouts),
                     n_jobs=32,
                     desc='process mineru layout')
    
    total_middle_json = {}
    for (name, minuer_figure_table_bbox_dict) in zip(names, res):
        total_middle_json[name] = minuer_figure_table_bbox_dict

    new_json_data = json.dumps(total_middle_json, indent=4)
    with open('BioVista/full_coreference/figure_table_layout.json', 'w') as f:
        f.write(new_json_data)
    with open('BioVista/markush_enumeration/figure_table_layout.json', 'w') as f:
        f.write(new_json_data)

    bbox_paths, sdf_smiles_paths, parsed_paths, file_names = [], [], [], []
    for name in tqdm(names):
        bbox_path = os.path.join(args.augmented_image_prepare_dir, f'{name}.pdf.json')
        sdf_smiles_path = os.path.join(args.augmented_image_prepare_dir, f'{name}.sdf')
        parsed_path = os.path.join(args.temp_page_image_dir, name)

        if not os.path.exists(bbox_path):
            continue

        bbox_paths.append(bbox_path)
        sdf_smiles_paths.append(sdf_smiles_path)
        parsed_paths.append(parsed_path)
        file_names.append(name)
    
    results = pmap_multi(draw_augmented_images_with_molminer, 
                         zip(bbox_paths, sdf_smiles_paths, parsed_paths, file_names),
                         full_path=args.full_save_dir, 
                         markush_path=args.markush_save_dir,
                         n_jobs=32,
                         desc='draw augmented images ... ')
    
    total_image2bboxindex = {}
    for image2bboxindex in results:
        for key in image2bboxindex.keys():
            total_image2bboxindex[key] = image2bboxindex[key]
    
    json_data = json.dumps(total_image2bboxindex, indent=4)
    with open('BioVista/full_coreference/image2bboxindex.json', 'w') as f:
        f.write(json_data)
    with open('BioVista/markush_enumeration/image2bboxindex.json', 'w') as f:
        f.write(json_data)