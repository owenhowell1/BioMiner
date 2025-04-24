import os 
source_dir = '/data/jiaxianyan/1-git/Scientific-Data-Extraction-main/SDE/SDEVista/DrugDesign/labels/jintao_latest_500_ejmc'
target_dir = 'example/full_md_molminer'

with open('/data/jiaxianyan/1-git/Scientific-Data-Extraction-main/SDE/SDEVista/DrugDesign/data/latest_500/file_names.txt', 'r') as f:
    names = f.read().strip().split('\n')

for name in names:
    source_file = os.path.join(source_dir, f'{name}.pdf.json')
    target_file = os.path.join(target_dir, f'{name}.json')
    if os.path.exists(source_file):
        os.system(f'cp {source_file} {target_file}')