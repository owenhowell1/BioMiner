import argparse
from BioMiner import BioMiner_Markush_Infernce
import os 
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_mllm_type', default='gemini-2.0-flash')
    parser.add_argument('--markush_prompt', default='BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_part_updated_with_index.txt')
    parser.add_argument('--markush_data_dir', default='BioVista/markush_enumeration/data')
    parser.add_argument('--save_dir', default='BioVista/component_result/markush_enumerate/gemini-2.0-flash')
    parser.add_argument('--image2bboxindex', default='BioVista/markush_enumeration/image2bboxindex.json')
    parser.add_argument('--layout_seg_json', default='BioVista/markush_enumeration/figure_table_layout.json')
    parser.add_argument('--cot', default=False)
    parser.add_argument('--base_url', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
        
    extractor = BioMiner_Markush_Infernce(args.vision_mllm_type, args.markush_prompt, 
                                          args.image2bboxindex, args.layout_seg_json,
                                          base_url=args.base_url, api_key=args.api_key)

    markush_image_paths = []
    markush_save_paths = []
    for dir in ['hybrid', 'image', 'text']:
        files = [file for file in os.listdir(os.path.join(args.markush_data_dir, dir)) if file.endswith('.png') and '-' not in file]
        images_layout_seg = [file for file in os.listdir(os.path.join(args.markush_data_dir, dir)) if '-' in file]

        for img in tqdm(images_layout_seg):
            img_path = os.path.join(args.markush_data_dir, dir, img)
            os.system(f'rm {img_path}')

        markush_image_paths.extend([os.path.join(args.markush_data_dir, dir, file) for file in files])
        file_names = [file.split('.')[0] for file in files]
        markush_save_paths.extend([os.path.join(args.save_dir, f'{dir}_{name}.json') for name in file_names])

    results = extractor.markush_zip_with_index_batch_split_image_layout(markush_image_paths,
                                                                        -1, 
                                                                        False, 
                                                                        args.cot)

    for res, save_path in zip(results, markush_save_paths):
        json_data = json.dumps(res, indent=4)
        with open(save_path, 'w') as f:
            f.write(json_data)  