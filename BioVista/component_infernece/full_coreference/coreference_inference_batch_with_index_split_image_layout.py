import argparse
from BioMiner import BioMiner_Markush_Infernce
import os 
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_mllm_type', default='gemini-2.0-flash')
    parser.add_argument('--full_coreference_prompt', default='BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_full_updated_with_index.txt')
    parser.add_argument('--full_coreference_data_dir', default='BioVista/full_coreference/data')
    parser.add_argument('--save_dir', default='BioVista/component_result/full_coreference/gemini-2.0-flash')
    parser.add_argument('--image2bboxindex', default='BioVista/full_coreference/image2bboxindex.json')
    parser.add_argument('--layout_seg_json', default='BioVista/full_coreference/figure_table_layout.json')
    parser.add_argument('--cot', default=False)
    parser.add_argument('--split_bbox_num', default=4)
    parser.add_argument('--segment_image', default=True)
    parser.add_argument('--base_url', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    extractor = BioMiner_Markush_Infernce(args.vision_mllm_type, args.full_coreference_prompt, 
                                          args.image2bboxindex, args.layout_seg_json,
                                          base_url=args.base_url, api_key=args.api_key)
    
    images = [file for file in os.listdir(args.full_coreference_data_dir) if file.endswith('.png')  and '-' not in file]
    images_layout_seg = [file for file in os.listdir(args.full_coreference_data_dir) if '-' in file]

    for img in tqdm(images_layout_seg):
        img_path = os.path.join(args.full_coreference_data_dir, img)
        os.system(f'rm {img_path}')

    coreference_full_image_paths = []
    coreference_full_save_paths = []
    for image in images:
        coreference_full_image_paths.append(os.path.join(args.full_coreference_data_dir, image))
        name = image.split('.')[0]
        coreference_full_save_paths.append(os.path.join(args.save_dir, f'{name}.json'))

    results = extractor.markush_zip_with_index_batch_split_image_layout(coreference_full_image_paths,
                                                                 args.split_bbox_num, args.segment_image, 
                                                                 args.cot)

    for res, save_path in zip(results, coreference_full_save_paths):
        with open(save_path, 'w') as f:
            json.dump(res, f, indent=4)
