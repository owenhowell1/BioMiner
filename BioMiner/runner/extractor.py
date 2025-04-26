import os 
import json
from tqdm import tqdm
from BioMiner import commons, runner, dataset
import pandas as pd
import numpy as np
import json
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import time
import random
import math
import cv2
from collections import defaultdict

def image_segment_given_box_xywh(image_path, save_path, segment_size):
    x, y, w, h = segment_size
    image_origin = cv2.imread(image_path)
    height, width = image_origin.shape[:2]
    x_start_pixel = int(x * width)
    x_end_pixel = int((x + w) * width)
    y_start_pixel = int (y * height)
    y_end_pixel = int( (y+ h) * height) 
    segmented_image = image_origin[y_start_pixel: y_end_pixel, x_start_pixel:x_end_pixel]
    cv2.imwrite(save_path, segmented_image)
    return save_path

def draw_bbox_xywh(image, x, y, w, h, index, bbox_background=True):
    # modify image in-place
    draw = ImageDraw.Draw(image)
    width = image.width
    height = image.height
    top_left_x = float(x * width)
    top_left_y = float(y * height)
    bottom_right_x = float((x + w) * width)
    bottom_right_y = float((y + h)  * height)
    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)
    top_right = (bottom_right_x, top_left_y)
    bottom_left = (top_left_x, bottom_right_y)
    line_width = 2
    line_color = (255, 0, 0)
    draw.line([top_left, top_right], fill=line_color, width=line_width)
    draw.line([top_right, bottom_right], fill=line_color, width=line_width)
    draw.line([bottom_right, bottom_left], fill=line_color, width=line_width)
    draw.line([bottom_left, top_left], fill=line_color, width=line_width)

    text = str(index)
    text_color = (255, 0, 0)
    text_position = ((top_left_x, top_left_y))
    
    font = ImageFont.truetype(font='./BioMiner/commons/arial.ttf', size=25)
    if bbox_background:
        left, top, right, bottom = draw.textbbox(text_position, text, font=font)
        draw.rectangle((left-5, top-5, right+5, bottom+5), fill='white')
    draw.text(text_position, text, font=font, fill=text_color)
    
    return 

def load_bioactivity_text_result(name, text_model_output_path, bioactivity_text_suffix):
    bioactivity_text_csv_path = os.path.join(text_model_output_path, name, f'{name}_text{bioactivity_text_suffix}_output.csv')
    data_df = pd.read_csv(bioactivity_text_csv_path)
    data_list = runner.mllm.bioactivity_data_df_to_list(data_df)
    return data_list


def load_bioactivity_image_result(name, vision_model_output_path, bioactivity_image_suffix):
    bioactivity_image_csv_path = os.path.join(vision_model_output_path, name, f'{name}_image{bioactivity_image_suffix}_output.csv')
    try:
        data_df = pd.read_csv(bioactivity_image_csv_path)
    except Exception as e:
        print(f'load image error {bioactivity_image_csv_path}')
        raise ValueError(e)
    data_list = runner.mllm.bioactivity_data_df_to_list(data_df)
    return data_list


def save_bioactivity_results(name, data_list_text, data_list_image, data_list, text_model_output_path, vision_model_output_path,
                          bioactivity_text_suffix, bioactivity_image_suffix):
    if not os.path.exists(os.path.join(text_model_output_path, name)):
        os.mkdir(os.path.join(text_model_output_path, name))

    if not os.path.exists(os.path.join(vision_model_output_path, name)):
        os.mkdir(os.path.join(vision_model_output_path, name))

    bioactivity_text_json_path = os.path.join(text_model_output_path, name, f'{name}_text{bioactivity_text_suffix}_output.json')
    bioactivity_text_csv_path = os.path.join(text_model_output_path, name, f'{name}_text{bioactivity_text_suffix}_output.csv')
    bioactivity_image_csv_path = os.path.join(vision_model_output_path, name, f'{name}_image{bioactivity_image_suffix}_output.csv')
    bioactivity_total_csv_path = os.path.join(text_model_output_path, name, f'{name}_merged{bioactivity_text_suffix}{bioactivity_image_suffix}_output.csv')
    bioactivity_unique_total_csv_path = os.path.join(text_model_output_path, name, f'{name}_merged{bioactivity_text_suffix}{bioactivity_image_suffix}_output_unique.csv')

    with open(bioactivity_text_json_path, 'w') as f:
        json.dump(data_list_text, f, indent=4)

    data_dict_text = runner.mllm.bioactivity_data_list_to_dict(data_list_text)
    data_dict_image = runner.mllm.bioactivity_data_list_to_dict(data_list_image)
    data_dict = runner.mllm.bioactivity_data_list_to_dict(data_list)

    pd.DataFrame(data_dict_text).to_csv(bioactivity_text_csv_path, index=False)
    pd.DataFrame(data_dict_image).to_csv(bioactivity_image_csv_path, index=False)
    pd.DataFrame(data_dict).to_csv(bioactivity_total_csv_path, index=False)
    pd.DataFrame(data_dict).drop_duplicates().to_csv(bioactivity_unique_total_csv_path, index=False)

    return


def save_structure_results(name, model_output_path, full_suffix, part_suffix, data_list_full, data_list_part, data_list):
    if not os.path.exists(os.path.join(model_output_path, name)):
        os.mkdir(os.path.join(model_output_path, name))
        
    structure_full_csv_path = os.path.join(model_output_path, name, f'{name}_structure_full_output_{full_suffix}.csv')
    structure_part_csv_path = os.path.join(model_output_path, name, f'{name}_structure_part_output_{part_suffix}.csv')
    structure_total_csv_path = os.path.join(model_output_path, name, f'{name}_structure_merged_output_{full_suffix}_{part_suffix}.csv')

    data_dict_full = runner.mllm.structure_data_list_to_dict(data_list_full)
    data_dict_part = runner.mllm.structure_data_list_to_dict(data_list_part)
    data_dict = runner.mllm.structure_data_list_to_dict(data_list)

    pd.DataFrame(data_dict_full).to_csv(structure_full_csv_path, index=False)
    pd.DataFrame(data_dict_part).to_csv(structure_part_csv_path, index=False)
    pd.DataFrame(data_dict).to_csv(structure_total_csv_path, index=False)

    return

def save_image_bioactivity_result_individually(name, model_output_path, suffix, 
                                            image_data_list, lig_image_paths, 
                                            image_api_output_list):

    if not os.path.exists(os.path.join(model_output_path, name)):
        os.mkdir(os.path.join(model_output_path, name))
    
    for (image_data, lig_image_path, api_output) in zip(image_data_list, lig_image_paths, image_api_output_list):
        image_name = os.path.basename(lig_image_path).split('.')[0]
        bioactivity_csv_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.csv')
        bioactivity_json_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.json')
        api_output_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.txt')
        data_dict = runner.mllm.bioactivity_data_list_to_dict(image_data)
        pd.DataFrame(data_dict).to_csv(bioactivity_csv_path, index=False)
        with open(api_output_path, 'w') as f:
            json.dump(api_output, f)
        with open(bioactivity_json_path, 'w') as f:
            json.dump(image_data, f, indent=4)
            
    return

def save_image_structure_result_individually(name, model_output_path, suffix, 
                                            image_data_list, lig_image_paths, 
                                            image_api_output_list, coreference_json_list_individual_images):

    if not os.path.exists(os.path.join(model_output_path, name)):
        os.mkdir(os.path.join(model_output_path, name))
    
    for (image_data, lig_image_path, api_output, coreference_json_list) in zip(image_data_list, lig_image_paths, image_api_output_list, coreference_json_list_individual_images):
        image_name = os.path.basename(lig_image_path).split('.')[0]
        structure_csv_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.csv')
        structure_json_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.json')
        api_output_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.txt')
        coreference_json_path = os.path.join(model_output_path, name, f'{image_name}_coreference_{suffix}.json')
        data_dict = runner.mllm.structure_data_list_to_dict(image_data)
        pd.DataFrame(data_dict).to_csv(structure_csv_path, index=False)
        with open(api_output_path, 'w') as f:
            json.dump(api_output, f)
        with open(structure_json_path, 'w') as f:
            json.dump(image_data, f, indent=4)
        with open(coreference_json_path, 'w') as f:
            json.dump(coreference_json_list, f, indent=4)
    return


def check_bioactivity_text_state(name, model_output_path, bioactivity_text_suffix, overwrite):
    bioactivity_csv_path = os.path.join(model_output_path, name, f'{name}_text{bioactivity_text_suffix}_output.csv')

    if not os.path.exists(bioactivity_csv_path) or overwrite:
        return False
    else:
        return True

def check_bioactivity_image_state(name, model_output_path, bioactivity_image_suffix, overwrite):
    bioactivity_csv_path = os.path.join(model_output_path, name, f'{name}_image{bioactivity_image_suffix}_output.csv')

    if not os.path.exists(bioactivity_csv_path) or overwrite:
        return False
    else:
        return True


def check_structure_state(name, model_output_path, suffix, overwrite):
    structure_total_csv_path = os.path.join(model_output_path, name, f'{name}_structure_merged_output_{suffix}.csv')

    if not os.path.exists(structure_total_csv_path) or overwrite:
        return False
    else:
        return True

def correct_output_failed_for_json(output):
    # 1. for long string ouput, it may lose the right bracket '}'
    if output[-2:] == '}]' and output[-3:] != '}}]':
        return output[:-1] + '}]'
    
    if output[-3:] == '}]}' and output[-3:] != '}}]':
        return output[:-2] + '}]'
    
def extraction_text_bioactivity_step(name, client, mllm_type, text_context, prompt, cot,
                                  model_output_path, suffix, overwrite):

    data_json_path = os.path.join(model_output_path, name, f'{name}_text{suffix}_output.json')

    if os.path.exists(data_json_path) and not overwrite:
        with open(data_json_path, 'r') as f:
            data_list = json.load(f)
    else:
        # print(f'{data_json_path} not exits ')
        # 1. call api with prompt and input text, origin output of client api
        output_origin = runner.mllm.call_api_text(client, prompt, mllm_type, text_context, cot)

        # 2. clean and process output of api
        # string type, [{"protein": "", "ligand": "" , "affinity": {"type": "", "value": "", "unit": ""}}]
        output = runner.mllm.process_api_output(output_origin)

        # 3. convert string output into json list
        # data_json = json.loads(output)
        try:
            data_json_temp = json.loads(output)
        except:
            # usually losing '}' in long string output
            output_temp = correct_output_failed_for_json(output)
            try:
                data_json_temp = json.loads(output_temp)
            except:
                print(f'Error text bioactivity extraction')
                data_json_temp = []

        # 4. convert json list output into self list output to filter undesired keys
        # List: [extracted_info, ...]             
        # extracted_info = {"protein": item["protein"], 
        #                   "ligand": item["ligand"], 
        #                   "affinity": {"type": item["affinity"]["type"],
        #                                "value": item["affinity"]["value"],
        #                                "unit": item["affinity"]["unit"]}}
        data_list = runner.mllm.bioactivity_data_api_output_to_list(data_json_temp, mllm_type)

    return data_list

def extraction_image_bioactivity_step(name, client, mllm_type, image_paths, prompt, cot,
                                   model_output_path, suffix, overwrite):
    data_list, data_list_individual_images, api_output_list = [], [], []

    for image_path in image_paths:
        image_name = os.path.basename(image_path).split('.')[0]
        api_output_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.txt')
        image_data_json_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.json')
        if os.path.exists(image_data_json_path) and not overwrite:
            # try:
            with open(api_output_path, 'r') as f:
                output_origin = json.load(f)[0]
            # except Exception as e:
            #     print(api_output_path)
            #     output_origin = ''
                # raise ValueError(e)
            with open(image_data_json_path, 'r') as f:
                data_list_temp = json.load(f)
        else:
            # 1. load image
            byte_io = io.BytesIO()
            with Image.open(image_path) as img:
                img.save(byte_io, format='png')
            bytes_image = byte_io.getvalue()
            image = base64.b64encode(bytes_image).decode('utf-8')

            # 2. origin output of client api, call api with prompt and input text, image
            output_origin = runner.mllm.call_api_iamge(client, prompt, mllm_type, image, cot=cot)

            # 3. clean and process output of api
            # string type, [{"protein": "", "ligand": "" , "affinity": {"type": "", "value": "", "unit": ""}}]
            output = runner.mllm.process_api_output(output_origin)

            # 4. convert string output into json list
            try:
                data_json_temp = json.loads(output)
            except:
                # usually losing '}' in long string output
                output_temp = correct_output_failed_for_json(output)
                try:
                    data_json_temp = json.loads(output_temp)
                except:
                    print(f'Error image bioactivity extraction: {image_path}')
                    data_json_temp = []

            # 5. convert json list output into self list output to filter undesired keys, list item:                 
            # extracted_info = {"protein": item["protein"], 
            #                   "ligand": item["ligand"], 
            #                   "affinity": {"type": item["affinity"]["type"],
            #                                "value": item["affinity"]["value"],
            #                                "unit": item["affinity"]["unit"]}}
            data_list_temp = runner.mllm.bioactivity_data_api_output_to_list(data_json_temp, mllm_type)

        data_list.extend(data_list_temp)
        data_list_individual_images.append(data_list_temp)
        api_output_list.append([output_origin])

    return data_list, data_list_individual_images, api_output_list

def call_api_image_run(vision_mllm_client, image_path, bbox_index, prompt, vision_mllm_type, cot):
    byte_io = io.BytesIO()
    with Image.open(image_path) as img:
        img.save(byte_io, format='png')
    bytes_image = byte_io.getvalue()
    image = base64.b64encode(bytes_image).decode('utf-8')

    new_prompt = prompt.replace('<index></index>', str(bbox_index))

    if cot:
        new_prompt += '''\nLet's think step by step.'''

    output_origin = runner.mllm.call_api_iamge(vision_mllm_client, new_prompt, vision_mllm_type, image)
    return output_origin

def extraction_ligand_structure_run(client, image_path, bbox_index_dict, prompt, vision_mllm_type, 
                                    cot, split_bbox_num, segment_image, enlarge_size):

    bbox_index =  list(bbox_index_dict.keys())
    file_name = os.path.basename(image_path).split('.')[0]
    dir_name = os.path.dirname(image_path)

    if split_bbox_num is None or len(bbox_index) <= split_bbox_num:
        if segment_image:
            current_bbox = []
            for b_i in  bbox_index:
                x, y, w, h, _ = bbox_index_dict[b_i]
                current_bbox.append((x,y,w,h))

            image_seg_path = os.path.join(dir_name, f'{file_name}-seg.png')
            segment_size = get_segment_size(current_bbox, enlarge_size)
            image_segment_given_box_xywh(image_path, image_seg_path, segment_size)

            return [call_api_image_run(client, image_seg_path, bbox_index, prompt, vision_mllm_type, cot)]
        else:
            return [call_api_image_run(client, image_path, bbox_index, prompt, vision_mllm_type, cot)]
    
    origin_image_path = bbox_index_dict[bbox_index[0]][-1]

    split_image_paths = []
    current_bbox_idx = []
    current_bbox = []
    output_origin_total = []
    image = Image.open(origin_image_path)

    for b_i in  bbox_index:
        x, y, w, h, _ = bbox_index_dict[b_i]
        current_bbox.append((x,y,w,h))
        draw_bbox_xywh(image, x, y, w, h, b_i)
        current_bbox_idx.append(b_i)

        if len(current_bbox_idx) == split_bbox_num:
            current_image_idx = len(split_image_paths)
            current_split_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}.png')
            current_split_seg_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}-seg.png')

            if segment_image:
                image.save(current_split_image_path)
                segment_size = get_segment_size(current_bbox, enlarge_size)
                image_segment_given_box_xywh(current_split_image_path, current_split_seg_image_path, segment_size)
                output_origin_temp = call_api_image_run(client, current_split_seg_image_path, current_bbox_idx, prompt, vision_mllm_type, cot)

            else:
                image.save(current_split_image_path)
                output_origin_temp = call_api_image_run(client, current_split_image_path, current_bbox_idx, prompt, vision_mllm_type, cot)

            current_bbox_idx = []
            current_bbox = []
            split_image_paths.append(current_split_image_path)
            output_origin_total.append(output_origin_temp)
            image = Image.open(origin_image_path)

    if len(current_bbox_idx) > 0:
        current_image_idx = len(split_image_paths)
        current_split_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}.png')
        current_split_seg_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}-seg.png')

        if segment_image:
            image.save(current_split_image_path)
            segment_size = get_segment_size(current_bbox, enlarge_size)
            image_segment_given_box_xywh(current_split_image_path, current_split_seg_image_path, segment_size)
            output_origin_temp = call_api_image_run(client, current_split_seg_image_path, current_bbox_idx, prompt, vision_mllm_type, cot)

        else:
            image.save(current_split_image_path)
            output_origin_temp = call_api_image_run(client, current_split_image_path, current_bbox_idx, prompt, vision_mllm_type, cot)

        current_bbox_idx = []
        split_image_paths.append(current_split_image_path)
        output_origin_total.append(output_origin_temp)
        image = Image.open(origin_image_path)

    return output_origin_total

def extraction_ligand_structure_full_step(name, client, mllm_type, image_paths, 
                                          bbox_index_dict_list, figure_table_layout_bbox_list, 
                                          index2smiles, prompt, model_output_path, suffix, cot, 
                                          split_bbox_num, segment_image, enlarge_size, 
                                          layout_seg, bbox_background, overwrite):
    processed_data_list_total = [] # for final result 
    processed_data_list_individual_images = [] # for individual check
    api_output_list = [] # for individual check
    coreference_json_list_individual_images = []  # for human in loop debug

    for image_path, bbox_index_dict, figure_table_layout_bbox in zip(image_paths, bbox_index_dict_list, figure_table_layout_bbox_list):
        image_name = os.path.basename(image_path).split('.')[0]
        api_output_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.txt')
        image_data_json_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.json')
        coreference_json_path = os.path.join(model_output_path, name, f'{image_name}_coreference_{suffix}.json')
        if os.path.exists(coreference_json_path) and not overwrite:
            with open(api_output_path, 'r') as f:
                output_origin_list = json.load(f)
            with open(coreference_json_path, 'r') as f:
                coreference_json_list = json.load(f)
        else:
            output_origin_list = []
            coreference_json_list = []

            dir_name = os.path.dirname(image_path)
            file_name = os.path.basename(image_path).split('.')[0]
            dir_files = os.listdir(dir_name)
            for f in dir_files:
                if 'seg' in f and 'full' in f and file_name in f:
                    os.system(f'rm {dir_name}/{f}')

            if not layout_seg or figure_table_layout_bbox is None:
                figure_table_layout_bbox = [[0,0,1.0,1.0]]
            else:
                segment_image = False

            layout_seg_image_paths, layout_seg_image_bbox_index_dicts = segment_image_by_layout(image_path, figure_table_layout_bbox, bbox_index_dict, bbox_background)

            for layout_seg_image_path, layout_seg_image_bbox_index_dict in zip(layout_seg_image_paths, layout_seg_image_bbox_index_dicts):
                # continue
                if len(layout_seg_image_bbox_index_dict.keys()) == 0:
                    continue

                segment_image_paths, segment_image_bbox_indexs = segment_image_to_new_images(layout_seg_image_path, layout_seg_image_bbox_index_dict, split_bbox_num, segment_image, enlarge_size, bbox_background)

                for segment_image_path, segment_image_bbox_index in zip(segment_image_paths, segment_image_bbox_indexs):
                    # 1. load image
                    try:
                        byte_io = io.BytesIO()            
                        with Image.open(segment_image_path) as img:
                            img.save(byte_io, format='png')
                        bytes_image = byte_io.getvalue()
                        image = base64.b64encode(bytes_image).decode('utf-8')
                    except:
                        raise ValueError(f'error image path {segment_image_path}')
                    # 2. origin output of client api, call api with prompt and input text, augmented image
                    output_origin = runner.mllm.call_api_iamge(client, prompt.replace('<index></index>', str(segment_image_bbox_index)), 
                                                               mllm_type, image, cot=cot)

                    # 3. clean and process output of api
                    # string type, [{"index": "", "identifier": ""}]
                    try:
                        output = runner.mllm.process_api_output(output_origin)
                    except:
                        print(f'Error 1 image structure extraction: {image_path}')
                        output = []
                    
                    # 4. convert string output into json list
                    try:
                        data_json_temp = json.loads(output)
                    except:
                        # usually losing '}' in long string output
                        output_temp = correct_output_failed_for_json(output)
                        try:
                            data_json_temp = json.loads(output_temp)
                        except:
                            print(f'Error 2 image structure extraction: {image_path}')
                            data_json_temp = []

                    output_origin_list.append(output_origin)
                    coreference_json_list.append(data_json_temp)

        image_data_list = []
        for data_json_temp in coreference_json_list:
            # 5. convert json list output into self list output to filter undesired keys and map index into smiles
            # List: [extracted_info, ...]                         
            # extracted_info = {"identifier": item["identifier"], "smiles": index2smiles[item['index']]}
            data_list_temp = runner.mllm.full_structure_data_api_putput_to_list(data_json_temp, mllm_type, index2smiles)
            image_data_list.extend(data_list_temp)

        processed_data_list_total.extend(image_data_list)
        processed_data_list_individual_images.append(image_data_list)
        api_output_list.append(output_origin_list)
        coreference_json_list_individual_images.append(coreference_json_list)

    return processed_data_list_total, processed_data_list_individual_images, api_output_list, coreference_json_list_individual_images

def extraction_ligand_structure_part_step(name, client, mllm_type, image_paths, 
                                          bbox_index_dict_list, figure_table_layout_bbox_list, 
                                          index2smiles, prompt, model_output_path, suffix, cot, 
                                          layout_seg, bbox_background, overwrite):
    processed_data_list_total = []
    processed_data_list_individual_images = []
    api_output_list = []
    coreference_json_list_individual_images = []  # for human in loop debug

    # print(image_paths)
    for image_path, bbox_index_dict, figure_table_layout_bbox in zip(image_paths, bbox_index_dict_list, figure_table_layout_bbox_list):
        image_name = os.path.basename(image_path).split('.')[0]
        api_output_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.txt')
        image_data_json_path = os.path.join(model_output_path, name, f'{image_name}_{suffix}.json')
        coreference_json_path = os.path.join(model_output_path, name, f'{image_name}_coreference_{suffix}.json')
        if os.path.exists(coreference_json_path) and not overwrite:
            with open(api_output_path, 'r') as f:
                output_origin_list = json.load(f)
            with open(coreference_json_path, 'r') as f:
                coreference_json_list = json.load(f)
        else:
            output_origin_list = []
            coreference_json_list = []
            if not layout_seg or figure_table_layout_bbox is None:
                figure_table_layout_bbox = [[0,0,1.0,1.0]]

            segment_image = False
            split_bbox_num = -1
            enlarge_size = 0.0

            dir_name = os.path.dirname(image_path)
            file_name = os.path.basename(image_path).split('.')[0]
            dir_files = os.listdir(dir_name)
            for f in dir_files:
                if 'seg' in f and 'part' in f and file_name in f:
                    os.system(f'rm {dir_name}/{f}')

            layout_seg_image_paths, layout_seg_image_bbox_index_dicts = segment_image_by_layout(image_path, figure_table_layout_bbox, bbox_index_dict, bbox_background)

            for layout_seg_image_path, layout_seg_image_bbox_index_dict in zip(layout_seg_image_paths, layout_seg_image_bbox_index_dicts):
                # continue
                if len(layout_seg_image_bbox_index_dict.keys()) == 0:
                    continue

                segment_image_paths, segment_image_bbox_indexs = segment_image_to_new_images(layout_seg_image_path, layout_seg_image_bbox_index_dict, split_bbox_num, segment_image, enlarge_size, bbox_background)

                for segment_image_path, segment_image_bbox_index in zip(segment_image_paths, segment_image_bbox_indexs):
                    # 1. load image
                    # print(image_path)
                    byte_io = io.BytesIO()
                    with Image.open(segment_image_path) as img:
                        img.save(byte_io, format='png')
                    bytes_image = byte_io.getvalue()
                    image = base64.b64encode(bytes_image).decode('utf-8')

                    # 2. origin output of client api, call api with prompt and input text, augmented image
                    output_origin = runner.mllm.call_api_iamge(client, prompt.replace('<index></index>', str(segment_image_bbox_index)), 
                                                                mllm_type, image, cot=cot)

                    # 3. clean and process output of api
                    # string type, [{"scaffold": "artificial index of scaffold",
                    #                "R-group": {"name of R-group1": "artifical index of R-group 1", ...}, 
                    #                "identifier": "molecule identifier"}]
                    try:
                        output = runner.mllm.process_api_output(output_origin)
                    except:
                        print(f'Error image structure extraction: {image_path}')
                        output = f'[]'
                    # 4. convert string output into json list
                    try:
                        data_json_temp = json.loads(output)
                    except:
                        # usually losing '}' in long string output
                        output_temp = correct_output_failed_for_json(output)
                        try:
                            data_json_temp = json.loads(output_temp)
                        except:
                            print(f'Error image structure extraction: {image_path}')
                            data_json_temp = []

                    output_origin_list.append(output_origin)
                    coreference_json_list.append(data_json_temp)


        image_data_list = []
        for data_json_temp in coreference_json_list:
            # 5. convert json list output into self list output to filter undesired keys and map index into smiles
            # List: [extracted_info, ...]                                         
            # extracted_info = {"identifier": item["identifier"], 
            #                   "smiles": index2smiles[item['index']]}
            data_list_temp = runner.mllm.part_structure_data_api_putput_to_list(data_json_temp, mllm_type, index2smiles)
            image_data_list.extend(data_list_temp)
    
        processed_data_list_total.extend(image_data_list)
        processed_data_list_individual_images.append(image_data_list)
        api_output_list.append(output_origin_list)
        coreference_json_list_individual_images.append(coreference_json_list)

    return processed_data_list_total, processed_data_list_individual_images, api_output_list, coreference_json_list_individual_images


def merge_step(data_list_text, data_list_image, merge_strategy):

    if merge_strategy == 'direct':
        return data_list_text + data_list_image
    
    elif merge_strategy == 'mllm_merge':
        return 

def extraction_bioactivity_step(name, text_mllm_type, text_client, vision_mllm_type, vision_client, 
                             text_context, image_paths, text_prompt, image_prompt,
                             merge_strategy, text_model_output_path, vision_model_output_path,
                             bioactivity_text_suffix, bioactivity_image_suffix, cot, overwrite_text, overwrite_image):
    

    data_list_text = extraction_text_bioactivity_step(name,
                                                    text_client, 
                                                    text_mllm_type,
                                                    text_context, 
                                                    text_prompt,
                                                    cot,
                                                    text_model_output_path,
                                                    bioactivity_text_suffix,
                                                    overwrite_text)
    
    # 2. extract bioactivity data in figures and tables
    data_list_image, data_list_individual_images, api_output_list = extraction_image_bioactivity_step(name, 
                                                                                                    vision_client, 
                                                                                                    vision_mllm_type, 
                                                                                                    image_paths,
                                                                                                    image_prompt,
                                                                                                    cot,
                                                                                                    vision_model_output_path,
                                                                                                    bioactivity_image_suffix,
                                                                                                    overwrite_image)

    # 3. merge bioactivity data from different modalities
    # data_list = [{"protein": item["protein"], 
    #               "ligand": item["ligand"], 
    #               "affinity": {"type": item["affinity"]["type"],
    #                            "value": item["affinity"]["value"],
    #                            "unit": item["affinity"]["unit"]}}, 
    #                ...]
    data_list = merge_step(data_list_text, data_list_image, merge_strategy)

    # 4. save bioactivity extraction results
    save_bioactivity_results(name, data_list_text, data_list_image, data_list, 
                          text_model_output_path, vision_model_output_path, 
                          bioactivity_text_suffix, bioactivity_image_suffix)
    
    save_image_bioactivity_result_individually(name, vision_model_output_path, bioactivity_image_suffix,
                                            data_list_individual_images, image_paths, api_output_list)

    return data_list


def extraction_ligand_structure_step(name, mllm_type, client, full_lig_image_paths, 
                                     part_lig_image_paths, index2smiles, 
                                     pdf_augmented_full_image_bbox_index_dict,
                                     pdf_augmented_part_image_bbox_index_dict, 
                                     full_page_figure_table_layout_bbox, 
                                     part_page_figure_table_layout_bbox,
                                     full_lig_prompt, part_lig_prompt, 
                                     model_output_path, full_suffix, part_suffix, cot,
                                     split_bbox_num, segment_image, enlarge_size, 
                                     structure_full_layout_seg, structure_part_layout_seg, 
                                     bbox_background, overwrite_full, overwrite_part):
    # # 0. wheather pass       
    # if check_structure_state(name, model_output_path, suffix, overwrite):
    #     return 

    # print(name, 'structure')
    # 1. extract full ligand structure
    # print('extract full')
    # print(full_lig_image_paths)
    data_list_total_full, \
        data_list_individual_images_full, \
            full_image_api_output_list, \
                 coreference_json_list_individual_images_full = extraction_ligand_structure_full_step(name,
                                                                                client, 
                                                                                mllm_type, 
                                                                                full_lig_image_paths, 
                                                                                pdf_augmented_full_image_bbox_index_dict, 
                                                                                full_page_figure_table_layout_bbox, 
                                                                                index2smiles, 
                                                                                full_lig_prompt,
                                                                                model_output_path,
                                                                                full_suffix, 
                                                                                cot,
                                                                                split_bbox_num, 
                                                                                segment_image, 
                                                                                enlarge_size, 
                                                                                structure_full_layout_seg, 
                                                                                bbox_background, 
                                                                                overwrite_full)
    # print('extract part')
    # print(part_lig_image_paths)
    # 2. extract ligand structure consist of scaffold and functional group
    data_list_total_part, \
        data_list_individual_images_part, \
            part_image_api_output_list, \
                 coreference_json_list_individual_images_part = extraction_ligand_structure_part_step(name,
                                                                                client, 
                                                                                mllm_type, 
                                                                                part_lig_image_paths, 
                                                                                pdf_augmented_part_image_bbox_index_dict, 
                                                                                part_page_figure_table_layout_bbox,
                                                                                index2smiles, 
                                                                                part_lig_prompt,
                                                                                model_output_path,
                                                                                part_suffix,
                                                                                cot,
                                                                                structure_part_layout_seg,
                                                                                bbox_background, 
                                                                                overwrite_part)

    # data_list = [{"identifier": item["identifier"], "smiles": index2smiles[item['index']]}, ...]
    data_list = data_list_total_full + data_list_total_part

    # 3. save structure extraction results
    save_structure_results(name, model_output_path, full_suffix, part_suffix, data_list_total_full, data_list_total_part, data_list)

    save_image_structure_result_individually(name, model_output_path, full_suffix,
                                            data_list_individual_images_full, full_lig_image_paths, 
                                            full_image_api_output_list, coreference_json_list_individual_images_full)
    
    save_image_structure_result_individually(name, model_output_path, part_suffix, 
                                            data_list_individual_images_part, part_lig_image_paths, 
                                            part_image_api_output_list, coreference_json_list_individual_images_part)

    return data_list

def save_overall_result(name, bioactivity_data_list, structure_data_list,
                        model_output_path, full_suffix, part_suffix):
    df_extracted_structure_data = pd.DataFrame(runner.mllm.structure_data_list_to_dict(structure_data_list))
    df_extracted_bioactivity_data = pd.DataFrame(runner.mllm.bioactivity_data_list_to_dict(bioactivity_data_list))
    # 1. filter NaN of extracted structure data
    df_extracted_structure_data = df_extracted_structure_data.dropna()
    
    # 2. filter NaN of extracted bioactivity data
    df_extracted_bioactivity_data = df_extracted_bioactivity_data.dropna()

    # 3. converting ligand identifier of bioactivity data into ligand smiles
    try:
        df_extracted_bioactivity_data['smiles'] = np.array(runner.metric_fn.identifier_to_smiles(df_extracted_structure_data, 
                                                                                            df_extracted_bioactivity_data))
    except Exception as e:
        print(name)
        raise ValueError(e)
    na_smiles_df = df_extracted_bioactivity_data[df_extracted_bioactivity_data['smiles'].isna()]
    failed_ligand_names = list(set(na_smiles_df['ligand'].values.tolist()))

    df_extracted_bioactivity_data = df_extracted_bioactivity_data.drop_duplicates()

    df_extracted_bioactivity_data.to_csv(os.path.join(model_output_path, name, f'{name}_merge_{full_suffix}_{part_suffix}.csv'))    
    with open(os.path.join(model_output_path, name, f'{name}_merge_failed_ligand_{full_suffix}_{part_suffix}.json'), 'w') as f:
        json.dump(failed_ligand_names, f, indent=4)

    if 'human' in full_suffix:
        df_extracted_bioactivity_data.to_csv(f'human_nlpr3/{name}_merge_{full_suffix}_{part_suffix}.csv')
        
    return df_extracted_bioactivity_data

def extraction_specific_structure_bioactivity(bioactivity_data_list, structure_data_list, name, pdb_structure_path,
                                           model_output_path, full_suffix, part_suffix):
    
    given_smiles, given_protein = runner.metric_fn.load_pdb_protein_name_ligand_smiles(name, pdb_structure_path)
    df_extracted_structure_data = pd.DataFrame(runner.mllm.structure_data_list_to_dict(structure_data_list))
    df_extracted_bioactivity_data = pd.DataFrame(runner.mllm.bioactivity_data_list_to_dict(bioactivity_data_list))

    # 1. filter NaN of extracted structure data
    df_extracted_structure_data = df_extracted_structure_data.dropna()
    
    # 2. filter NaN of extracted bioactivity data
    df_extracted_bioactivity_data = df_extracted_bioactivity_data.dropna()

    # 3. converting ligand identifier of bioactivity data into ligand smiles
    try:
        df_extracted_bioactivity_data['smiles'] = np.array(runner.metric_fn.identifier_to_smiles(df_extracted_structure_data, 
                                                                                              df_extracted_bioactivity_data))
    except:
        raise ValueError(name)
    df_extracted_bioactivity_data = df_extracted_bioactivity_data.dropna()

    df_extracted_bioactivity_data['similarity'] = [commons.process_mol.calculate_similarity(s, given_smiles) for s in df_extracted_bioactivity_data['smiles']]

    df_extracted_bioactivity_data = df_extracted_bioactivity_data.sort_values(by=['similarity'], ascending=False)
    
    df_extracted_bioactivity_data.to_csv(os.path.join(model_output_path, name, f'{name}_bioactivity_given_structure_{full_suffix}_{part_suffix}.csv'))    

    return

def get_feedback_from_critic_agent():

    return

def extract_markush_part_with_bbox_index(image_path, bbox_index, markush_prompt,
                                         vision_mllm_type, cot, base_url, api_key):

    vision_mllm_client = runner.mllm.get_api_client(base_url, api_key)
    byte_io = io.BytesIO()
    with Image.open(image_path) as img:
        img.save(byte_io, format='png')
    bytes_image = byte_io.getvalue()
    image = base64.b64encode(bytes_image).decode('utf-8')

    new_prompt = markush_prompt.replace('<index></index>', str(bbox_index))

    if cot:
        new_prompt += '''\nLet's think step by step.'''

    output_origin = runner.mllm.call_api_iamge(vision_mllm_client, new_prompt, vision_mllm_type, image)

    output = runner.mllm.process_api_output(output_origin)
    
    try:
        data_json_temp = json.loads(output)
    except:
        # usually losing '}' in long string output
        output_temp = correct_output_failed_for_json(output)
        try:
            data_json_temp = json.loads(output_temp)
        except:
            print(f'Error image structure extraction: {image_path}')
            data_json_temp = []

    # process format for gpt
    if vision_mllm_type in runner.mllm.GPTSTPYEMODELS:
        if 'data'  in data_json_temp:
            data_json_temp = data_json_temp['data']
        
    if isinstance(data_json_temp, dict):
        data_json_temp = [data_json_temp]

    return data_json_temp


def get_segment_size(bbox_list, enlarge_size = 1/8):

    xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []
    for bbox in bbox_list:
        x, y, w, h = bbox
        xmin_list.append(x)
        ymin_list.append(y)
        xmax_list.append(x+w)
        ymax_list.append(y+h)

    xmin, ymin, xmax, ymax = min(xmin_list), min(ymin_list), max(xmax_list), max(ymax_list)
    xmin_enlarge = max(xmin - enlarge_size, 0)
    ymin_enlarge = max(ymin - enlarge_size, 0)
    xmax_enlarge = min(xmax + enlarge_size, 1)
    ymax_enlarge = min(ymax + enlarge_size, 1)

    new_x = xmin_enlarge
    new_y = ymin_enlarge
    new_w = xmax_enlarge - xmin_enlarge
    new_h = ymax_enlarge - ymin_enlarge
    return new_x, new_y, new_w, new_h

def segment_image_to_new_images(image_path, bbox_index_dict, split_bbox_num, 
                                segment_image, enlarge_size=1/8, bbox_background=True):

    bbox_index =  list(bbox_index_dict.keys())
    file_name = os.path.basename(image_path).split('.')[0]
    dir_name = os.path.dirname(image_path)

    # if molecule num < split bbox num 
    if len(bbox_index) <= split_bbox_num:
        if segment_image:
            current_bbox = []
            for b_i in  bbox_index:
                x, y, w, h, _ = bbox_index_dict[b_i]
                current_bbox.append((x,y,w,h))

            image_seg_path = os.path.join(dir_name, f'{file_name}-seg.png')
            segment_size = get_segment_size(current_bbox, enlarge_size)
            image_segment_given_box_xywh(image_path, image_seg_path, segment_size)

            return [image_seg_path], [bbox_index]
        else:
            return [image_path], [bbox_index]

    # split too many bbox
    origin_image_path = bbox_index_dict[bbox_index[0]][-1]
    split_image_paths, split_image_bbox_index = [], []
    current_bbox_idx, current_bbox = [], []
    image = Image.open(origin_image_path)

    for b_i in  bbox_index:
        x, y, w, h, _ = bbox_index_dict[b_i]
        current_bbox.append((x,y,w,h))
        draw_bbox_xywh(image, x, y, w, h, b_i, bbox_background=bbox_background)
        current_bbox_idx.append(b_i)

        if len(current_bbox_idx) == split_bbox_num:
            current_image_idx = len(split_image_paths)
            current_split_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}.png')
            current_split_seg_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}-seg.png')

            if segment_image:
                image.save(current_split_image_path)
                segment_size = get_segment_size(current_bbox, enlarge_size)
                image_segment_given_box_xywh(current_split_image_path, current_split_seg_image_path, segment_size)
                split_image_paths.append(current_split_seg_image_path)
            else:
                image.save(current_split_image_path)
                split_image_paths.append(current_split_image_path)

            split_image_bbox_index.append(current_bbox_idx)
            current_bbox_idx, current_bbox = [], []
            image = Image.open(origin_image_path)

    # process left bbox
    if len(current_bbox_idx) > 0:
        current_image_idx = len(split_image_paths)
        current_split_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}.png')
        current_split_seg_image_path = os.path.join(dir_name, f'{file_name}-{current_image_idx}-seg.png')

        if segment_image:
            image.save(current_split_image_path)
            segment_size = get_segment_size(current_bbox, enlarge_size)
            image_segment_given_box_xywh(current_split_image_path, current_split_seg_image_path, segment_size)
            split_image_paths.append(current_split_seg_image_path)
        else:
            image.save(current_split_image_path)
            split_image_paths.append(current_split_image_path)

        split_image_bbox_index.append(current_bbox_idx)
        current_bbox_idx, current_bbox = [], []
        image = Image.open(origin_image_path)


    return split_image_paths, split_image_bbox_index

def check_bbox_in_area(bbox, area_bbox):
    # print(area_bbox)
    x, y, w, h = bbox
    a_x, a_y, a_w, a_h = area_bbox

    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    a_xmin = a_x
    a_ymin = a_y
    a_xmax = a_x + a_w
    a_ymax = a_y + a_h

    if xmin >= a_xmin and xmax<=a_xmax and ymin>=a_ymin and ymax <= a_ymax:
        return True
    else:
        return False

def get_new_bbox_in_area(bbox, area_bbox):
    x, y, w, h = bbox
    a_x, a_y, a_w, a_h = area_bbox

    xmin = x
    ymin = y
    a_xmin = a_x
    a_ymin = a_y

    new_xmin = (xmin - a_xmin)/a_w
    new_ymin = (ymin - a_ymin)/a_h
    new_w = w/a_w
    new_h = h/a_h

    return new_xmin, new_ymin, new_w, new_h

def extract_markush_part_with_bbox_index_split_complex_image(image_path, bbox_index_dict, 
                                                             markush_prompt, vision_mllm_type,
                                                             cot, split_bbox_num, segment_image,
                                                             base_url, api_key):

    segment_image_paths, segment_image_bbox_indexs = segment_image_to_new_images(image_path, bbox_index_dict, split_bbox_num, segment_image)
    data_json_total = []
    for current_split_seg_image_path, current_bbox_idx in zip(segment_image_paths, segment_image_bbox_indexs):
        data_json_temp = extract_markush_part_with_bbox_index(current_split_seg_image_path, current_bbox_idx, markush_prompt, vision_mllm_type, cot, base_url, api_key)
        data_json_total.extend(data_json_temp)

    return data_json_total

def segment_image_by_layout(image_path, figure_table_layout_bbox, bbox_index_dict, bbox_background=True):

    file_name = os.path.basename(image_path).split('.')[0]
    dir_name = os.path.dirname(image_path)

    bbox_index =  list(bbox_index_dict.keys())
    layout_segment_flag = True
    layoutidx_to_bbox_idx = defaultdict(list)
    for b_i in bbox_index:
        x, y, w, h, _ = bbox_index_dict[b_i]
        bbox_in_layout_flag = False
        for idx, layout_bbox in enumerate(figure_table_layout_bbox):
            if check_bbox_in_area((x,y,w,h), layout_bbox):
                bbox_in_layout_flag = True
                layoutidx_to_bbox_idx[idx].append(b_i)
                break
        if bbox_in_layout_flag == False:
            layout_segment_flag = False

    if not layout_segment_flag:
        return [image_path], [bbox_index_dict]
    
    layout_seg_image_paths, layout_seg_image_bbox_index_dicts = [], []

    for idx, layout_bbox in enumerate(figure_table_layout_bbox):
        image_layout_seg_path_temp = os.path.join(dir_name, f'{file_name}-layout-seg-{idx}-temp.png')
        image_layout_seg_path = os.path.join(dir_name, f'{file_name}-layout-seg-{idx}.png')
        image_layout_seg_origin_path = os.path.join(dir_name, f'{file_name}-layout-origin-seg-{idx}.png')
        image_layout_bbox_index_dict = {}

        layout_bbox_idx = layoutidx_to_bbox_idx[idx]
        origin_image_path = bbox_index_dict[bbox_index[0]][-1]
        image = Image.open(origin_image_path)

        for b_i in layout_bbox_idx:
            x, y, w, h, _ = bbox_index_dict[b_i]
            # try:
            draw_bbox_xywh(image, x, y, w, h, b_i, bbox_background=bbox_background)
            # except Exception as e:
            #     print(f'error draw bbox {origin_image_path}')
            #     print(e)
            #     raise ValueError(f'error draw bbox {origin_image_path}')
            new_xmin, new_ymin, new_w, new_h = get_new_bbox_in_area((x, y, w, h,), layout_bbox)
            image_layout_bbox_index_dict[b_i] = (new_xmin, new_ymin, new_w, new_h, image_layout_seg_origin_path)
        
        if len(layout_bbox_idx) > 0:
            image.save(image_layout_seg_path_temp)
            image_segment_given_box_xywh(origin_image_path, image_layout_seg_origin_path, layout_bbox)
            image_segment_given_box_xywh(image_layout_seg_path_temp, image_layout_seg_path, layout_bbox)
            os.system(f'rm {image_layout_seg_path_temp}')
            layout_seg_image_paths.append(image_layout_seg_path)
            layout_seg_image_bbox_index_dicts.append(image_layout_bbox_index_dict)

    return layout_seg_image_paths, layout_seg_image_bbox_index_dicts
        

def extract_markush_part_with_bbox_index_split_complex_image_seg_layout(image_path, bbox_index_dict, figure_table_layout_bbox,
                                                                        markush_prompt, vision_mllm_type,
                                                                        cot, split_bbox_num, segment_image,
                                                                        base_url, api_key):

    if figure_table_layout_bbox is None:
        figure_table_layout_bbox = [[0,0,1.0,1.0]]
    else:
        segment_image = False
    layout_seg_image_paths, layout_seg_image_bbox_index_dicts = segment_image_by_layout(image_path, figure_table_layout_bbox, bbox_index_dict)
    data_json_total = []

    for layout_seg_image_path, layout_seg_image_bbox_index_dict in zip(layout_seg_image_paths, layout_seg_image_bbox_index_dicts):
        # continue
        if len(layout_seg_image_bbox_index_dict.keys()) == 0:
            continue
        data_json_temp = extract_markush_part_with_bbox_index_split_complex_image(layout_seg_image_path, layout_seg_image_bbox_index_dict, 
                                                                                    markush_prompt, vision_mllm_type,
                                                                                    cot, split_bbox_num, segment_image,
                                                                                    base_url, api_key)
        data_json_total.extend(data_json_temp)

    return data_json_total