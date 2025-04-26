import argparse
from tqdm import tqdm
import os 
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import json
from BioMiner.commons.mol_detection import draw_bbox_xywh
from BioMiner.commons.utils import pmap_multi
from BioMiner.commons.process_pdf import pdf_load_pypdf_images


yolo_name = 'MOL-v11l-241113.pt'
# yolo_name = 'moldet_yolo11l_960_doc.pt'

if yolo_name == 'MOL-v11l-241113.pt':
    det_model = YOLO("BioMiner/commons/MOL-v11l-241113.pt").to("cuda:0")
elif yolo_name == 'moldet_yolo11l_960_doc.pt':
    det_model = YOLO("BioMiner/commons/moldet_yolo11l_960_doc.pt").to("cuda:0")

def det_batch_images(imgs, page_names, iou_threshold=0.85):
    # input: list of PIL images
    # return: list of list images
    if yolo_name == 'MOL-v11l-241113.pt':
        results1 = det_model.predict(imgs, imgsz=640, conf=0.5)  # 模型预测结果
    elif yolo_name == 'moldet_yolo11l_960_doc.pt':
        results1 = det_model.predict(imgs, imgsz=960, conf=0.5)

    all_mols = []
    all_bboxes = []
    all_pages = []
    for idx in tqdm(range(len(imgs)), 'yolo molecule detection... '):
        result1 = results1[idx]
        img = imgs[idx]
        page = page_names[idx]
        img_width, img_height = img.size
        img_area = img_width * img_height
        
        mols = []
        bboxes = []
        pages = []
        boxes1 = result1.boxes
        total_box_area = 0

        # 计算每个检测框的面积，并统计所有检测框的面积和
        for xyxy in boxes1.xyxy:
            x1, y1, x2, y2 = xyxy.detach().cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            total_box_area += box_area

        # 如果检测框占比超过 85%，直接添加原图
        if total_box_area / img_area > iou_threshold:
            mols.append(img)
            bboxes.append([0, 0, 1, 1])
        else:
            # 否则对每个检测框进行裁剪
            for xyxy in boxes1.xyxy:
                x1, y1, x2, y2 = xyxy.detach().cpu().numpy()
                mols.append(img.crop((x1, y1, x2, y2)))
                bboxes.append([x1 / img_width, y1 / img_height, (x2-x1) / img_width, (y2-y1) /img_height])
                pages.append(page)

            # index = 0
            # for xyxy in boxes1.xyxy:
            #     x1, y1, x2, y2 = xyxy.detach().cpu().numpy()
            #     # draw_bbox_xywh(img, x1 / img_width, y1 / img_height, (x2-x1)/ img_width, (y2-y1)/img_height, index)
            #     index += 1

        # img.save('./molparser_detect_demo.png')
        all_mols.extend(mols)
        all_bboxes.extend(bboxes)
        all_pages.extend(pages)

    return all_mols, all_bboxes, all_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', type=str, default='example/pdfs')
    parser.add_argument('--file_name', type=str, default='BioVista/molecule_detection/data/file_names.txt')
    parser.add_argument('--page_image_dir', type=str, default='BioVista/temp_page_images')
    parser.add_argument('--save_dir', type=str, default='BioVista/component_result/molecule_deteciton/YOLO_box_640')
    args = parser.parse_args()
    
    with open(args.file_name, 'r') as f:
        names = f.read().strip().split('\n')

    pdf_paths = []
    for name in names:
        pdf_paths.append(os.path.join(args.pdf_dir, f'{name}.pdf'))

    # page_image_pathss = pmap_multi(pdf_load_pypdf_images, 
    #                                 zip(names, pdf_paths),
    #                                 save_path=args.page_image_dir,
    #                                 n_jobs=32,
    #                                 desc='converting pdf pages to images')
    
    os.makedirs(args.save_dir, exist_ok=True)

    for name in tqdm(names, desc='yolo molecule detection... '):
        files = os.listdir(os.path.join(args.page_image_dir, name))

        png_parsed_images = [file for file in files if 'bbox' not in file and file.endswith('png') and len(file.split('_')) == 4]

        _, all_bboxes, all_pages = det_batch_images([Image.open(os.path.join(args.page_image_dir, name, image)) for image in png_parsed_images],
                                                    [int(image.split('.')[0].split('_')[-1]) for image in png_parsed_images])

        res_list = []
        for index, (bbox, page) in enumerate(zip(all_bboxes, all_pages)):
            res_list.append({'index':index,
                             'page':page,
                             'bbox':bbox
                             })
            
        json_data = json.dumps(res_list, indent=4)
        with open(os.path.join(args.save_dir, f'{name}.pdf.json'), 'w') as f:
            f.write(json_data)
