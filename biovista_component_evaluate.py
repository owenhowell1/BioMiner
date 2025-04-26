from BioMiner import commons
import argparse
import os
from tqdm import tqdm
import pandas as pd
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import cv2

def calculate_f1_score(precision, recall):
    """
    Calculates the F1 score given precision and recall.

    Args:
        precision: The precision score (float between 0 and 1).
        recall: The recall score (float between 0 and 1).

    Returns:
        The F1 score (float between 0 and 1), or None if either precision
        or recall is zero and causes a division by zero error.
    """
    if precision + recall == 0:  # Handle the case where both are zero
        return 0.0 #or None, or raise ValueError, depending how you want to handle
                # a division by zero.  Returning 0 is a common choice.
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
def calculate_mean(lst):
    if len(lst) == 0:
        return 0  # 如果列表为空，返回0或其他适当的值
    return sum(lst) / len(lst)

class MarkushZipEvaluator(object):
    def __init__(self, gd_dir):
        self.gd_json_files = [file for file in os.listdir(gd_dir) if not file.endswith('.json.csv.json')]
        self.gd_dir = gd_dir
        self.gd_json_files.sort()

    def evaluate(self, pred_dir):
        self.evaluate_without_coreference(pred_dir)
        self.evluate_with_coreference(pred_dir, pred_dir)

    def evaluate_without_coreference(self, pred_dir, evaluate_save_dir=None):
        valid_accuracy, valid_recall = [], []
        res = {'name':[],
                'accuracy':[],
                'recall':[]}
        for json_file in tqdm(self.gd_json_files):
            pred_json_path = os.path.join(pred_dir, json_file)
            gd_json_path = os.path.join(self.gd_dir, json_file)
            accuracy, recall = self._evaluate_json_without_coreference(pred_json_path, gd_json_path, evaluate_save_dir)
            if accuracy != -1:
                valid_accuracy.append(accuracy)
            if recall != -1:
                valid_recall.append(recall)

            res['name'].append(json_file)
            res['accuracy'].append(accuracy)
            res['recall'].append(recall)

        pd.DataFrame(res).to_csv(os.path.join(pred_dir, 'total_res_without_coreference.csv'))

        print(f'markush zip accuracy: {calculate_mean(valid_accuracy)}, recall: {calculate_mean(valid_recall)}')
        return calculate_mean(valid_accuracy), calculate_mean(valid_recall)

    def evluate_with_coreference(self, pred_dir, evaluate_save_dir=None):
        valid_accuracy, valid_recall = [], []
        res = {'name':[],
               'accuracy':[],
               'recall':[]}
        for json_file in tqdm(self.gd_json_files):
            pred_json_path = os.path.join(pred_dir, json_file)
            gd_json_path = os.path.join(self.gd_dir, json_file)
            accuracy, recall = self._evaluate_json_with_coreference(pred_json_path, gd_json_path, evaluate_save_dir)
            if accuracy != -1:
                valid_accuracy.append(accuracy)
            if recall != -1:
                valid_recall.append(recall)

            res['name'].append(json_file)
            res['accuracy'].append(accuracy)
            res['recall'].append(recall)

        pd.DataFrame(res).to_csv(os.path.join(pred_dir, 'total_res.csv'))
        print(f'markush zip and coreference accuracy: {calculate_mean(valid_accuracy)}, recall: {calculate_mean(valid_recall)}')
        return calculate_mean(valid_accuracy), calculate_mean(valid_recall)

    def _evaluate_json_without_coreference(self, pred_json_path, gd_json_path, evaluate_save_dir=None):
        # print(pred_json_path)
        # print(gd_json_path)

        with open(pred_json_path, 'r') as f:
            origin_pred_json_data = json.load(f)

        try:
            origin_pred_json_data = origin_pred_json_data[0]['json']
        except:
            with open(pred_json_path, 'r') as f:
                origin_pred_json_data = json.load(f)

        with open(gd_json_path, 'r') as f:
            origin_gd_json_data = json.load(f)

        if len(origin_gd_json_data) == 0:
            accuracy, recall = -1, -1
            return accuracy, recall
        
        pred_json_data = self._process_without_coference(origin_pred_json_data)
        gd_json_data = self._process_without_coference(origin_gd_json_data)

        if len(pred_json_data) == 0:
            accuracy, recall = 0, 0
            return accuracy, recall
        
        accuracy_num = 0
        for pred_json_data_item in pred_json_data:
            if pred_json_data_item in gd_json_data:
                accuracy_num += 1
        accuracy = accuracy_num / len(pred_json_data)

        recall_num = 0
        for gd_json_data_item in gd_json_data:
            if gd_json_data_item in pred_json_data:
                recall_num += 1
        recall = recall_num / len(gd_json_data)


        return accuracy, recall

    def _evaluate_json_with_coreference(self, pred_json_path, gd_json_path, evaluate_save_dir=None):
        # print(pred_json_path)
        # print(gd_json_path)
        with open(pred_json_path, 'r') as f:
            origin_pred_json_data = json.load(f)

        try:
            origin_pred_json_data = origin_pred_json_data[0]['data']
        except:
            with open(pred_json_path, 'r') as f:
                origin_pred_json_data = json.load(f)

        with open(gd_json_path, 'r') as f:
            origin_gd_json_data = json.load(f)

        unique_pred_data_num = len(origin_pred_json_data)
        unique_gd_data_num = len(origin_gd_json_data)

        if len(origin_gd_json_data) == 0 :
            accuracy, recall = -1, -1
            return accuracy, recall

        if len(origin_pred_json_data) == 0:
            accuracy, recall = 0, 0
            return accuracy, recall
        
        pred_json_data, pred_json_data_with_index = self._process_with_coreferenecs(origin_pred_json_data)
        gd_json_data, gd_json_data_with_index = self._process_with_coreferenecs(origin_gd_json_data)

        accuracy_index, recall_index = [], []
        for pred_json_data_item in pred_json_data_with_index:
            data, index = pred_json_data_item
            if data in gd_json_data:
                accuracy_index.append(index)
        
        for gd_json_data_item in gd_json_data_with_index:
            data, index = gd_json_data_item
            if data in pred_json_data:
                recall_index.append(index)

        accuracy = len(set(accuracy_index)) / unique_pred_data_num
        recall = len(set(recall_index)) / unique_gd_data_num

        unique_recall_index = list(set(recall_index))
        unique_accuracy_index = list(set(accuracy_index))

        recall_failed_index = []
        recall_failed_data = []
        for i in range(len(origin_gd_json_data)):
            if i not in unique_recall_index:
                recall_failed_index.append(i)
                recall_failed_data.append(origin_gd_json_data[i])

        accuracy_failed_index = []
        accuracy_failed_data = []
        for i in range(len(origin_pred_json_data)):
            if i not in unique_accuracy_index:
                accuracy_failed_index.append(i)
                accuracy_failed_data.append(origin_pred_json_data[i])

        file_name = os.path.basename(pred_json_path).split('.')[0]

        recall_failed_data_json = json.dumps(recall_failed_data, indent=4)
        accuracy_failed_data_json = json.dumps(accuracy_failed_data, indent=4)

        with open(os.path.join(evaluate_save_dir, f'{file_name}_recall_failed.json'), 'w') as f:
            f.write(recall_failed_data_json)
        with open(os.path.join(evaluate_save_dir, f'{file_name}_accuracy_failed.json'), 'w') as f:
            f.write(accuracy_failed_data_json)

        return accuracy, recall

    def _process_without_coference(self, json_data):
        data_list = []
        if isinstance(json_data, dict):
            json_data = [json_data]
        for index, json_item in enumerate(json_data):
            if 'scaffold' not in json_item.keys() or "R-group" not in json_item.keys():
                continue
            scaffold = json_item['scaffold']
            substitues_dict = json_item['R-group']
            substitues_list = self._convert_substitues_dict_to_list_for_evaluate(substitues_dict)

            data_list.append((scaffold, substitues_list))

        return data_list

    def _process_with_coreferenecs(self, json_data):
        if isinstance(json_data, dict):
            json_data = [json_data]
        data_list, data_with_index_list = [], []
        for index, json_item in enumerate(json_data):
            if 'scaffold' not in json_item.keys() or 'identifier' not in json_item.keys() or "R-group" not in json_item.keys():
                continue
            scaffold = json_item['scaffold']
            substitues_dict = json_item['R-group']
            substitues_list = self._convert_substitues_dict_to_list_for_evaluate(substitues_dict)
            ideni = json_item['identifier']
            if ideni is None or not isinstance(ideni, str):
                continue
            idenis = ideni.split(';')
            for ideni in idenis:
                data_list.append((scaffold, substitues_list, ideni.replace(' ', '').lower()))
                data_with_index_list.append(((scaffold, substitues_list, ideni.replace(' ', '').lower()), index))
                
        return data_list, data_with_index_list

    def _convert_substitues_dict_to_list_for_evaluate(self, substitues_dict):
        if not isinstance(substitues_dict, dict):
            return []
        keys = []
        for key in substitues_dict.keys():
            keys.append(key)
        keys.sort()

        substitues_list = []
        for key in keys:
            substitues_list.append((key, substitues_dict[key]))

        return substitues_list

class FullMoleculeCoreferenceEvaluator(object):
    def __init__(self, gd_dir):
        self.gd_dir = gd_dir
        self.gd_json_files = [file for file in os.listdir(gd_dir) if file.endswith('.json') and not file.endswith('.json.csv.json')]
        self.gd_json_files.sort()
        self.too_many_molecules_num = 0

    def evaluate(self, pred_dir):
        
        valid_accuracy, valid_recall = [], []
        res = {'name':[],
               'accuracy':[],
               'recall':[]}
        for json_file in tqdm(self.gd_json_files):
            pred_json_path = os.path.join(pred_dir, json_file)
            gd_json_path = os.path.join(self.gd_dir, json_file)
            accuracy, recall = self._evaluate_json_with_coreference(pred_json_path, gd_json_path, pred_dir)
            if accuracy != -1:
                valid_accuracy.append(accuracy)
            if recall != -1:
                valid_recall.append(recall)
            res['name'].append(json_file)
            res['accuracy'].append(accuracy)
            res['recall'].append(recall)

        pd.DataFrame(res).to_csv(os.path.join(pred_dir, 'total_res.csv'))
        print(f'full molecule coreference accuracy: {calculate_mean(valid_accuracy)}, recall: {calculate_mean(valid_recall)}')
        print(f'too many molecules num : {self.too_many_molecules_num}')
        return calculate_mean(valid_accuracy), calculate_mean(valid_recall)
    
    def _evaluate_json_with_coreference(self, pred_json_path, gd_json_path, evaluate_save_dir=None):
        # print(pred_json_path)
        with open(pred_json_path, 'r') as f:
            origin_pred_json_data = json.load(f)

        try:
            origin_pred_json_data = origin_pred_json_data[0]['json']
        except:
            with open(pred_json_path, 'r') as f:
                origin_pred_json_data = json.load(f)

        with open(gd_json_path, 'r') as f:
            origin_gd_json_data = json.load(f)

        unique_pred_data_num = len(origin_pred_json_data)
        unique_gd_data_num = len(origin_gd_json_data)

        # if unique_gd_data_num > 4:
        #     self.too_many_molecules_num += 1

        if len(origin_gd_json_data) == 0:
            accuracy, recall = -1, -1
            return accuracy, recall

        if len(origin_pred_json_data) == 0:
            accuracy, recall = 0, 0
            return accuracy, recall
        
        pred_json_data, pred_json_data_with_index = self._process_with_coreferenecs(origin_pred_json_data)
        gd_json_data, gd_json_data_with_index = self._process_with_coreferenecs(origin_gd_json_data)

        accuracy_index, recall_index = [], []
        for pred_json_data_item in pred_json_data_with_index:
            data, index = pred_json_data_item
            if data in gd_json_data:
                accuracy_index.append(index)
        
        for gd_json_data_item in gd_json_data_with_index:
            data, index = gd_json_data_item
            if data in pred_json_data:
                recall_index.append(index)

        accuracy = len(set(accuracy_index)) / unique_pred_data_num
        recall = len(set(recall_index)) / unique_gd_data_num

        unique_recall_index = list(set(recall_index))
        unique_accuracy_index = list(set(accuracy_index))

        recall_failed_index = []
        recall_failed_data = []
        for i in range(len(origin_gd_json_data)):
            if i not in unique_recall_index:
                recall_failed_index.append(i)
                recall_failed_data.append(origin_gd_json_data[i])

        accuracy_failed_index = []
        accuracy_failed_data = []
        for i in range(len(origin_pred_json_data)):
            if i not in unique_accuracy_index:
                accuracy_failed_index.append(i)
                accuracy_failed_data.append(origin_pred_json_data[i])

        file_name = os.path.basename(pred_json_path).split('.')[0]

        recall_failed_data_json = json.dumps(recall_failed_data, indent=4)
        accuracy_failed_data_json = json.dumps(accuracy_failed_data, indent=4)

        with open(os.path.join(evaluate_save_dir, f'{file_name}_recall_failed.json'), 'w') as f:
            f.write(recall_failed_data_json)
        with open(os.path.join(evaluate_save_dir, f'{file_name}_accuracy_failed.json'), 'w') as f:
            f.write(accuracy_failed_data_json)

        return accuracy, recall
    
    def _process_with_coreferenecs(self, json_data):
        if isinstance(json_data, dict):
            json_data = [json_data]
        data_list, data_with_index_list = [], []
        # print(json_data)
        for index, json_item in enumerate(json_data):
            # print(json_item)
            try:
                molecule_index = json_item['index']
                ideni = json_item['identifier']
            except:
                continue
            if ideni is None or not isinstance(ideni, str):
                continue
            idenis = ideni.split(';')
            for id in idenis:
                data_list.append((molecule_index, id.replace('\n', '').replace(' ', '').lower()))
                data_with_index_list.append(((molecule_index, id.replace('\n', '').replace(' ', '').lower()), index))
                
        return data_list, data_with_index_list
    
class MoleculeDetectionEvaluator(object):
    def __init__(self, names, parse_dir, gt_bbox_json_dir):
        self.parse_dir = parse_dir
        self.names = names
        self.prepare_gt(gt_bbox_json_dir)


    def prepare_gt(self, gt_bbox_json_dir):
        gt_images = []
        gt_annotations = []
        bbox_id = 0
        self.image_size_dict = {}

        for name in tqdm(self.names):
            json_file = f'{name}.pdf.json'
            if not os.path.exists(os.path.join(gt_bbox_json_dir, json_file)):
                continue

            with open(os.path.join(gt_bbox_json_dir, json_file), 'r') as f:
                bbox_json_data = json.load(f)

            pdf_index = json_file.split('_')[0]
            pdf_name = json_file.split('.')[0]
            for bbox in bbox_json_data:
                page = bbox['page']
                image_id = int(f'{pdf_index}{page}')
                image_path = os.path.join(self.parse_dir, pdf_name, f'{pdf_name}_image_{page}.png')
                file_name = f'{pdf_name}_image_{page}.png'

                if image_id not in self.image_size_dict.keys():
                    image_origin = cv2.imread(image_path)
                    height, width = image_origin.shape[:2]
                    self.image_size_dict[image_id] = (height, width)
                    gt_images.append(
                    {
                        'id': image_id,
                        'width': width,
                        'height': height,
                        'file_name': file_name,
                    }
                    )
                else:
                    height, width = self.image_size_dict[image_id]

                category_id = 1
                new_bbox = [ bbox['bbox'][0] * width, bbox['bbox'][1] * height, bbox['bbox'][2] * width, bbox['bbox'][3] * height]
                area = new_bbox[2] * new_bbox[3]
                iscrowd = 0
                gt_annotations.append(
                    {
                        'id': bbox_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': new_bbox,
                        'iscrowd': iscrowd,
                        'area': area
                    }
                )
                bbox_id = bbox_id + 1


        gt = {
            'images': gt_images,
            'annotations': gt_annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "structure",
                    "supercategory": "struture"
                }
            ]
        }
        with open('./gt_path.json', 'w') as f:
            json.dump(gt, f)

        return

    def prepare_dt(self, dt_bbox_json_dir):
        
        dt = []

        for name in tqdm(self.names):
            json_file = f'{name}.pdf.json'
            if not os.path.exists(os.path.join(dt_bbox_json_dir, json_file)):
                continue

            with open(os.path.join(dt_bbox_json_dir, json_file), 'r') as f:
                bbox_json_data = json.load(f)

            pdf_index = json_file.split('_')[0]
            pdf_name = json_file.split('.')[0]

            for bbox in bbox_json_data:
                page = bbox['page']
                image_id = int(f'{pdf_index}{page}')

                if image_id not in self.image_size_dict.keys():
                    continue
                else:
                    height, width = self.image_size_dict[image_id]

                new_bbox = [ bbox['bbox'][0] * width, bbox['bbox'][1] * height, bbox['bbox'][2] * width, bbox['bbox'][3] * height]
                dt.append(
                    {
                        'image_id': image_id,
                        'category_id': 1,
                        'bbox': new_bbox,
                        'score': 1
                    }
                )

        with open('./dt_path.json', 'w') as f:
            json.dump(dt, f)
            
        return


    def evaluate(self, dt_bbox_json_dir):
        self.prepare_dt(dt_bbox_json_dir)

        coco_gt = COCO('./gt_path.json')

        coco_dt = coco_gt.loadRes('./dt_path.json')

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(coco_eval.stats)

        return

class OCSREvaluator(object):
    def __init__(self, df_gd_smiles):
        
        gd_smiles = df_gd_smiles['smiles'].values.tolist()
        self.gd_smiles = gd_smiles

    def evaluate(self, df_pred_smiles):
        total_num, success_num = 0, 0
        markush_num, markush_success_num = 0, 0
        chiral_num, chiral_success_num = 0, 0
        full_num, full_success_num = 0, 0
        eval_res = []

        pred_smiles = df_pred_smiles['smiles'].values.tolist()

        for gd_s, p_s in tqdm(zip(self.gd_smiles, pred_smiles)):

            if pd.isnull(gd_s) or gd_s.lower() == 'c':
                eval_res.append(None)
                continue

            total_num += 1
            if '*' in gd_s:
                markush_num += 1
            else:
                full_num += 1
            if '@' in gd_s:
                chiral_num += 1
            
            if commons.process_mol.check_smiles_match(p_s, gd_s):
                success_num += 1
                if '*' in gd_s:
                    markush_success_num += 1
                else:
                    full_success_num += 1
                if '@' in gd_s:
                    chiral_success_num += 1
                eval_res.append(True)
            else:
                eval_res.append(False)
        
        print(f'success_num / total_num: {success_num}/{total_num}={success_num/total_num}')
        print(f'success_markush_num / total_markush_num: {markush_success_num}/{markush_num}={markush_success_num/markush_num}')
        print(f'success_chiral_num / total_chiral_num: {chiral_success_num}/{chiral_num}={chiral_success_num/chiral_num}')
        print(f'success_full_num / total_full_num: {full_success_num}/{full_num}={full_success_num/full_num}')

        acc = success_num / total_num
        print('OCSR accuracy:',acc)

        return acc, eval_res
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='BioVista/config/evaluate.yaml')
    args = parser.parse_args()

    config = commons.utils.get_config_easydict(args.config_path)

    # molecule detection
    # molparser, molminer, moldetect, ours
    with open(config.file_name, 'r') as f:
        names = f.read().strip().split('\n')
    evaluator = MoleculeDetectionEvaluator(names, config.parse_dir, config.moldetect_gd)
    moldet_mAP = evaluator.evaluate(config.moldetect_pred)

    # ocsr
    # molparser, molminer, molscribe, molnexttr, ours
    df_gd_smiles = pd.read_csv(config.ocsr_gd)
    df_pred_smiles = pd.read_csv(config.ocsr_pred)
    pdf_names = list(set(df_gd_smiles['pdf_name'].values.tolist()))
    df_pred_smiles = df_pred_smiles[df_pred_smiles['pdf_name'].isin(pdf_names)]

    evaluator = OCSREvaluator(df_gd_smiles)
    ocsr_acc, eval_res = evaluator.evaluate(df_pred_smiles)
    df_pred_smiles['eval_res'] = eval_res
    df_pred_smiles.to_csv(os.path.join(os.path.dirname(config.ocsr_pred), os.path.basename(config.ocsr_pred).split('.')[0] + '_eval.csv'))

    # full coreference
    evaluator = FullMoleculeCoreferenceEvaluator(config.coreference_gd)
    evaluator.evaluate(config.coreference_pred)

    # # markush zip and coreference
    evaluator = MarkushZipEvaluator(config.markush_gd)
    evaluator.evaluate(config.markush_pred)