import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import json
from tqdm import tqdm
from BioMiner.commons.utils import pmap_multi
from BioMiner.commons.process_pdf import pdf_load_pypdf_images, image_segment_given_box_xywh, load_pdf_pages_contain_tables_and_figures
from BioMiner.commons.mineru_pdf import run_mineru, get_mineru_table_body_bbox, get_mineru_complete_figure_table_bbox
from BioMiner.commons.mol_detection import run_yolo_batch, load_md_external_res, merge_full_page_and_seg_table_bbox
from BioMiner.commons.ocsr import visualize_all_box, run_molscribe_batch, load_ocsr_external_res, prepare_full_markush_process
from BioMiner.dataset.utils import load_prompts, bioactivity_prompt_strategy, structure_prompt_strategy
from BioMiner.runner.extractor import extraction_ligand_structure_step, extraction_bioactivity_step, save_overall_result, extract_markush_part_with_bbox_index_split_complex_image_seg_layout
from BioMiner.runner.mllm import get_api_client
from BioMiner.runner.metric_fn import resulter, evaluate_step

def process_single_pdf_chemical_structure_mllm(
    name, bboxes_with_pred_smiles, mineru_complete_figure_table_bbox,
    page_image_path, augment_full_markush_path, extraction_result_path, vision_mllm_type, base_url, api_key,
    structure_full_prompt, structure_part_prompt, structure_full_suffix, structure_part_suffix,
    structure_cot, split_bbox_num, segment_image, enlarge_size,
    structure_full_layout_seg, structure_part_layout_seg, bbox_background,
    overwrite_structure_full, overwrite_structure_part
):
    augmented_full_image_paths, augmented_part_image_paths, index_smiles_dict, image2bboxindex = prepare_full_markush_process(name, bboxes_with_pred_smiles, page_image_path, augment_full_markush_path)

    vision_mllm_client = get_api_client(base_url, api_key)

    pdf_augmented_full_image_bbox_index_dict=[image2bboxindex[os.path.basename(image_path)] for image_path in augmented_full_image_paths]
    pdf_augmented_part_image_bbox_index_dict=[image2bboxindex[os.path.basename(image_path)] for image_path in augmented_part_image_paths]

    # print(mineru_complete_figure_table_bbox)
    full_page_figure_table_layout_bbox, part_page_figure_table_layout_bbox = [], []
    for image_path in augmented_full_image_paths:
        file_name = os.path.basename(image_path)
        file_name_items = file_name.split('.')[0].split('_')
        page_idx = int(file_name_items[-1])
        try:
            full_page_figure_table_layout_bbox.append(mineru_complete_figure_table_bbox[page_idx])
        except:
            full_page_figure_table_layout_bbox.append(None)

    for image_path in augmented_part_image_paths:
        file_name = os.path.basename(image_path)
        file_name_items = file_name.split('.')[0].split('_')
        page_idx = int(file_name_items[-1])
    
        try:
            part_page_figure_table_layout_bbox.append(mineru_complete_figure_table_bbox[page_idx])
        except:
            part_page_figure_table_layout_bbox.append(None)

    os.makedirs(os.path.join(extraction_result_path, name), exist_ok=True)
    structure_data_list = extraction_ligand_structure_step(name, 
                                                            vision_mllm_type, 
                                                            vision_mllm_client,
                                                            augmented_full_image_paths,
                                                            augmented_part_image_paths, 
                                                            index_smiles_dict, 
                                                            pdf_augmented_full_image_bbox_index_dict,
                                                            pdf_augmented_part_image_bbox_index_dict, 
                                                            full_page_figure_table_layout_bbox, 
                                                            part_page_figure_table_layout_bbox,
                                                            structure_full_prompt, 
                                                            structure_part_prompt,
                                                            extraction_result_path,
                                                            structure_full_suffix,
                                                            structure_part_suffix,
                                                            structure_cot, 
                                                            split_bbox_num, 
                                                            segment_image, 
                                                            enlarge_size, 
                                                            structure_full_layout_seg, 
                                                            structure_part_layout_seg, 
                                                            bbox_background,
                                                            overwrite_structure_full, 
                                                            overwrite_structure_part)
            
    return structure_data_list


def process_single_bioactivity_measurement(
    name, pdf_text, pdf_image_paths,
    text_mllm_type, vision_mllm_type, base_url, api_key,
    bioactivity_text_prompt, bioactivity_image_prompt,
    merge_strategy, extraction_result_path,
    bioactivity_text_suffix, bioactivity_image_suffix,
    bioactivity_cot, overwrite_bioactivity_text, overwrite_bioactivity_image):

    text_mllm_client = get_api_client(base_url, api_key)
    vision_mllm_client = get_api_client(base_url, api_key)

    os.makedirs(os.path.join(extraction_result_path, name), exist_ok=True)
    bioactivity_data_lists = extraction_bioactivity_step(name, 
                                                        text_mllm_type, 
                                                        text_mllm_client, 
                                                        vision_mllm_type,
                                                        vision_mllm_client,
                                                        pdf_text,
                                                        pdf_image_paths, 
                                                        bioactivity_text_prompt,
                                                        bioactivity_image_prompt, 
                                                        merge_strategy,
                                                        extraction_result_path, 
                                                        extraction_result_path,
                                                        bioactivity_text_suffix,
                                                        bioactivity_image_suffix,
                                                        bioactivity_cot,
                                                        overwrite_bioactivity_text,
                                                        overwrite_bioactivity_image)
            
    return bioactivity_data_lists

def process_mineru_single(name, mineru_layout, pdf_path, page_image_path):
    table_body_bbox = get_mineru_table_body_bbox(mineru_layout)
    complete_figure_table_bbox = get_mineru_complete_figure_table_bbox(mineru_layout)
    pages_images_contain_tables_and_figures_paths = load_pdf_pages_contain_tables_and_figures(name, mineru_layout, pdf_path, page_image_path)

    return table_body_bbox, complete_figure_table_bbox, pages_images_contain_tables_and_figures_paths

class BioMiner():
    def __init__(self, config, biovista_evaluate=False, external_full_md_res_dir=None, external_ocsr_res_dir=None):
        self.init_biominer(config)
        if biovista_evaluate:
            self.init_biovista(config)
        self.biovista_evaluate = biovista_evaluate
        self.external_full_md_res_dir = external_full_md_res_dir
        self.external_ocsr_res_dir = external_ocsr_res_dir

    def init_pred_dir(self, pdf_paths):
        self.page_image_path = os.path.join(self.output_dir, 'page_images')
        self.mineru_path = os.path.join(self.output_dir, 'mineru')
        self.full_page_detection_path = os.path.join(self.output_dir, f'md_full_{self.mol_detection_model}')
        self.seg_table_detection_path = os.path.join(self.output_dir, 'md_part_yolo')
        self.merge_detection_path = os.path.join(self.output_dir, f'md_merge_{self.mol_detection_model}_yolo',)
        self.augment_full_markush_path = os.path.join(self.output_dir, f'md_merge_{self.mol_detection_model}_yolo_distinguish_full_markush_after_ocsr')
        self.extraction_result_path = os.path.join(self.output_dir, 'extraction_gemini-2.0-flash')
        self.stage_one_result_path = os.path.join(self.output_dir, 'stage_one.pkl')

        names = []
        for pdf_path in pdf_paths:
            name = os.path.basename(pdf_path).split('.')[0]
            names.append(name)

        return names

    def init_biominer(self, config):
        self.n_jobs = config.n_jobs
        self.device = config.device
        self.output_dir = config.output_dir
        self.overwrite_structure_full = config.overwrite_structure_full
        self.overwrite_structure_part = config.overwrite_structure_part
        self.overwrite_bioactivity_text = config.overwrite_bioactivity_text
        self.overwrite_bioactivity_image = config.overwrite_bioactivity_image

        self.text_mllm_type = config.model.text_mllm_type
        self.vision_mllm_type = config.model.vision_mllm_type
        self.base_url = config.model.base_url
        self.api_key = config.model.api_key

        self.mol_detection_model = config.model.mol_detection_model
        self.ocsr_model = config.model.ocsr_model

        self.structure_cot = config.model.structure_cot
        self.split_bbox_num = config.model.structure_full_split_molecule_num
        self.segment_image = config.model.structure_full_seg_image
        self.enlarge_size = config.model.structure_full_seg_enlarge
        self.structure_full_layout_seg = config.model.structure_full_layout_seg
        self.structure_part_layout_seg = config.model.structure_part_layout_seg
        self.bbox_background = config.model.structure_index_bbox_background
        self.merge_strategy = config.model.merge_strategy
        self.bioactivity_cot = config.model.bioactivity_cot

        bioactivity_text_prompt, bioactivity_image_prompt, structure_full_prompt, structure_part_prompt, merge_prompt = load_prompts()
        self.bioactivity_text_prompt = bioactivity_text_prompt
        self.bioactivity_image_prompt = bioactivity_image_prompt
        self.structure_full_prompt = structure_full_prompt
        self.structure_part_prompt = structure_part_prompt
        self.merge_prompt = merge_prompt

        bioactivity_text_suffix = bioactivity_prompt_strategy[config.model.bioactivity_text_strategy]
        bioactivity_image_suffix = bioactivity_prompt_strategy[config.model.bioactivity_image_strategy]

        if config.data.parse_text_method == 'mineru':
            bioactivity_text_suffix += '_txtmu'
            if config.data.no_text_table:
                bioactivity_text_suffix += '_ntb'
        if config.data.parse_image_method == 'mineru_contain':
            bioactivity_image_suffix += '_imgmuc'
        elif config.data.parse_image_method == 'mineru_seg':
            bioactivity_image_suffix += '_imgmus'

        structure_full_suffix = f'{config.model.mol_detection_model}_bbox_{config.model.ocsr_model}_ocsr' + '_' + structure_prompt_strategy[config.model.structure_full_strategy]
        structure_part_suffix = f'{config.model.mol_detection_model}_bbox_{config.model.ocsr_model}_ocsr' + '_' + structure_prompt_strategy[config.model.structure_part_strategy]
        
        if config.model.structure_cot:
            structure_full_suffix += '_cot'
            structure_part_suffix += '_cot'
        
        if config.model.structure_full_split_molecule_num is not None:
            structure_full_suffix += f'_split{config.model.structure_full_split_molecule_num}'

        if config.model.structure_full_seg_image:
            structure_full_suffix += f'seg{config.model.structure_full_seg_enlarge}'
  
        if config.model.structure_index_bbox_background:
            structure_full_suffix += f'_bg'
            structure_part_suffix += f'_bg'

        if config.model.structure_full_layout_seg:
            structure_full_suffix += f'_lo'

        if config.model.structure_part_layout_seg:
            structure_part_suffix += f'_lo'

        self.structure_full_suffix = structure_full_suffix
        self.structure_part_suffix = structure_part_suffix
        self.bioactivity_text_suffix = bioactivity_text_suffix
        self.bioactivity_image_suffix = bioactivity_image_suffix

        return 

    def init_biovista(self, config):
        self.top_n = config.test.top_n
        self.pdb_name_path = config.test.pdb_name_path
        self.pdb_label_path = config.test.pdb_label_path
        self.pdb_structure_path = config.test.pdb_structure_path
        self.labels_base_dir = config.test.labels_base_dir
        self.dataset_name = config.test.dataset_name

        return

    def agent_preprocess(self, names, pdf_paths):
        mineru_layouts, mineru_texts = [], []

        # convert pdf pages to images
        page_image_pathss = pmap_multi(pdf_load_pypdf_images, 
                                      zip(names, pdf_paths),
                                      save_path=self.page_image_path,
                                      n_jobs=self.n_jobs,
                                      desc='converting pdf pages to images')
        
        # MinerU layout analysis and reading order determination
        for name, pdf_path in tqdm(zip(names, pdf_paths), desc='run mineru'):
            mineru_layout, mineru_text = run_mineru(name, pdf_path, self.mineru_path, self.device)
            mineru_layouts.append(mineru_layout)
            mineru_texts.append(mineru_text)

        # process mineru layout
        res = pmap_multi(process_mineru_single,
                   zip(names, mineru_layouts, pdf_paths),
                   page_image_path=self.page_image_path,
                   n_jobs=self.n_jobs,
                   desc='process mineru layout')
        mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss = map(list, zip(*res))

        return page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss
    
    def agent_chemical_structure_ocsr_and_md(self, names, page_image_pathss, mineru_table_body_bboxs):
        bboxes_with_pred_smiless = []
        for name, page_image_paths, mineru_table_body_bbox in zip(names, page_image_pathss, mineru_table_body_bboxs):
            # Molecule Detection
            ## you can use any other molecule detection methods here
            ## i) full page detection
            ### molminer - best performance
            if self.mol_detection_model == 'external':
                full_page_bboxes = load_md_external_res(os.path.join(self.external_full_md_res_dir, f'{name}.json'))
            ## yolo
            else:
                full_bboxes_list = run_yolo_batch(name, page_image_paths, self.full_page_detection_path, self.device)
                full_page_bboxes = []
                for page, bboxes in enumerate(full_bboxes_list):
                    full_page_bboxes.extend([{'page':page, 'bbox':bbox} for bbox in bboxes])

            # print('full page molecule bbox:',full_page_bboxes)
            ## ii) segmented table detection
            ### yolo
            segmented_table_pages, segmented_table_paths, segmented_table_layouts = [], [], []
            for page in mineru_table_body_bbox.keys():
                page_table_body_bboxs = mineru_table_body_bbox[page]
                origin_image_path = os.path.join(self.page_image_path, name, f'{name}_image_{page}.png')
                for bbox_idx, bbox in enumerate(page_table_body_bboxs):
                    segmented_table_path = os.path.join(self.page_image_path, name, f'{name}_image_{page}_tb_{bbox_idx}.png')
                    x, y, w, h = bbox
                    image_segment_given_box_xywh(origin_image_path, segmented_table_path, x, y, w, h)
                    segmented_table_paths.append(segmented_table_path)
                    segmented_table_layouts.append(bbox)
                    segmented_table_pages.append(page)

            seg_table_bboxes_list = run_yolo_batch(name, segmented_table_paths, self.seg_table_detection_path, self.device)
            seg_table_bboxes = []
            for page, bboxes, layout in zip(segmented_table_pages, seg_table_bboxes_list, segmented_table_layouts):
                seg_table_bboxes.append({'page':page, 'tb_layout_bbox':layout, 'bboxes':bboxes})

            ## iii) merge full page and segmented table 
            merge_pdf_bboxes = merge_full_page_and_seg_table_bbox(full_page_bboxes, seg_table_bboxes)
            # OCSR
            all_segmented_box_paths = visualize_all_box(name, merge_pdf_bboxes, self.page_image_path, self.merge_detection_path)
            ## you can use any other OCSR methods here
            ## molparser - best performance
            if self.ocsr_model == 'external':
                bboxes_with_pred_smiles = load_ocsr_external_res(os.path.join(self.external_ocsr_res_dir, f'{name}.json'))
            ## molscirbe
            else:
                bboxes_with_pred_smiles = run_molscribe_batch(merge_pdf_bboxes, all_segmented_box_paths, self.device)

            bboxes_with_pred_smiless.append(bboxes_with_pred_smiles)

        return bboxes_with_pred_smiless
    
    def agent_chemical_structure_mllm_inference(self, names, bboxes_with_pred_smiless, mineru_complete_figure_table_bboxs):

        args = [
                (
                    name,
                    bboxes_with_pred_smiles,
                    mineru_complete_figure_table_bbox,
                    self.page_image_path,
                    self.augment_full_markush_path,
                    self.extraction_result_path,
                    self.vision_mllm_type,
                    self.base_url,
                    self.api_key,
                    self.structure_full_prompt,
                    self.structure_part_prompt,
                    self.structure_full_suffix,
                    self.structure_part_suffix,
                    self.structure_cot,
                    self.split_bbox_num,
                    self.segment_image,
                    self.enlarge_size,
                    self.structure_full_layout_seg,
                    self.structure_part_layout_seg,
                    self.bbox_background,
                    self.overwrite_structure_full,
                    self.overwrite_structure_part
                )
                for name, bboxes_with_pred_smiles, mineru_complete_figure_table_bbox
                in zip(names, bboxes_with_pred_smiless, mineru_complete_figure_table_bboxs)
            ]

        # Run in parallel using pmap_multi
        structure_data_lists = pmap_multi(
            process_single_pdf_chemical_structure_mllm,
            args,
            n_jobs=self.n_jobs,
            desc='Processing chemical structure MLLM inference'
        )
            
        return structure_data_lists

    def agent_chemical_structure(self, names, page_image_pathss, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs):

        if self.external_ocsr_res_dir is None:
            bboxes_with_pred_smiless = self.agent_chemical_structure_ocsr_and_md(names, page_image_pathss, mineru_table_body_bboxs)
        else:
            bboxes_with_pred_smiless = []
            for name in names:
                bboxes_with_pred_smiles = load_ocsr_external_res(os.path.join(self.external_ocsr_res_dir, f'{name}.json'))
                bboxes_with_pred_smiless.append(bboxes_with_pred_smiles)

        # bboxes_with_pred_smiless = self.agent_chemical_structure_ocsr_and_md(names, page_image_pathss, mineru_table_body_bboxs)
        extracetd_chemical_structures = self.agent_chemical_structure_mllm_inference(names, bboxes_with_pred_smiless, mineru_complete_figure_table_bboxs)

        return extracetd_chemical_structures
    
    def agent_bioactivity_measurement(self, names, pdf_texts, pdf_image_pathss):
        
        args = [
            (
                name,
                pdf_text,
                pdf_image_paths,
                self.text_mllm_type,
                self.vision_mllm_type,
                self.base_url,
                self.api_key,
                self.bioactivity_text_prompt,
                self.bioactivity_image_prompt,
                self.merge_strategy,
                self.extraction_result_path,
                self.bioactivity_text_suffix,
                self.bioactivity_image_suffix,
                self.bioactivity_cot,
                self.overwrite_bioactivity_text,
                self.overwrite_bioactivity_image
            )
            for name, pdf_text, pdf_image_paths
            in zip(names, pdf_texts, pdf_image_pathss)
        ]

        # Run in parallel using pmap_multi
        bioactivity_data_lists = pmap_multi(
            process_single_bioactivity_measurement,
            args,
            n_jobs=self.n_jobs,
            desc='Processing bioactivity measurement'
        )

        return bioactivity_data_lists
    
    def agent_postprocess(self, names, bioactivity_data_list, structure_data_list):
        df_extracted_overall_data_list = []
        for name, bioactivity_data, structure_data in tqdm(zip(names, bioactivity_data_list, structure_data_list), desc='postprocessing'):
            df_extracted_overall_data = save_overall_result(name, 
                                                            bioactivity_data, 
                                                            structure_data,
                                                            self.extraction_result_path, 
                                                            self.structure_full_suffix, 
                                                            self.structure_part_suffix)
            df_extracted_overall_data_list.append(df_extracted_overall_data)

        return df_extracted_overall_data_list
    

    def extract_pdf(self, pdf_paths):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        names = self.init_pred_dir(pdf_paths)
        page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss = self.agent_preprocess(names, pdf_paths)
        extracetd_chemical_structure_lists = self.agent_chemical_structure(names, page_image_pathss, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs)
        bioactivity_data_lists = self.agent_bioactivity_measurement(names, mineru_texts, pages_images_contain_tables_and_figures_pathss)
        df_extracted_overall_data_list = self.agent_postprocess(names, bioactivity_data_lists, extracetd_chemical_structure_lists)
        if self.biovista_evaluate:
            self.evaluate_biovista(names)
        return bioactivity_data_lists, extracetd_chemical_structure_lists, df_extracted_overall_data_list


    def opensource_stage_one(self, pdf_paths):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        names = self.init_pred_dir(pdf_paths)
        page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss = self.agent_preprocess(names, pdf_paths)
        all_segmented_box_pathss, merge_pdf_bboxess = [], []
        for name, page_image_paths, mineru_table_body_bbox in zip(names, page_image_pathss, mineru_table_body_bboxs):
            # Molecule Detection
            ## you can use any other molecule detection methods here
            ## i) full page detection
            ### molminer - best performance
            if self.mol_detection_model == 'external':
                full_page_bboxes = load_md_external_res(os.path.join(self.external_full_md_res_dir, f'{name}.json'))
            ## yolo
            else:
                full_bboxes_list = run_yolo_batch(name, page_image_paths, self.full_page_detection_path, self.device)
                full_page_bboxes = []
                for page, bboxes in enumerate(full_bboxes_list):
                    full_page_bboxes.extend([{'page':page, 'bbox':bbox} for bbox in bboxes])

            # print('full page molecule bbox:',full_page_bboxes)
            ## ii) segmented table detection
            ### yolo
            segmented_table_pages, segmented_table_paths, segmented_table_layouts = [], [], []
            for page in mineru_table_body_bbox.keys():
                page_table_body_bboxs = mineru_table_body_bbox[page]
                origin_image_path = os.path.join(self.page_image_path, name, f'{name}_image_{page}.png')
                for bbox_idx, bbox in enumerate(page_table_body_bboxs):
                    segmented_table_path = os.path.join(self.page_image_path, name, f'{name}_image_{page}_tb_{bbox_idx}.png')
                    x, y, w, h = bbox
                    image_segment_given_box_xywh(origin_image_path, segmented_table_path, x, y, w, h)
                    segmented_table_paths.append(segmented_table_path)
                    segmented_table_layouts.append(bbox)
                    segmented_table_pages.append(page)

            seg_table_bboxes_list = run_yolo_batch(name, segmented_table_paths, self.seg_table_detection_path, self.device)
            seg_table_bboxes = []
            for page, bboxes, layout in zip(segmented_table_pages, seg_table_bboxes_list, segmented_table_layouts):
                seg_table_bboxes.append({'page':page, 'tb_layout_bbox':layout, 'bboxes':bboxes})

            ## iii) merge full page and segmented table 
            merge_pdf_bboxes = merge_full_page_and_seg_table_bbox(full_page_bboxes, seg_table_bboxes)
            merge_pdf_bboxess.append(merge_pdf_bboxes)
            # OCSR
            all_segmented_box_paths = visualize_all_box(name, merge_pdf_bboxes, self.page_image_path, self.merge_detection_path)
            all_segmented_box_pathss.append(all_segmented_box_paths)
            
        with open(self.stage_one_result_path, 'wb') as f:
            pickle.dump((page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss, merge_pdf_bboxess, all_segmented_box_pathss), f)
        
        return

    def load_stage_one_result(self):
        with open(self.stage_one_result_path, 'rb') as f:
            page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss, merge_pdf_bboxess, all_segmented_box_pathss = pickle.load(f)

        return page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss, merge_pdf_bboxess, all_segmented_box_pathss

    def opensource_stage_two(self, pdf_paths):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        names = self.init_pred_dir(pdf_paths)
        page_image_pathss, mineru_texts, mineru_table_body_bboxs, mineru_complete_figure_table_bboxs, pages_images_contain_tables_and_figures_pathss, merge_pdf_bboxess, all_segmented_box_pathss = self.load_stage_one_result()
        bboxes_with_pred_smiless = []
        for name, merge_pdf_bboxes, all_segmented_box_paths in zip(names, merge_pdf_bboxess, all_segmented_box_pathss):
            if self.ocsr_model == 'external':
                bboxes_with_pred_smiles = load_ocsr_external_res(os.path.join(self.external_ocsr_res_dir, f'{name}.json'))
            ## molscirbe
            else:
                bboxes_with_pred_smiles = run_molscribe_batch(merge_pdf_bboxes, all_segmented_box_paths, self.device)

            bboxes_with_pred_smiless.append(bboxes_with_pred_smiles)

        extracetd_chemical_structure_lists = self.agent_chemical_structure_mllm_inference(names, bboxes_with_pred_smiless, mineru_complete_figure_table_bboxs)
        bioactivity_data_lists = self.agent_bioactivity_measurement(names, mineru_texts, pages_images_contain_tables_and_figures_pathss)
        df_extracted_overall_data_list = self.agent_postprocess(names, bioactivity_data_lists, extracetd_chemical_structure_lists)
        if self.biovista_evaluate:
            self.evaluate_biovista(names)
        return bioactivity_data_lists, extracetd_chemical_structure_lists, df_extracted_overall_data_list
    
    def evaluate_biovista(self, names):
        result = resulter(names, self.top_n, self.extraction_result_path, self.extraction_result_path, self.structure_full_suffix, self.structure_part_suffix)
        result.initialize_evaluating_result()

        evaluation_results = pmap_multi(evaluate_step, 
                                        zip(names),
                                        pdb_name_path=self.pdb_name_path, 
                                        pdb_label_path=self.pdb_label_path, 
                                        pdb_structure_path=self.pdb_structure_path,
                                        labels_base_dir=self.labels_base_dir, 
                                        dataset_name=self.dataset_name, 
                                        text_model_output_path=self.extraction_result_path,
                                        vision_model_output_path=self.extraction_result_path,
                                        structure_full_suffix=self.structure_full_suffix,
                                        structure_part_suffix=self.structure_part_suffix,
                                        bioactivity_text_suffix=self.bioactivity_text_suffix,
                                        bioactivity_image_suffix=self.bioactivity_image_suffix,
                                        top_n=self.top_n,
                                        n_jobs=self.n_jobs,
                                        desc='biovista evaluating ...'
                                        )
        for name, evaluation_result in zip(names, evaluation_results):
            result.update_evaluating_result(name, evaluation_result)

        result.output_evaluating_result()

        return



class BioMiner_Markush_Infernce(object):
    def __init__(self, vision_mllm_type, markush_prompt_path, 
                 image2bboxindex_path=None, layout_seg_json_path=None,
                 context_examples=None, base_url=None, api_key=None):
        with open(markush_prompt_path, 'r') as f:
            self.markush_prompt = f.read().strip()
        
        self.vision_mllm_type = vision_mllm_type
        self.base_url = base_url
        self.api_key = api_key

        self.image2bboxindex_path = image2bboxindex_path
        self.layout_seg_json_path = layout_seg_json_path
        self.context_examples = context_examples

        if image2bboxindex_path is not None:
            with open(image2bboxindex_path, 'r') as f:
                image2bboxindex = json.load(f)
            self.image2bboxindex = image2bboxindex

        if layout_seg_json_path is not None:
            with open(layout_seg_json_path, 'r') as f:
                layoutsegbbox = json.load(f)
            self.layoutsegbbox = layoutsegbbox

    def markush_zip_with_index_batch_split_image_layout(self, image_paths, split_bbox_num, segment_image, cot=False):
        
        assert self.image2bboxindex_path is not None
        assert self.layout_seg_json_path is not None
        image_bbox_dicts = []
        figure_table_layout_bboxs = []
        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            image_bbox_dicts.append(self.image2bboxindex[file_name])

            file_name_items = file_name.split('.')[0].split('_')
            pdf_idex = file_name_items[0]
            pdf_pdb_name = file_name_items[1]
            pdf_name = f'{pdf_idex}_{pdf_pdb_name}'
            page_idx = file_name_items[-1]
            
            try:
                figure_table_layout_bboxs.append(self.layoutsegbbox[pdf_name][page_idx])
            except:
                figure_table_layout_bboxs.append(None)

        data_jsons = pmap_multi(extract_markush_part_with_bbox_index_split_complex_image_seg_layout, 
                                zip(image_paths, image_bbox_dicts, figure_table_layout_bboxs),
                                markush_prompt=self.markush_prompt,
                                vision_mllm_type=self.vision_mllm_type,
                                cot=cot, split_bbox_num=split_bbox_num,
                                segment_image=segment_image,
                                base_url=self.base_url, api_key=self.api_key,
                                n_jobs=16, desc='extracting markush structures ')

        return data_jsons
