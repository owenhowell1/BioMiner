# BioVista

BioVista consists of six evaluation tasks, including two end-to-end tasks and four component-level tasks. 
Here we detail the information on these six evaluation tasks.

## Statistics and Access of BioVista

| **Tasks**    | **Metrics** | **Input** | **Ground-truth Labels** | **Download** |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
|  Bioactivity Triplet Extraction | F1, Precision, Recall | 500 Papers | 16,457 bioactivity | [Link](https://drive.google.com/file/d/1xITA_Mub9u1MCZjlgs26qL_dMCrzS9Rn/view?usp=drive_link) | 
|  Structure-Bioactivity Annotation | Recall@N | 500 structure-paper pairs |  500 structure-bioactivity pairs | [Link](https://drive.google.com/file/d/1cLC3NePtxctIurnCVaNLw0JzBWYrWTVW/view?usp=drive_link) | 
|  Molecule Detection | Average Precision | 500 Papers | 11,212 boundary boxes  | [Link](https://drive.google.com/file/d/1SvsdgrizDqg5V-MGXEPGMmu4JmmhiFae/view?usp=drive_link) | 
|  OCSR | Accuracy | 8,861 2D molecule structure depictions | 8,861 SMILES  | [Link](https://drive.google.com/file/d/11iCH0j7U9iFgwXahjkWZouNyqATfEPoL/view?usp=drive_link) | 
|  Full Structure Coreference Recognition | F1, Precision, Recall | 962 Augmented Images | 5,105 full structure-coreference pairs | [Link](https://drive.google.com/file/d/13UXEEV3lSEu-H5ZHtuDP3FT0kkYqmCpp/view?usp=drive_link) | 
|  Markush Enumeration | F1, Precision, Recall | 355 Augmented Images | 3,513 Markush Scaffold-R Group-Coreference Pairs | [Link](https://drive.google.com/file/d/1UAVxjqtB9HL5uUF3HqYfsvGsYFK3-nFk/view?usp=drive_link) | 


## Prepare of BioVista Input
Due to the problem of copyright, some input of BioVista can be provided.
After downloading the data provided above and zipping them in the directory `./BioVista`, some data should be further prepared. Here, we provided the instruction to prepare these input:

- **PDFs**: Among the 6 tasks, bioactivity triplet extraction, structure-bioactivity annotation, and molecule dection require the pdf input. We provide the 
Put the downloaded pdfs in directory you want, for example `example/pdfs`.
- **Augmented Images**: Running the following codes to generate augmented images for Markush enumeration and full structure coreference recognition

```
python3 BioVista/generate_augmented_images.py --pdf_dir=example/pdfs

```


## Description of BioVista
### End-to-end Tasks
Two end-to-end evaluation tasks, **protein-ligand bioactivity extraction** and **structure-bioactivity annotation**, are designed to measure the overall extraction ability of protein-ligand bioactivity extraction methods:

- **Protein-ligand Bioactivity Extraction**: Given a paper, extracting all protein-ligand bioactivity triplet data. 
- **Structure-bioactivity Annotation**: Given a PDB structure and corresponding paper, annotating the structure with reported bioactivity data in the paper. 

### Component-level Tasks
Four component-level tasks, **molecule detection**, **OCSR**, **explicit full structure corefernece recognition**, and **Markush enumeration**, are specially designed to provide a deep analysis of the core chemical structure extraction agent of BioMiner:

- **Molecule Detection**: Given a paper, detecting boundary boxes of all 2D molecule depictions.
- **Optical Chemical**: Structure Recognition: Given a 2D molecule depiction, converting the molecule image into 1D molecule SMILES.
- **Explicit Full Structure Coreference Recognition**: Given a DSM-augmented image with explicit full structures, recognizing structures' coreference within the image. 
- **Markush Enumeration**:  Given a DSM-augmented image containing Markush structures, enumerating all Markush full structures and coreferences within the image. 

