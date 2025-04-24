pip3 install google-generativeai
pip3 install joblib
pip3 install beautifulsoup4
pip3 install tqdm
pip3 install rdkit-pypi
pip3 install PyPDF2
conda install -y tabula-py
pip3 install pdf2image
pip3 install JPype1
pip3 install openai
pip3 install pubchempy
conda install -y poppler poppler-qt
pip3 install opencv-python==4.5.5.64
pip3 install matplotlib
pip3 install scikit-image
pip3 install tensorboardX
pip3 install timm==1.0.13
pip3 install easyocr
pip3 install pycocotools
pip3 install transformers
pip3 install huggingface-hub
pip3 install SmilesPE
pip3 install OpenNMT-py==2.2.0
pip3 install lmdb Bio easydict prody biopython traitlets
pip3 install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
magic-pdf --version
pip3 install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python3 download_models.py
python3 -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
wget https://github.com/opendatalab/MinerU/raw/master/demo/pdfs/small_ocr.pdf
magic-pdf -p small_ocr.pdf -o ./output
pip3 install fastapi
pip3 install litserve
pip3 install filetype
pip3 install rcsbsearchapi
pip3 install gpustat
conda install -y -c dglteam dgl-cuda11.7
conda install -y -c conda-forge biopandas
conda install -y -c anaconda yaml
conda install -y -c conda-forge pyyaml
conda install -y -c conda-forge psutil
conda install -y -c anaconda scikit-learn
conda install -y -c conda-forge openbabel
