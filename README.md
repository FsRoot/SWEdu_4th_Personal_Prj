# SWEdu_4th_Personal_Prj
SW_Education_4th_Personal_Prj

## This project is a personal project of the 4th SW training program
source project : https://github.com/IDEA-Research/GroundingDINO

## checklist 4 goal

Object recognition by type  
Object path output  
~~Tracking with selected specific object~~

## Confirmation required before starting
check!
Use long paths in Windows 10 version 1607 and later :
https://learn.microsoft.com/ko-kr/windows/win32/fileio/maximum-file-path-limitation?tabs=registry


Required : c++, gpu driver(recommend), CUDA toolkit

check require cuda version - https://pytorch.org/get-started/locally/  
nvidia gpu driver install- https://www.nvidia.com/Download/index.aspx?lang=kr  
cuda toolki install - https://developer.nvidia.com/cuda-toolkit-archive  
cuddn install - https://developer.nvidia.com/rdp/cudnn-archive  

check ur os environment variables

PATH in user
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\include  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\lib  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\bin  

CUDA_PATH in system variable
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}

CUDA_PATH_{version}
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}

## Start!

1. git clone : 
https://github.com/FsRoot/SWEdu_4th_Personal_Prj.git

2. run 1.base_setup.py
   check finished run base_setup  
addict                 2.4.0  
certifi                2024.2.2  
charset-normalizer     3.3.2  
colorama               0.4.6  
contourpy              1.2.1  
cycler                 0.12.1  
defusedxml             0.7.1  
filelock               3.9.0  
fonttools              4.51.0  
fsspec                 2024.3.1  
gitdb                  4.0.11  
GitPython              3.1.43  
groundingdino          0.1.0        {root}\{project_name}\GroundingDINO  
huggingface-hub        0.22.2  
idna                   3.7  
importlib_metadata     7.1.0  
Jinja2                 3.1.2  
kiwisolver             1.4.5  
MarkupSafe             2.1.3  
matplotlib             3.8.4  
mpmath                 1.3.0  
networkx               3.2.1  
numpy                  1.26.3  
opencv-python          4.9.0.80  
opencv-python-headless 4.9.0.80  
packaging              24.0  
pillow                 10.2.0  
pip                    23.2.1  
platformdirs           4.2.0  
pycocotools            2.0.7  
pyparsing              3.1.2  
python-dateutil        2.9.0.post0  
PyYAML                 6.0.1  
regex                  2024.4.16  
requests               2.31.0  
safetensors            0.4.3  
scipy                  1.13.0  
setuptools             68.2.0  
six                    1.16.0  
smmap                  5.0.1  
supervision            0.19.0  
sympy                  1.12  
timm                   0.9.16  
tokenizers             0.19.1  
tomli                  2.0.1  
torch                  2.2.2+{pythoch cuda or cpu version}  
torchaudio             2.2.2+{pythoch cuda or cpu version}  
torchvision            0.17.2+{pythoch cuda or cpu version}  
tqdm                   4.66.2  
transformers           4.40.0  
typing_extensions      4.8.0  
urllib3                2.2.1  
wget                   3.2  
wheel                  0.41.2  
yapf                   0.40.2  
zipp                   3.18.1  

3. run 2.run_prj
   
   and check 4 runing!