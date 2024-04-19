#1. setup
#1-1. base_setup.py 실행

import os
import sys
import subprocess

weights_url = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
weights_path = os.getcwd() + '/GroundingDINO/weights/groundingdino_swint_ogc.pth'

def install_package(package):
    try:
        print(f"##Install attempt == {package}##")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"##Install Success == {package}##")
    except ImportError:
        print(f"##Install ERROR == {package}##")

install_package('GitPython')
install_package('wget')
install_package('wheel')

import git
from git import Repo
try:
    print("##Git_Clone attempt##")
    Repo.clone_from('https://github.com/IDEA-Research/GroundingDINO.git','GroundingDINO')
except:
    print("##Git_Clone Fail##")
    a = 0

def mkdir():
    try:
        os.mkdir('GroundingDINO/weights')
        os.mkdir('GroundingDINO/data')
        os.mkdir('GroundingDINO/result_data')
    except:
        print("##mkdir Fail##")
        
def install_weights():
    try:
        import wget
        print("##Download attempt weights##")
        wget.download(weights_url, weights_path)
        print("##Download Success!##")
    except ImportError:
        print("##Download ERROR##")

def input_gitclone():
    try:
        print("##Download attempt testdata##")
        Repo.clone_from('https://github.com/FsRoot/obj_detecting_testing.git', os.getcwd() + '/groundingDINO/data')
        print("##Download Success!##")
    except ImportError:
        return              #이미 존재시 자체 오류코드 출력, data폴더 삭제 후 재실행

def install_torch_cu():
    print("##Install attempt torch##")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
        print("##torch Install Success##")
    except ImportError:
        print("##torch Install ERROR##")

mkdir() #폴더 생성
install_weights()   #가중치 다운
input_gitclone()    #테스트데이터 다운
install_torch_cu()  #torch를 쿠다 환경으로 실행

print(os.getcwd())
os.chdir(os.getcwd() + '/GroundingDINO')
print(os.getcwd())
subprocess.check_call([sys.executable, "-m", "pip", "install", '-e', '.'])


