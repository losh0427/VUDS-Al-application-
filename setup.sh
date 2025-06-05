
apt-get update

apt-get install -y python3.10 python3.10-distutils python3-pip

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

pip install --upgrade pip
apt-get update && apt-get install -y libgl1
apt-get update && apt-get install -y poppler-utils
apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

pip3 install -r requirements.txt
pip3 install \
torch==2.3.1+cu118 \
torchvision==0.18.1+cu118 \
torchaudio==2.3.1+cu118 \
--index-url https://download.pytorch.org/whl/cu118
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

