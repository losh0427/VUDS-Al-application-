pip install --upgrade pip
apt-get update && apt-get install -y libgl1
pip3 install \
torch==2.3.1+cu118 \
torchvision==0.18.1+cu118 \
torchaudio==2.3.1+cu118 \
--index-url https://download.pytorch.org/whl/cu118
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
apt-get update && apt-get install -y poppler-utils
