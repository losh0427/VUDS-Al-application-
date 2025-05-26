pip install --upgrade pip
apt-get update && apt-get install -y libgl1
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
apt-get update && apt-get install -y poppler-utils
