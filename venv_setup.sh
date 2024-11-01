# rm -rf build/ dist/ py312/
python3.12 -m venv py312
source py312/bin/activate
pip install -U pip
pip install --upgrade pip
pip3 install -r app/requirements.txt
