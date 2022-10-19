# label-fakenews-medical
labeling post crawling from facebook to category medical and fakenews

# install lib
python -m venv tmp/.venv

source tmp/.venv/bin/activate

pip install -r requirements.txt

python -m src.app.main
