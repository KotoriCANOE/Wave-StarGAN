chcp 65001
cd /d "%~dp0"

python train.py "D:\Datasets\Speech\VCTK-Corpus" --processes 2 --max-steps 64000 --random-seed 0 --device /gpu:0 --batch-size 2 --postfix 44

pause

python train.py "F:\Datasets\NSH_APP" --processes 2 --max-steps 64000 --random-seed 0 --device /gpu:0 --batch-size 2 --postfix 41
