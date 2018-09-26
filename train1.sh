python train.py "$HOME/Datasets/NSH_APP" --processes 1 --max-steps 44000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 4 --restore

exit

python train.py "$HOME/Datasets/NSH_APP" --processes 1 --max-steps 127000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 3
