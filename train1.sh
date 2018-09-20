python train.py "$HOME/Datasets/NSH_APP" --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 1

exit

python train.py "$HOME/Datasets/NSH_APP_npz" --packed --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 1
