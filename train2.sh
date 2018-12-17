export CUDA_VISIBLE_DEVICES=1
python train.py "/my/Datasets/Speech/VCTK-Corpus" --processes 2 --max-steps 256000 --random-seed 0 --device /gpu:1 --batch-size 12 --postfix 46

exit
