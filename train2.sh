export CUDA_VISIBLE_DEVICES=1
python train.py "/my/Datasets/NSH_APP" --discriminator-model "/my/Project/SpeakerRecognition/model153.tmp/model.pb" --processes 4 --max-steps 256000 --random-seed 0 --device /gpu:1 --batch-size 12 --postfix 42

exit

