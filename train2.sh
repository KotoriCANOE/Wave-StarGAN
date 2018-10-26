python train.py "/my/Datasets/NSH_APP" --discriminator-model "/my/Project/SpeakerRecognition/model153.tmp/model.pb" --processes 4 --max-steps 96000 --random-seed 0 --device /gpu:1 --batch-size 12 --postfix 36

exit

python train.py "/my/Datasets/NSH_APP" --processes 4 --max-steps 32000 --random-seed 0 --device /gpu:1 --batch-size 12 --postfix 24
