python train.py "/my/Datasets/NSH_APP" --discriminator-model "/my/Project/SpeakerRecognition/model151.tmp/model.pb" --processes 1 --max-steps 16000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 17

exit

python train.py "/my/Datasets/NSH_APP_npz" --packed --num-domains 10 --processes 1 --max-steps 16000 --random-seed 0 --device /gpu:0 --batch-size 12 --postfix 13
