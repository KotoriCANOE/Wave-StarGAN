postfix=42
python graph.py --postfix $postfix --num-domains 10 --discriminator-model "/my/Project/SpeakerRecognition/model153.tmp/model.pb"
python freeze_graph.py --input_binary False --input_graph model$postfix.tmp/model.graphdef --input_checkpoint model$postfix.tmp/model --output_graph model$postfix.tmp/model.pb --output_node_names Output

exit
