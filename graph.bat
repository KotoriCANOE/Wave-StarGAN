cd /d "%~dp0"

FOR %%i IN (40) DO (
	python graph.py --postfix %%i --num-domains 10 --discriminator-model "D:\Project\SpeakerRecognition\model153.tmp\model.pb"
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Output
)

pause

FOR %%i IN (20) DO (
	python graph.py --postfix %%i --num-domains 10 --discriminator-model "D:\Project\SpeakerRecognition\model152.tmp\model.pb"
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Output
)
