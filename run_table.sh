#python make_table_training.py outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc > layer4+encoder10+fc_result.md
#python make_table_training.py outputs/OrganoidsINRIA/swinunetr/encoder10+fc > encoder10+fc_result.md
#cat layer4+encoder10+fc_result.md encoder10+fc_result.md > combined_swinunetr_results.md

#python make_table_training.py outputs/OrganoidsINRIA/resnet50 > resnet50_result.md
python make_table_training.py outputs/OrganoidsINRIA/resnet18 > resnet18_result.md

#cat resnet50_result.md resnet18_result.md > combined_resnet_results.md
