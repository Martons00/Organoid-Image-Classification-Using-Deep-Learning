python table_training.py outputs/OrganoidsINRIA_reduced/swinunetr > swinunetr_reduced_result.md
python table_training.py outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc > swinunetr_result.md
python table_training.py outputs/OrganoidsINRIA_reduced/resnet50 > resnet50_reduced_result.md
python table_training.py outputs/OrganoidsINRIA_reduced/resnet18 > resnet18_reduced_result.md

# BASE="/home/mraffael/martone_project/Organoids_Dataset_256"
# DEST="$BASE/removed"
# LIST="/home/mraffael/martone_project/Organoid-Image-Classification-Using-Deep-Learning/problematic_samples.txt"   # ogni riga: Cystiques_Nice_Reduce/202407_Nice_orga3_3.tif

# cd "$BASE" && rsync -aR --files-from="$LIST" ./ "$DEST"/ --remove-source-files
