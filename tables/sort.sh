python filter_markdown_table.py
python sort_md_table.py results/complete/H_results.md --key TestAcc --desc --output results/complete/sorted_H_results.md
python sort_md_table.py results/complete/R_results.md --key TestAcc --desc --output results/complete/sorted_R_results.md

python sort_md_table.py results/light/H_results.md --key TestAcc --desc --output results/light/sorted_H_results.md
python sort_md_table.py results/light/R_results.md --key TestAcc --desc --output results/light/sorted_R_results.md

python sort_md_table.py ./results/light/all_results.md --key TestAcc --desc --output ./results/light/sorted_all_results.md