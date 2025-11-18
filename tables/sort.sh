#python sort_md_table.py results_validation/light/all_result_reduced.md --key ValAcc --desc --output results_validation/light/Sorted_all_result_reduced.md
#python sort_md_table.py results_validation/light/all_result_full.md --key ValAcc --desc --output results_validation/light/Sorted_all_result_full.md
cd results && python filter_markdown_table.py && cd ..
python sort_md_table.py results/light/00_reduced_results.md --key TestAcc --desc --output results/Sorted_all_results_reduced.md
python sort_md_table.py results/light/00_full_results.md --key TestAcc --desc --output results/Sorted_all_results_full.md