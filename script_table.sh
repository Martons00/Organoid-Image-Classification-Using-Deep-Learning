# python table_training.py outputs/OrganoidsINRIA_reduced/swinunetr > tables/results_testing/swinunetr_reduced_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced/swinvit > tables/results_testing/swinvit_reduced_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced/resnet50 > tables/results_testing/resnet50_reduced_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced/resnet18 > tables/results_testing/resnet18_reduced_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced/densenet > tables/results_testing/densenet_reduced_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced/swinunetr+noah > tables/results_testing/swinunetr+noah_reduced_result.md

# python table_training.py outputs/OrganoidsINRIA_reduced_128/swinunetr > tables/results_testing/swinunetr_reduced_128_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced_128/swinvit > tables/results_testing/swinvit_reduced_128_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced_128/resnet18 > tables/results_testing/resnet18_reduced_128_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced_128/densenet > tables/results_testing/densenet_reduced_128_result.md
# python table_training.py outputs/OrganoidsINRIA_reduced_128/swinunetr+noah > tables/results_testing/swinunetr+noah_reduced_128_result.md


# python table_training.py outputs/OrganoidsINRIA/swinunetr > tables/results_testing/swinunetr_full_result.md
# python table_training.py outputs/OrganoidsINRIA/resnet50 > tables/results_testing/resnet50_full_result.md
# python table_training.py outputs/OrganoidsINRIA/resnet18 > tables/results_testing/resnet18_full_result.md
# python table_training.py outputs/OrganoidsINRIA/densenet > tables/results_testing/densenet_full_result.md
# python table_training.py outputs/OrganoidsINRIA/swinunetr+noah > tables/results_testing/swinunetr+noah_full_result.md
# python table_training.py outputs/OrganoidsINRIA/swinvit > tables/results_testing/swinvit_full_result.md

# 
python table_training.py outputs/OrganoidsINRIA_MC/swinunetr > tables/results_testing/swinunetr_MC_result.md
# python table_training.py outputs/OrganoidsINRIA_MC/swinvit > tables/results_testing/swinvit_MC_result.md
# 
python table_training.py outputs/OrganoidsINRIA_MC/resnet18 > tables/results_testing/resnet18_MC_result.md
# 
python table_training.py outputs/OrganoidsINRIA_MC/densenet > tables/results_testing/densenet_MC_result.md
# python table_training.py outputs/OrganoidsINRIA_MC/swinunetr+noah > tables/results_testing/swinunetr+noah_MC_result.md


curl -X POST "https://zenodo.org/api/deposit/depositions/17235990/actions/newversion?access_token=RJymxhbPTVcuVHqjfr3Ps9razAZhuQMp520TtCJhUFIjo3HS6iav13FVKqM6"
curl -X PUT --progress-bar -H "Authorization: Bearer RJymxhbPTVcuVHqjfr3Ps9razAZhuQMp520TtCJhUFIjo3HS6iav13FVKqM6" --upload-file organoidsDatasetv2.zip "https://zenodo.org/api/files/18759368/organoidsDatasetv2.zip"
curl -X POST "https://zenodo.org/api/deposit/depositions/18759368/actions/publish?access_token=RJymxhbPTVcuVHqjfr3Ps9razAZhuQMp520TtCJhUFIjo3HS6iav13FVKqM6"

mraffael@fsophia:~/martone_project$ curl -X POST "https://zenodo.org/api/deposit/depositions/17235990/actions/newversion?access_token=RJymxhbPTVcuVHqjfr3Ps9razAZhuQMp520TtCJhUFIjo3HS6iav13FVKqM6"
{"created": "2026-02-24T14:27:21.983867+00:00", "modified": "2026-02-24T14:27:22.144022+00:00", "id": 18759368, "conceptrecid": "17234146", "conceptdoi": "10.5281/zenodo.17234146", "metadata": {"title": "Dataset of mice prostate Organoid after EDC effects", "description": "<p>Dataset of mice prostate Organoid after EDC effects from Paris and Nice laboratories. Cropped, 512x512 uint8.</p>", "access_right": "open", "creators": [{"name": "INRIA Raffaele Martone", "affiliation": null}], "license": "cc-by-4.0", "imprint_publisher": "Zenodo", "upload_type": "dataset", "prereserve_doi": {"doi": "10.5281/zenodo.18759368", "recid": 18759368}}, "title": "Dataset of mice prostate Organoid after EDC effects", "links": {"self": "https://zenodo.org/api/deposit/depositions/18759368", "html": "https://zenodo.org/deposit/18759368", "badge": "https://zenodo.org/badge/doi/.svg", "files": "https://zenodo.org/api/deposit/depositions/18759368/files", "bucket": "https://zenodo.org/api/files/ce39b6c3-83ab-41f4-bf62-32b07c6cf936", "latest_draft": "https://zenodo.org/api/deposit/depositions/18759368", "latest_draft_html": "https://zenodo.org/deposit/18759368", "publish": "https://zenodo.org/api/deposit/depositions/18759368/actions/publish", "edit": "https://zenodo.org/api/deposit/depositions/18759368/actions/edit", "discard": "https://zenodo.org/api/deposit/depositions/18759368/actions/discard", "newversion": "https://zenodo.org/api/deposit/depositions/18759368/actions/newversion"}, "record_id": 18759368, "owner": 1400266, "files": [{"id": "534b6d87-667b-4bf4-a0c2-9c98934257f6", "filename": "Organoids_Dataset.zip", "filesize": 18110725059, "checksum": "70b3e96062eaff4d6343b715fbaab04e", "links": {"self": "https://zenodo.org/api/deposit/depositions/18759368/files/534b6d87-667b-4bf4-a0c2-9c98934257f6", "download": "https://zenodo.org/api/records/18759368/draft/files/Organoids_Dataset.zip/content"}}], "state": "unsubmitted", "submitted": false}mraffael@fsophia:~/mart