oarnodes -J --sql "gpu IN (134,135,136,137,199,200,201)" \
| jq -r '
  [.[] | select(.jobs == null or .jobs == "") | [.host, (.gpu|tostring)]]
  | unique
  | (["host","gpu"], .[])
  | @tsv
' \
| column -t -s $'\t'
