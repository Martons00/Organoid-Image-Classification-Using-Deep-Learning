oarnodes -J --sql "gpu IN (128,129,130,134,135,136,137,199,200,201)" \
| jq -r '
  # Raccogli tutte le risorse in un array
  [ .[] ] as $a
  # Trova le gpu per cui TUTTE le risorse nel gruppo hanno jobs nullo/vuoto
  | (
      $a
      | sort_by(.gpu)                    # richiesto prima di group_by
      | group_by(.gpu)                   # [[obj,...] per ogni gpu]
      | map(select(all(.[]; (.jobs == null or .jobs == ""))))  # solo gruppi tutti liberi
      | map(.[0].gpu | tostring)         # elenco gpu “tutte libere” come stringhe
    ) as $ok
  # Tieni solo le risorse la cui gpu è in $ok (evita cattivo scoping catturando .gpu in $g)
  | $a
  | map(select((.gpu | tostring) as $g | ($ok | index($g))))
  | unique_by([.cluster,.gpu])
  | ( .[] | [ .cluster, (.gpu|tostring) ] )
  | @tsv
' 
