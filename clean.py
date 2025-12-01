def rimuovi_righe_speciali(input_path, output_path):
    # Legge il file input, elimina righe con i pattern richiesti e scrive l'output pulito
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not (line.startswith('After') or line.startswith('Before') or line.startswith('FINAL')):
                f_out.write(line)

if __name__ == "__main__":
    input_file = 'OAR_2130804.out'
    output_file = 'OAR_2130804_cleaned.out'
    rimuovi_righe_speciali(input_file, output_file)