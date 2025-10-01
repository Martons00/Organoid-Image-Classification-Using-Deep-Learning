import os
import re
import csv
import json
from datetime import datetime
from collections import Counter

class KeywordFileAnalyzer:
    def __init__(self, input_folder, keywords=None, case_sensitive=False):
        """
        Inizializza l'analizzatore di parole chiave nei file

        Args:
            input_folder (str): Percorso della cartella da analizzare
            keywords (list): Lista delle parole chiave da cercare
            case_sensitive (bool): Se True, la ricerca è sensibile alle maiuscole
        """
        self.input_folder = input_folder
        self.keywords = keywords if keywords else ["Nice", "Paris", "Noyau"]
        self.case_sensitive = case_sensitive
        self.results = {
            'keyword_counts': {kw: 0 for kw in self.keywords},
            'total_files': 0,
            'files_analyzed': [],
            'files_with_keywords': [],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def analyze_folder(self):
        """
        Analizza la cartella e conta le occorrenze delle parole chiave
        """
        if not os.path.exists(self.input_folder):
            print(f"ERRORE: La cartella '{self.input_folder}' non esiste!")
            return False

        print(f"Analisi in corso della cartella: {self.input_folder}")
        print(f"Parole chiave cercate: {', '.join(self.keywords)}")
        print(f"Ricerca case-sensitive: {'Sì' if self.case_sensitive else 'No'}")
        print("-" * 60)

        # Scansiona ricorsivamente tutti i file
        for root, dirs, files in os.walk(self.input_folder):
            for filename in files:
                self.results['total_files'] += 1
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, self.input_folder)

                file_info = {
                    'filename': filename,
                    'relative_path': relative_path,
                    'full_path': full_path,
                    'keywords_found': []
                }

                # Cerca ogni parola chiave nel nome del file
                for keyword in self.keywords:
                    flags = 0 if self.case_sensitive else re.IGNORECASE
                    if re.search(re.escape(keyword), filename, flags):
                        self.results['keyword_counts'][keyword] += 1
                        file_info['keywords_found'].append(keyword)

                self.results['files_analyzed'].append(file_info)

                # Se sono state trovate parole chiave, aggiungi alla lista speciale
                if file_info['keywords_found']:
                    self.results['files_with_keywords'].append(file_info)

        return True

    def print_report(self):
        """
        Stampa il report a console
        """
        print("\n" + "="*70)
        print("REPORT FINALE - CONTEGGIO PAROLE CHIAVE")
        print("="*70)
        print(f"Data analisi: {self.results['analysis_date']}")
        print(f"Cartella: {self.input_folder}")
        print(f"Totale file analizzati: {self.results['total_files']}")
        print()

        print("CONTEGGIO PER PAROLA CHIAVE:")
        print("-" * 35)
        total_occurrences = 0
        for keyword, count in self.results['keyword_counts'].items():
            print(f"  {keyword:15} : {count:3d} occorrenze")
            total_occurrences += count

        print("-" * 35)
        print(f"  {'TOTALE':15} : {total_occurrences:3d} occorrenze")
        print()

        if self.results['files_with_keywords']:
            print(f"FILE CON PAROLE CHIAVE TROVATE ({len(self.results['files_with_keywords'])}):")
            print("-" * 70)
            for i, file_info in enumerate(self.results['files_with_keywords'], 1):
                keywords_str = ", ".join(file_info['keywords_found'])
                print(f"{i:2d}. {file_info['filename']}")
                print(f"    Parole: {keywords_str}")
                print(f"    Percorso: {file_info['relative_path']}")
                print()
        else:
            print("Nessun file contenente le parole chiave è stato trovato.")

    def save_csv_report(self, output_file="keyword_analysis_report.csv"):
        """
        Salva il report in formato CSV
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'relative_path', 'full_path', 'keywords_found', 'keyword_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Scrivi solo i file che contengono parole chiave
            for file_info in self.results['files_with_keywords']:
                writer.writerow({
                    'filename': file_info['filename'],
                    'relative_path': file_info['relative_path'],
                    'full_path': file_info['full_path'],
                    'keywords_found': '; '.join(file_info['keywords_found']),
                    'keyword_count': len(file_info['keywords_found'])
                })

        print(f"Report CSV salvato in: {output_file}")
        return output_file

    def save_json_report(self, output_file="keyword_analysis_report.json"):
        """
        Salva il report completo in formato JSON
        """
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.results, jsonfile, indent=2, ensure_ascii=False)

        print(f"Report JSON completo salvato in: {output_file}")
        return output_file

def main():

    INPUT_FOLDER = "/Volumes/LaCie/Organoids/Chouxfleurs_raw"
    #INPUT_FOLDER = "/Volumes/LaCie/Organoids/Noyaux/Compact"
    #INPUT_FOLDER = "/Volumes/LaCie/Organoids/Noyaux/Cystiques"


    # PAROLE CHIAVE DA CERCARE (puoi modificare questa lista)
    KEYWORDS = ["Nice", "Paris", "Noyau"]

    # RICERCA SENSIBILE ALLE MAIUSCOLE (True/False)
    CASE_SENSITIVE = False

    # NOMI FILE OUTPUT
    CSV_OUTPUT = "Chouxfleurs_raw_conteggio_parole_chiave.csv"
    JSON_OUTPUT = "Chouxfleurs_raw_analisi_completa.json"
    # ====================================================

    print("ANALIZZATORE PAROLE CHIAVE NEI TITOLI DEI FILE")
    print("=" * 50)

    # Crea l'analizzatore
    analyzer = KeywordFileAnalyzer(INPUT_FOLDER, KEYWORDS, CASE_SENSITIVE)

    # Esegui l'analisi
    if analyzer.analyze_folder():
        # Mostra il report
        analyzer.print_report()

        # Salva i report
        analyzer.save_csv_report(CSV_OUTPUT)
        analyzer.save_json_report(JSON_OUTPUT)

        print("\n" + "="*70)
        print("ANALISI COMPLETATA CON SUCCESSO!")
        print("="*70)
    else:
        print("ERRORE durante l'analisi. Controlla il percorso della cartella.")

if __name__ == "__main__":
    main()
