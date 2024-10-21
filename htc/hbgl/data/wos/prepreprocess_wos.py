import pandas as pd

INPUT_DIR = 'Meta-data/Data.xlsx'
OUTPUT_DIR = 'Meta-data/Data.txt'

def convert_to_text():
    df = pd.read_excel(INPUT_DIR)
    df.to_csv(OUTPUT_DIR, index=False, sep='\t')

if __name__ == '__main__':
    convert_to_text()