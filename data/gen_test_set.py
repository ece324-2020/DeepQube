import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Output txt file.")
    args = parser.parse_args()

    filename = args.filename.replace('.txt', '')

    path = 'raw/test/'

    paths = Path(path).glob('**/*.json')

    f = open(filename + ".txt", "w")

    def format_scramble(s: str):
        str1 = ' '
        as_list = list(s.split())
        str2 = str1.join(as_list)
        str2 = str2.replace('R2', 'R R')
        str2 = str2.replace('U2', 'U U')
        str2 = str2.replace('F2', 'F F')
        return str2 + '\n'

    cnt = 0

    for p in paths:
        with open(str(p)) as json_file:
            data = json.load(json_file)
            events = data['wcif']['events']
            for event in events:
                if event['id'] == '222':
                    scrambles = event['rounds'][0]['scrambleSets']
                    for s1 in scrambles:
                        for s2 in s1['scrambles']:
                            f.write(format_scramble(s2))
                            cnt += 1

    f.close()

    print(f'{cnt} test set scrambles saved to {filename}.txt')


main()
