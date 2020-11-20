import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Output txt file.")
    parser.add_argument("--length", help="Options: 4, 6, 8, 10. Do not include argument for full length scrambles.")
    args = parser.parse_args()

    scramble_length = int(args.length) if args.length else 11

    if scramble_length not in [4, 6, 8, 10, 11]:
        print('EXIT: invalid scramble length')
        return -1

    filename = args.filename.replace('.txt', '')

    path = 'raw/' + ('full' if scramble_length == 11 else str(scramble_length)) + '/'

    paths = Path(path).glob('**/*.json')

    f = open(filename + ".txt", "w")

    def format_scramble(s: str, length: int):
        str1 = ' '
        as_list = list(s.split())[0:length]
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
                            f.write(format_scramble(s2, scramble_length))
                            cnt += 1

    f.close()

    print(f'{cnt} scrambles saved to {filename}.txt\nMove count (HTM): {scramble_length}')


main()
