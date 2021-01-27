import pandas as pd
import numpy as np
import re


def load_data(batch_size, file):
    df = pd.read_csv(r'data/' + file)
    print(df['file'].head(30))


def parse_wiki_data():
    users = []
    up_for_election = []
    result_election = []
    text = []

    with open('data/wiki-rfa.txt', encoding="utf8") as f:
        j = 0
        i = 0
        for line in f:
            if line[0:3] == 'SRC':
                users.append(line[4:len(line) - 1])
                i += 1
            if line[0:3] == 'TGT':
                up_for_election.append(line[4:len(line) - 1])
                i += 1
            if line[0:3] == 'RES':
                result_election.append(int(line[4:len(line) - 1]))
                i += 1
            if line[0:3] == 'TXT':
                text.append(striphtml(re.sub(r'\'''[^\''']+\'''', '', line[4:len(line) - 1])[5:]))
                i += 1

            if i % 160000 == 0 and i > 1:
                wiki_df = pd.DataFrame({'voter': users, 'subject': up_for_election,
                                        'result': result_election, 'text': text})
                print(wiki_df['text'])
                break


def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


if __name__ == '__main__':
    parse_wiki_data()
