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
        dict = {'\'\'\'': '', '--': '', 'ec}}': '', '{{ec}}': '', '\'\'\'\'\'': '', '\'\'\'\'': '', '}}':'', '{{':''}
        dict2 = {'[[':'',']]':'', '[':'',']':''}
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
                strip_text = striphtml(re.sub(r'\'\'\'[^\'\'\']+\'\'\'', '', line[4:len(line) - 1]))
                strip_text = multiple_replace(dict, strip_text)
                for j, char in enumerate(strip_text):
                    z = j
                    if char.isalpha() or char == '[':
                        break

                strip_text = strip_text[z:]
                strip_text = replace_substring(strip_text)
                strip_text = multiple_replace(dict2, strip_text)

                text.append(strip_text)

                i += 1

            if i % 160000 == 0 and i > 1:
                wiki_df = pd.DataFrame({'voter': users, 'subject': up_for_election,
                                        'result': result_election, 'text': text})
                print(wiki_df)
                wiki_df.to_pickle('data/df_for_bert')
                break


def replace_substring(data):
    old_string = data
    try:
        found = re.findall(r'\[\[[\d\D]*?]]', data)
        if len(found) > 0:
            for text_f in found:
                text = text_f.partition(':')

                if text[2] is not '':
                    target_text = text[2][:-2].partition('|')
                    if target_text[2] is '':
                        text_sub = target_text[0]
                        data = data.replace(text_f, text_sub)
                    else:
                        text_sub = target_text[2]
                        data = data.replace(text_f, text_sub)
        return data
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = ''  # apply your error handling

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def multiple_replace(dict, line):
    rgx = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return rgx.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], line)


if __name__ == '__main__':
    parse_wiki_data()
