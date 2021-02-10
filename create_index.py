import string
from collections import Counter

import pandas as pd
import numpy as np
import torch
import transformers
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer


def remove_prev_email(df):
    for i, text in enumerate(df['content']):

        split_text = text.split('From:')
        splitter_text = split_text[0].split('----')
        if splitter_text[0] != '':
            text_to_add = splitter_text[0].replace('\n', '').replace('\t', '')
            if len(text_to_add.split(' ')) > 8:
                df.loc[i, 'content'] = text_to_add
            else:
                df.drop(i, inplace=True)
        else:
            df.drop(i, inplace=True)

    return df.reset_index(drop=True)


def tokenize_data(df_text, tokenizer, colname='text', batch=None):
    # I truncate the text if it's too large, might possibly cause issues not sure though
    attention_masks = []
    padded = []
    num_input_per_email = []
    for i, text in enumerate(df_text[colname]):
        num_input = 1
        encoded_part = tokenizer.encode(text, add_special_tokens=True)
        while len(encoded_part) > 512:
            to_encode = np.array(encoded_part[:512])
            attention_masks_part = np.where(to_encode != 0, 1, 0)
            padded.append(to_encode)
            attention_masks.append(attention_masks_part)

            encoded_part = encoded_part[512:]

            num_input += 1

        padded_part = np.array([encoded_part[x] if x < len(encoded_part) else 0 for x in range(512)])
        attention_masks_part = np.where(padded_part != 0, 1, 0)

        padded.append(padded_part)
        attention_masks.append(attention_masks_part)
        num_input_per_email.append(num_input)

        if batch and i >= batch:
            break

    input_ids = torch.tensor(np.array(padded), dtype=torch.long)
    attention_masks = torch.tensor(np.array(attention_masks), dtype=torch.int64)

    return input_ids, attention_masks, np.array(num_input_per_email)


def create_val_set(df_text, seed):
    to_tag = df_text.sample(100, axis=0, random_state=seed).reset_index(drop=True)
    tags = np.zeros(len(to_tag.index))
    text_array = np.array(to_tag['content'])
    for i, text in enumerate(to_tag['content']):
        correct = False
        print('New Mail to Tag {}/100:'.format(i))
        text_list = text.split(' ')
        while len(text_list) > 35:
            to_print = ' '.join(text_list[0:35])
            text_list = text_list[35:]
            print(to_print)
        print(' '.join(text_list))
        while correct is False:
            try:
                tag = input("Please tag with 1 being trust 0 neutral -1 no trust: ")
                tag = int(tag)
                if tag == 0 or tag == 1 or tag == -1:
                    tags[i] = tag
                    correct = True
                else:
                    print('Please enter -1, 0 or 1')
            except ValueError:
                print('Please enter a number')

    tagged_df = pd.DataFrame(list(zip(text_array, tags)), columns=['text', 'label'])
    tagged_df.to_pickle('data/tagged_enron_new')


def validate_model_performance(df, model, tokenizer_bert):
    print(df)
    labels = df['label']
    input_ids, attention_masks, input_per_mail = tokenize_data(df, tokenizer_bert)
    prediction = do_predict(model, input_ids, attention_masks)[0].numpy()
    norm_prediction = softmax(prediction)

    label_pred = np.argmax(prediction, axis=1)
    for i, pred in enumerate(label_pred):
        if pred == 2:
            label_pred[i] = 1
        elif pred == 0:
            label_pred[i] = -1
        elif pred == 1:
            label_pred[i] = 0

    real_label_pred = np.empty(labels.shape)
    j = 0
    for z, num_input in enumerate(input_per_mail):
        if num_input == 1:
            real_label_pred[z] = label_pred[j]
            j += 1
        if num_input > 1:
            predictions = []
            for i in range(num_input):
                predictions.append(label_pred[j])
                j += 1

            temp = Counter(predictions)
            real_label_pred[z] = temp.most_common(1)[0][0]

    print(confusion_matrix(labels, real_label_pred, labels=[-1, 0, 1]))
    print(sum(real_label_pred == labels))


def create_trust_index(df, model, tokenizer, batch_size=100):
    input_ids, attention_masks, input_per_mail = tokenize_data(df, tokenizer, colname='content')
    print(input_per_mail.shape)
    print(df.shape)
    predictions_per_part = []

    for i in range(input_ids.shape[0] // batch_size):
        print('Batch {} / {}'.format(i, input_ids.shape[0]//batch_size))
        output = do_predict(model, input_ids[i * batch_size:(i + 1) * batch_size],
                            attention_masks[i * batch_size:(i + 1) * batch_size])
        predict = np.argmax(output[0].numpy(), axis=1)

        for j, pred in enumerate(predict):
            if pred == 2:
                predict[j] = 1
            elif pred == 0:
                predict[j] = -1
            elif pred == 1:
                predict[j] = 0

        predictions_per_part.extend(list(predict))

    output_final = do_predict(model, input_ids[(input_ids.shape[0] // batch_size) * batch_size:],
                              attention_masks[(input_ids.shape[0] // batch_size) * batch_size:])
    predict_final = np.argmax(output_final[0].numpy(), axis=1)

    for p, pred in enumerate(predict_final):
        if pred == 2:
            predict_final[p] = 1
        elif pred == 0:
            predict_final[p] = -1
        elif pred == 1:
            predict_final[p] = 0
    predictions_per_part.extend(predict_final)

    predictions = np.empty(df.shape[0])
    k = 0
    for z, num_input in enumerate(input_per_mail):
        if num_input == 1:
            predictions[z] = predictions_per_part[k]
            k += 1
        if num_input > 1:
            predictions_temp = []
            for i in range(num_input):
                predictions_temp.append(predictions_per_part[k])
                k += 1

            temp = Counter(predictions_temp)
            predictions[z] = temp.most_common(1)[0][0]
    print(predictions)
    df['trust_index'] = predictions
    df.to_pickle('data/df_with_trust')


def do_predict(model, input_ids, attention_masks):
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    return outputs


if __name__ == '__main__':
    bert_model = transformers.BertForSequenceClassification.from_pretrained(r'Finetuned_BERT')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    df = pd.read_feather('data/cleaned_emails.feather')
    df = remove_prev_email(df)
    create_trust_index(df, bert_model, tokenizer, batch_size=10)
    # input_ids, attention_masks, input_per_mail = tokenize_data(df, tokenizer, batch=10)

    # validation_set = pd.read_pickle('data/tagged_enron_new')
    # validate_model_performance(validation_set, bert_model, tokenizer)
