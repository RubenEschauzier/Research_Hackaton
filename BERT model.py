import time
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import datetime


def process_text_data(data, tokenizer):
    # I truncate the text if it's too large, might possibly cause issues not sure though
    tokenized = data['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)))
    max_len = max([len(i) for i in tokenized.values])
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask)
    attention_masks = torch.tensor(attention_mask)
    input_ids = torch.tensor(padded, dtype=torch.int64)

    return input_ids, attention_masks


def dataLoaders(dataset):
    """this function splits dataset, and creates dataloaders for training and validation sets."""
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 32

    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


def load_model():
    pass


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # Format as hh:mm:ss


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_validate(model, scheduler, optimizer, epochs, train_dataloader, validation_dataloader):
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0

        # """what happens here with drop out rate?"""
        model.train()

        for step, batch in enumerate(train_dataloader):
            # """each batch contains three pytorch tensors: input ids, attention masks, labels)"""
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            # one forward pass is performed on one epoch at the same time
            # gradients are set to zero every time
            # backward pass to capture gradients for back propagation"""
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            hidden_states_train = outputs.hidden_states

            total_train_loss += loss.item()
            loss.backward()

            """ Clip the norm of the gradients to 1.0 to prevent the "exploding gradients" problem.
            update parameters and learning rate"""
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        """calculate average loss over all examples"""
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    """measure our performance on our validation set"""
    print("")
    print("Running Validation...")
    t0 = time.time()

    """evaluation mode makes sure that you can still get to the gradients even if drop out"""
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            """forward pass, no grad as a graph is not necessary in forward prop
            Get the "logits" output : values prior to activation function like the softmax."""
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            hidden_states_train = outputs.hidden_states

        total_eval_loss += loss.item()

        """ Move logits and labels to CPU"""
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        """ calculate total accuracy over all batches."""
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    """ final accuracy for this validation run."""
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    """ average loss over all of the batches."""
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    # I have to select True otherwise it will not compute cost etc
    output_attentions=True,  # Whether the model returns attentions weights.
    output_hidden_states=True)  # Whether the model returns all hidden-states.


def train_model():
    pass


if __name__ == '__main__':
    df = pd.read_pickle('data/df_for_bert_full')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    labels = torch.tensor(df['result'])
    input_ids, attention_masks = process_text_data(df, tokenizer)
    dataset = TensorDataset(input_ids[1:100], attention_masks[1:100], labels[1:100])

    train_dl, val_dl = dataLoaders(dataset)

    pre_trained_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        # I have to select True otherwise it will not compute cost etc
        output_attentions=True,  # Whether the model returns attentions weights.
        output_hidden_states=False)  # Whether the model returns all hidden-states.

    optimizer_bert = AdamW(model.parameters(),
                           lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                           eps=1e-8)  # args.adam_epsilon  - default is 1e-8.

    num_epochs = 4  # Number of training epochs. Many epochs may be over-fitting training data.
    total_steps = len(input_ids) * num_epochs  # total number of training steps
    print(len(input_ids))
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert,
                                                     num_warmup_steps=0,  # Default value in run_glue.py
                                                     num_training_steps=total_steps)

    train_validate(pre_trained_model, scheduler_bert, optimizer_bert, num_epochs, train_dl, val_dl)
