""" Inference: Generate pseudo soft and hard label"""
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from model.bertlstm import BertLSTM
from config import TRAINING_SCHEME, DEVICE, DATA_DIR, FILENAME, LABEL_SIZE
from utils.load_data import load_and_cache_examples
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])

def inference(model, dataset):
    """
    Run inference
    """
    model.to(DEVICE)
    model.eval()

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=TRAINING_SCHEME["batch_size"])

    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
    preds = None

    for step, batch in enumerate(epoch_iterator):
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            # Convert input format of BERT model
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3],
                      'label_ids': batch[4]}
            # Init
            model.batch_size = len(batch[0])
            model.lstm_hidden = model.init_hidden()
            # Fit data to model
            outputs = model(**inputs)
            # Get loss and logits
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.exp(preds)

    soft_label = [x[1] for x in preds]
    hard_label = [np.argmax(p) for p in preds]

    return soft_label, hard_label


def main():
    """ Main function"""
    parser = ArgumentParser()
    parser.add_argument('--model_path', dest="model_path", required=True)
    parser.add_argument('--inference_file', dest="inference_file", required=True)
    parser.add_argument('--output_file', dest="output_file", required=True)
    args = vars(parser.parse_args())

    # Defined model to use
    TRANSFORMER_MODEL = BertLSTM
    TRANSFORMER_TOKENIZER = BertTokenizer
    TRANSFORMER_CONFIG = BertConfig

    # Prepare data
    tokenizer = TRANSFORMER_TOKENIZER.from_pretrained(TRAINING_SCHEME["model_path"], do_lower_case=TRAINING_SCHEME["do_lower_case"])
    dataset = load_and_cache_examples(tokenizer, args["inference_file"])

    # Load model
    config = TRANSFORMER_CONFIG.from_pretrained(TRAINING_SCHEME["model_path"], num_labels=LABEL_SIZE)
    model = TRANSFORMER_MODEL.from_pretrained(args["model_path"], from_tf=False, config=config)

    # Log model config
    logger.info("***** Running inference *****")
    soft_label, hard_label = inference(model, dataset)

    # Save result
    df = pd.read_csv(DATA_DIR + FILENAME[args["inference_file"]][0])
    for i in range(1, len( FILENAME[args["inference_file"]])):
        temp = pd.read_csv(DATA_DIR + FILENAME[args["inference_file"]][i])
        df = df.concat([df, temp])
        df = df.reset_index()
        del df["index"]
    df["soft_label"] = soft_label
    df["hard_label"] = hard_label

    df.to_csv(args["output_file"], index=False)


if __name__ == "__main__":
    main()
