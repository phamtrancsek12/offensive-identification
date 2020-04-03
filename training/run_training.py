""" Main training script """
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from model.bertlstm import BertLSTM
from train import train
from config import TRAINING_SCHEME, TRAIN_FILE, VALID_FILE, LABEL_SIZE
from utils.load_data import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])
set_seed()


def main():

    # Defined model to use
    TRANSFORMER_MODEL = BertLSTM
    TRANSFORMER_TOKENIZER = BertTokenizer
    TRANSFORMER_CONFIG = BertConfig

    # Prepare data
    tokenizer = TRANSFORMER_TOKENIZER.from_pretrained(TRAINING_SCHEME["model_path"], do_lower_case=TRAINING_SCHEME["do_lower_case"])
    train_dataset = load_and_cache_examples(tokenizer, TRAIN_FILE)
    valid_dataset = load_and_cache_examples(tokenizer, VALID_FILE)

    # Load model
    config = TRANSFORMER_CONFIG.from_pretrained(TRAINING_SCHEME["model_path"], num_labels=LABEL_SIZE)
    model = TRANSFORMER_MODEL.from_pretrained(TRAINING_SCHEME["model_path"], from_tf=False, config=config)

    # Log model config
    logger.info("***** Running training *****")
    logger.info("classifier type = {}".format(TRAINING_SCHEME["classifier_type"]))
    logger.info("pretrained version = {}".format(TRAINING_SCHEME["pretrained_version"]))
    logger.info("Num examples = {}".format(len(train_dataset)))
    logger.info("Num Epochs = {}".format(TRAINING_SCHEME["num_training_epoches"]))

    # Train
    model, global_step, tr_loss = train(train_dataset, valid_dataset, model)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Done!")


if __name__ == "__main__":
    main()
