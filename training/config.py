""" config """
import torch

DATA_DIR = "../data"
FILENAME = {
    "train": ["train.csv"],
    "valid": ["valid.csv"],
    "generate": ["generate.csv"],
}
TRAIN_FILE = "train"
VALID_FILE = "valid"
LABEL_LIST = [0, 1]
LABEL_SIZE = len(LABEL_LIST)

SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices: {}".format(DEVICE))

# Define different training schemes (for teacher and student model)
TRAINING_SCHEMES = [
    {
        # Bert
        "classifier_type": "BertLSTM",
        "model_path": "model/bert_tweets_pretrained/pytorch",
        "pretrained_version": "tweets_pretrained",
        "do_lower_case": True,
        "max_seq_length": 128,
        # LSTM & FC
        "lstm_num_layers": 2,
        "lstm_hidden_dim": 1024,
        "relu_dim_list": [1024, 256, 128, 64],
        # Training config
        "num_training_epoches": 3,
        "batch_size": 64,
        "weight_decay": 0.01,
        "adam_eps": 1e-8,
        "warmup_proportion": 0.25,
        "max_grad_norm": 1.0,
        "learning_rate": 1e-5,
        "dropout_prob": 0.5,
        "soft_label_ratio": 0.3

    }
]
TRAINING_SCHEME = TRAINING_SCHEMES[0] # Choose training scheme