# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 32
    bidirectional = True
    output_size = 3
    max_epochs = 60
    lr = 0.75
    batch_size = 64
    max_sen_len = 30 # Sequence length for RNN
    dropout_keep = 0.8
    data_len='4000'
    model_name='TextRNN'