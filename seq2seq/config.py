# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 32
    bidirectional = True
    output_size = 3
    max_epochs = 60
    lr = 0.75   #0.5 ori
    batch_size = 128
    dropout_keep = 0.8
    max_sen_len = None # Sequence length for RNN None for orinal
    data_len='8000'
    model_name='Seq2Seq'