# config.py

class Config(object):
    num_channels = 256
    linear_size = 256
    output_size = 3
    max_epochs = 100
    lr = 0.001
    batch_size = 128
    seq_len = 1024 # 1014 in original paper
    dropout_keep = 0.5
    data_len='8000'
    model_name='CharCNN'