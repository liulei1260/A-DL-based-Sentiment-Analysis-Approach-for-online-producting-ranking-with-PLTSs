# config.py

class Config(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 3
    max_epochs = 20
    lr = 0.3 #0.3
    batch_size = 64
    max_sen_len = 30 #30
    dropout_keep = 0.8 #0.8
    data_len='8000'
    model_name='TextCNN'