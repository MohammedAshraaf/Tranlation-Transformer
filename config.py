class Config:
    special_tokens = {
        '<unk>': 0,
        '<pad>': 1,
        '<bos>': 2,
        '<eos>': 3
    }

    model_config = {
        'EMB_SIZE': 512,
        'NHEAD': 8,
        'FFN_HID_DIM': 512,
        'BATCH_SIZE': 128,
        'NUM_ENCODER_LAYERS':  3,
        'NUM_DECODER_LAYERS': 3,
    }

    training_config = {
        'batch_size': 64
    }
