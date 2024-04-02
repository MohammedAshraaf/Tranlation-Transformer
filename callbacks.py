import lightning.pytorch as pl


def get_callbacks():
    """
    Generates the callbacks for monitoring the training
    :return: list of callbacks
    """
    # Training Checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='training_checkpoints/translation_exp',
        filename='transformer-epoch{epoch:03d}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    model_summary_callback = pl.callbacks.ModelSummary(max_depth=1)

    callbacks = [
        checkpoint_callback,
        lr_monitor_callback,
        model_summary_callback
    ]
    return callbacks
