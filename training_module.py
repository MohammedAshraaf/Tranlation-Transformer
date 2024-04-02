import torch
import lightning.pytorch as pl
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingModule(pl.LightningModule):
    def __init__(self, model, loss_cls, padding_idx):
        super().__init__()
        self.model = model
        self.loss_cls = loss_cls
        self.padding_idx = padding_idx

    def __generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz),  device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.__generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        src_padding_mask = (src == self.padding_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.padding_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        src, tgt = batch
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.__create_mask(src, tgt_input)
        logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = self.loss_cls(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.__create_mask(src, tgt_input)
        logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        val_loss = self.loss_cls(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-7,
            verbose=True
        )
        return [optimizer], [scheduler]
