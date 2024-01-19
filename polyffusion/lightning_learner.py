import lightning
import torch


class LightningLearner(lightning.LightningModule):
    def __init__(self, model, optimizer, params, param_scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.param_scheduler = param_scheduler  # teacher-forcing stuff

        self.save_hyperparameters("params", "param_scheduler")

    def _categorize_loss_dict(self, loss_dict, category):
        return {f"{category}/{k}": v for k, v in loss_dict.items()}

    def training_step(self, batch, batch_idx):
        if self.param_scheduler is not None:
            scheduled_params = self.param_scheduler.step()
            loss_dict = self.model.get_loss_dict(
                batch, self.global_step, **scheduled_params
            )
        else:
            scheduled_params = None
            loss_dict = self.model.get_loss_dict(batch, self.global_step)

        # check NaN
        for loss_value in list(loss_dict.values()):
            if isinstance(loss_value, torch.Tensor) and torch.isnan(loss_value).any():
                raise RuntimeError(
                    f"Detected NaN loss at step {self.global_step}, epoch {self.epoch}"
                )
        loss = loss_dict["loss"]

        loss_dict = self._categorize_loss_dict(loss_dict, "train")
        self.log_dict(loss_dict, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.param_scheduler is not None:
            scheduled_params = self.param_scheduler.step()
            loss_dict = self.model.get_loss_dict(
                batch, self.global_step, **scheduled_params
            )
        else:
            scheduled_params = None
            loss_dict = self.model.get_loss_dict(batch, self.global_step)

        loss_dict = self._categorize_loss_dict(loss_dict, "val")
        self.log_dict(loss_dict)

    def configure_optimizers(self):
        return self.optimizer
