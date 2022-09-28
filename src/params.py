import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


params = AttrDict(
    # Training params
    batch_size=128,
    max_epoch=100,
    learning_rate=2e-4,
    max_grad_norm=1e5,
    fp16=False,

    # Data params
    num_workers=4,
    pin_memory=True,

    # Model params
    beta=0.1,
    weights=(1, 0.5),
    z_dim=512,
    # ==== directly copied from diffwave
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=True,
    noise_schedule=np.linspace(1e-4, 0.05, 1000).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    # ====

    # unconditional sample len
)
