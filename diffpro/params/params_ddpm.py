from . import AttrDict

params = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=100,
    learning_rate=2e-5,
    max_grad_norm=10,
    fp16=False,

    # Data params
    num_workers=4,
    pin_memory=True,

    # Model params
    beta=0.1,
    weights=(1, 0.5),

    # Number of channels in the image. $3$ for RGB.
    image_channels=2,
    # Image size
    image_size_h=128,
    image_size_w=128,
    # Number of channels in the initial feature map
    n_channels=64,
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers=[1, 2, 2, 4],
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention=[False, False, False, True],
    # Number of time steps $T$
    n_steps=1000,
)
