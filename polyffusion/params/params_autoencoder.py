from . import AttrDict

params = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=100,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=False,

    # Data params
    num_workers=4,
    pin_memory=True,

    # unet
    in_channels=3,
    out_channels=3,
    z_channels=4,
    channels=64,
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    emb_channels=4,

    # kl FIXME: not used
    disc_start=50001,
    kl_weight=0.000001,
    disc_weight=0.5,
)
