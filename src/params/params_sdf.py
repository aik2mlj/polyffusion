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
    in_channels=2,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=8,
    tf_layers=1,
    d_cond=36,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215,

    # img
    img_h=128,
    img_w=128,

    # conditional
    cond_mode="mix"  # {mix, cond, uncond}
)
