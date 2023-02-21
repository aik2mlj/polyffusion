from . import AttrDict

pnotree_z_dim = 512 * 4  # 4 pnotree concated, each 512

d_cond = pnotree_z_dim

params = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=50,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

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
    n_heads=4,
    tf_layers=1,
    d_cond=d_cond,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215,

    # img
    img_h=128,
    img_w=128,

    # conditional
    cond_type="pnotree",
    cond_mode="mix",  # {mix, cond, uncond}
)
