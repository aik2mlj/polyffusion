from . import AttrDict

chd_z_dim = 512
txt_z_dim = 256 * 4  # 4 pnotree concated, each 256

d_cond = chd_z_dim + txt_z_dim

params = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=100,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,
    # Data params
    num_workers=0,
    pin_memory=False,
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
    cond_type="chord+txt",
    cond_mode="mix2",  # {mix, cond, uncond}
    # whether to use chord encoder from polydis
    use_enc=True,
    chd_n_step=32,
    chd_input_dim=36,
    chd_z_input_dim=512,
    chd_hidden_dim=512,
    chd_z_dim=chd_z_dim,
    txt_emb_size=256,
    txt_hidden_dim=1024,
    txt_z_dim=256,
    txt_num_channel=10,
)
