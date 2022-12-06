from . import AttrDict

chd_z_dim = 512

params = AttrDict(
    # Training params
    batch_size=128,
    max_epoch=1000,
    learning_rate=1e-3,
    max_grad_norm=10,
    fp16=True,
    tfr_chd=(0.5, 0),  # teacher-forcing rate for chord

    # Data params
    num_workers=4,
    pin_memory=True,

    # chd
    chd_n_step=32,
    chd_input_dim=36,
    chd_z_input_dim=512,
    chd_hidden_dim=512,
    chd_z_dim=chd_z_dim
)
