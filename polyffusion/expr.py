from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader import get_val_dataloader
from inference_sdf import *
from utils import check_prmat2c_integrity, nested_map, convert_json_to_yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


def prompt_generation(
    expr: Experiments, num, output_dir, check_integrity=True
):  # start from the 3-rd bar
    val_dl = get_val_dataloader(16)
    gen = []
    for i, batch in enumerate(tqdm(val_dl)):
        if i >= num:
            break
        batch = nested_map(
            batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
        prmat2c, pnotree, chord, prmat = batch
        x0 = expr.predict(prmat2c, None, 0.0, False)
        gen.append(x0)
        # orig = prmat2c
        # bar_list = [2, 3, 4, 5, 6, 7]
        # expr.inpaint()
    gen = torch.cat(gen)
    if check_integrity:
        print(check_prmat2c_integrity(gen))
    prmat2c_to_midi_file(gen, f"{output_dir}/uncond.mid")


def acc_arrangement(expr: Experiments, num, output_dir):
    val_dl = get_val_dataloader(16, use_track=[0])  # only melody
    gen = []
    for i, batch in enumerate(tqdm(val_dl)):
        if i >= num:
            break
        batch = nested_map(
            batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
        prmat2c, pnotree, chord, prmat = batch
        x0 = expr.inpaint(
            prmat2c, "below", prmat2c, None, uncond_scale=0.0, no_output=True
        )
        gen.append(x0)
    gen = torch.cat(gen)
    prmat2c_to_midi_file(gen, f"{output_dir}/acc_arr.mid")


def inpaint_bars(expr: Experiments, num, output_dir):
    val_dl = get_val_dataloader(16)
    gen = []
    for i, batch in enumerate(tqdm(val_dl)):
        if i >= num:
            break
        batch = nested_map(
            batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
        prmat2c, pnotree, chord, prmat = batch
        x0 = expr.inpaint(
            prmat2c,
            "bars",
            prmat2c,
            None,
            uncond_scale=0.0,
            bar_list=[2, 3, 4, 5],
            no_output=True,
        )
        gen.append(x0[:, :, 32:96, :])  # NOTE: only output the inpainted notes
    gen = torch.cat(gen)
    prmat2c_to_midi_file(gen, f"{output_dir}/inp_bars.mid")


def chd_conditioning(expr: Experiments, model, num, output_dir, uncond_scale=1.0):
    val_dl = get_val_dataloader(16)
    gen = []
    chd = []
    for i, batch in enumerate(tqdm(val_dl)):
        if i >= num:
            break
        batch = nested_map(
            batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
        prmat2c, pnotree, chord, prmat = batch
        print(chord.shape)
        cond = model._encode_chord(chord)
        x0 = expr.generate(cond, None, uncond_scale, no_output=True)
        gen.append(x0)
        chd.append(chord)
    gen = torch.cat(gen)
    chd = torch.stack(chd).cpu().numpy()
    print(chd.shape)
    np.save(f"{output_dir}/chd[{uncond_scale}].npy", chd)
    prmat2c_to_midi_file(gen, f"{output_dir}/chd_cond[{uncond_scale}].mid")


def txt_conditioning(
    expr: Experiments, model, num, output_dir, uncond_scale=1.0, use_track=[0, 1, 2]
):
    val_dl = get_val_dataloader(16, use_track=use_track)
    gen = []
    orig = []
    for i, batch in enumerate(tqdm(val_dl)):
        if i >= num:
            break
        batch = nested_map(
            batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
        prmat2c, pnotree, chord, prmat = batch
        print(prmat.shape)
        cond = model._encode_txt(prmat)
        x0 = expr.generate(cond, None, uncond_scale, no_output=True)
        gen.append(x0)
        orig.append(prmat2c)
    gen = torch.cat(gen)
    orig = torch.cat(orig)
    print(orig.shape)
    prmat2c_to_midi_file(gen, f"{output_dir}/txt_cond[{uncond_scale}].mid")
    prmat2c_to_midi_file(orig, f"{output_dir}/txt_orig[{uncond_scale}].mid")


if __name__ == "__main__":
    parser.add_argument(
        "--model_dir", help="directory in which trained model checkpoints are stored"
    )
    parser.add_argument("--type", help="{uncond, inp_below, inp_bars, chd}")
    parser.add_argument("--batch_num", default=10, help="how many batches to store")
    parser.add_argument("--output_dir", help="where to put the file")
    parser.add_argument("--seed")
    parser.add_argument(
        "--uncond_scale",
        default=1.0,
        help="unconditional scale for classifier-free guidance",
    )
    parser.add_argument(
        "--chkpt_name",
        default="weights_best.pt",
        help="which specific checkpoint to use (default: weights_best.pt)",
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="whether to show the images of generated piano-roll",
    )
    parser.add_argument("--use_track", help="use which tracks for pop909")
    args = parser.parse_args()
    model_label = Path(args.model_dir).parent.name
    print(f"model_label: {model_label}")

    if args.seed is not None:
        SEED = int(args.seed)
        print(f"fixed SEED = {SEED}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    # params ready
    if os.path.exists(f"{args.model_dir}/params.json"):
        convert_json_to_yaml(f"{args.model_dir}/params.json")
    params = OmegaConf.load(f"{args.model_dir}/params.yaml")

    # model ready
    autoencoder = None
    unet_model = UNetModel(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        channels=params.channels,
        attention_levels=params.attention_levels,
        n_res_blocks=params.n_res_blocks,
        channel_multipliers=params.channel_multipliers,
        n_heads=params.n_heads,
        tf_layers=params.tf_layers,
        d_cond=params.d_cond,
    )

    ldm_model = LatentDiffusion(
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        n_steps=params.n_steps,
        latent_scaling_factor=params.latent_scaling_factor,
        autoencoder=autoencoder,
        unet_model=unet_model,
    )

    pnotree_enc, pnotree_dec = None, None
    chord_enc, chord_dec = None, None
    txt_enc = None
    if params.cond_type == "pnotree":
        pnotree_enc, pnotree_dec = load_pretrained_pnotree_enc_dec(
            PT_PNOTREE_PATH, 20, device
        )
    elif params.cond_type == "chord":
        if params.use_enc:
            chord_enc, chord_dec = load_pretrained_chd_enc_dec(
                PT_CHD_8BAR_PATH,
                params.chd_input_dim,
                params.chd_z_input_dim,
                params.chd_hidden_dim,
                params.chd_z_dim,
                params.chd_n_step,
            )
    elif params.cond_type == "txt":
        if params.use_enc:
            txt_enc = load_pretrained_txt_enc(
                PT_POLYDIS_PATH,
                params.txt_emb_size,
                params.txt_hidden_dim,
                params.txt_z_dim,
                params.txt_num_channel,
            )
    else:
        raise NotImplementedError

    model = Polyffusion_SDF.load_trained(
        ldm_model,
        f"{args.model_dir}/chkpts/{args.chkpt_name}",
        params.cond_type,
        params.cond_mode,
        chord_enc,
        chord_dec,
        pnotree_enc,
        pnotree_dec,
        txt_enc,
    ).to(device)
    sampler = SDFSampler(
        model.ldm,
        is_show_image=args.show_image,
    )
    expmt = Experiments(model_label, params, sampler)

    num = int(args.batch_num)
    if args.type == "uncond":
        prompt_generation(expmt, num, args.output_dir)
    elif args.type == "inp_below":
        acc_arrangement(expmt, num, args.output_dir)
    elif args.type == "inp_bars":
        inpaint_bars(expmt, num, args.output_dir)
    elif args.type == "chd_cond":
        chd_conditioning(
            expmt, model, num, args.output_dir, uncond_scale=float(args.uncond_scale)
        )
    elif args.type == "txt_cond":
        use_track = [0, 1, 2]
        if args.use_track is not None:
            use_track = [int(x) for x in args.use_track.split(",")]
        print(use_track)
        txt_conditioning(
            expmt,
            model,
            num,
            args.output_dir,
            uncond_scale=float(args.uncond_scale),
            use_track=use_track,
        )
    else:
        raise NotImplementedError
