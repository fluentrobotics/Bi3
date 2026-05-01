import argparse
import json
import os
from collections import namedtuple


def load_yaml_config(config_path):
    try:
        import yaml
    except ModuleNotFoundError:
        return load_simple_yaml_config(config_path)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return config


def load_simple_yaml_config(config_path):
    config = {}
    with open(config_path, 'r') as fp:
        for line_num, line in enumerate(fp, start=1):
            line = strip_inline_comment(line).strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError("Unsupported YAML syntax on line {}: {}".format(line_num, line))
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError("Missing config key on line {}".format(line_num))
            config[key] = parse_yaml_scalar(value.strip())
    return config


def strip_inline_comment(line):
    quote = None
    for idx, char in enumerate(line):
        if char in ["'", '"']:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return line[:idx]
    return line


def parse_yaml_scalar(value):
    if value == "":
        return None

    lower_value = value.lower()
    if lower_value in ["true", "false"]:
        return lower_value == "true"
    if lower_value in ["null", "none"]:
        return None

    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(part.strip()) for part in inner.split(",")]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_config_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.config


def apply_config_defaults(parser, config_path):
    if config_path is None:
        return

    config = load_yaml_config(config_path)
    valid_dests = {
        action.dest: action
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    case_insensitive_dests = {dest.lower(): dest for dest in valid_dests}

    defaults = {}
    for raw_key, value in config.items():
        key = raw_key.replace("-", "_")
        dest = key if key in valid_dests else case_insensitive_dests.get(key.lower())
        if dest is None:
            raise ValueError("Unknown config key '{}' in {}".format(raw_key, config_path))
        defaults[dest] = value

    parser.set_defaults(**defaults)


def validate_required_args(parser, args, required_arg_names):
    for arg_name in required_arg_names:
        if getattr(args, arg_name) is None:
            parser.error("--{} is required unless provided by --config".format(arg_name.replace("_", "-")))


def get_train_args():
    config_path = parse_config_path()
    parser = argparse.ArgumentParser(description="AutoBots")
    parser.add_argument("--config", type=str, default=None, help="YAML config file with argument defaults.")
    # Section: General Configuration
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for saving results")

    # Section: Dataset
    parser.add_argument("--dataset", type=str, default=None, choices=["Argoverse", "Nuscenes", "trajnet++",
                                                                       "interaction-dataset", "bi3"],
                        help="Dataset to train on.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to dataset files.")
    parser.add_argument("--use-map-image", type=bool, default=False, help="Use map image if applicable.")
    parser.add_argument("--use-map-lanes", type=bool, default=False, help="Use map lanes if applicable.")

    # Section: Algorithm
    parser.add_argument("--model-type", type=str, default=None, choices=["Autobot-Joint", "Autobot-Ego"],
                        help="Whether to train for joint prediction or ego-only prediction.")
    parser.add_argument("--num-modes", type=int, default=5, help="Number of discrete latent variables for Autobot.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model's hidden size.")
    parser.add_argument("--num-encoder-layers", type=int, default=1,
                        help="Number of social-temporal layers in Autobot's encoder.")
    parser.add_argument("--num-decoder-layers", type=int, default=1,
                        help="Number of social-temporal layers in Autobot's decoder.")
    parser.add_argument("--tx-hidden-size", type=int, default=384,
                        help="hidden size of transformer layers' feedforward network.")
    parser.add_argument("--tx-num-heads", type=int, default=16, help="Transformer number of heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout strenght used throughout model.")

    # Section: Loss Function
    parser.add_argument("--entropy-weight", type=float, default=1.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--kl-weight", type=float, default=1.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--use-FDEADE-aux-loss", type=bool, default=True,
                        help="Whether to use FDE/ADE auxiliary loss in addition to NLL (accelerates learning).")

    # Section: Training params:
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam-epsilon", type=float, default=1e-4, help="Adam optimiser epsilon value")
    parser.add_argument("--learning-rate-sched", type=int, nargs='+', default=[5, 10, 15, 20],
                        help="Learning rate Schedule.")
    parser.add_argument("--grad-clip-norm", type=float, default=5, metavar="C", help="Gradient clipping norm")
    parser.add_argument("--num-epochs", type=int, default=150, metavar="I", help="number of iterations through the dataset.")
    apply_config_defaults(parser, config_path)
    args = parser.parse_args()
    validate_required_args(parser, args, ["dataset", "dataset_path", "model_type"])

    if args.use_map_image and args.use_map_lanes:
        raise Exception('We do not support having both the map image and the map lanes...')

    # Perform config checks
    if "trajnet" in args.dataset:
        assert "Joint" in args.model_type, "Can't run AutoBot-Ego on TrajNet..."
        assert not args.use_map_image and not args.use_map_lanes, "TrajNet++ has no scene map information..."
    elif "bi3" in args.dataset:
        assert not args.use_map_image and not args.use_map_lanes, "Bi3 has no scene map information..."
    elif "Argoverse" in args.dataset:
        assert "Ego" in args.model_type, "Can't run AutoBot-Joint on Argoverse..."
    elif "interaction-dataset" in args.dataset:
        assert "Ego" not in args.model_type, "Can't run AutoBot-Ego on Interaction Dataset..."
        assert not args.use_map_image, "Interaction-dataset does not have image-based scene information..."

    results_dirname = create_results_folder(args)
    save_config(args, results_dirname)

    return args, results_dirname


def get_eval_args():
    config_path = parse_config_path()
    parser = argparse.ArgumentParser(description="AutoBot")
    parser.add_argument("--config", type=str, default=None, help="YAML config file with argument defaults.")
    parser.add_argument("--models-path", type=str, default=None, help="Load model checkpoint")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path.")
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    apply_config_defaults(parser, config_path)
    args = parser.parse_args()
    validate_required_args(parser, args, ["models_path", "dataset_path"])

    config, model_dirname = load_config(args.models_path)
    config = namedtuple("config", config.keys())(*config.values())
    return args, config, model_dirname


def create_results_folder(args):
    model_configname = ""
    model_configname += "Autobot_joint" if "Joint" in args.model_type else "Autobot_ego"
    model_configname += "_C"+str(args.num_modes) + "_H"+str(args.hidden_size) + "_E"+str(args.num_encoder_layers)
    model_configname += "_D"+str(args.num_decoder_layers) + "_TXH"+str(args.tx_hidden_size) + "_NH"+str(args.tx_num_heads)
    model_configname += "_EW"+str(int(args.entropy_weight)) + "_KLW"+str(int(args.kl_weight))
    model_configname += "_NormLoss" if args.use_FDEADE_aux_loss else ""
    model_configname += "_roadImg" if args.use_map_image else ""
    model_configname += "_roadLanes" if args.use_map_lanes else ""
    if args.exp_id is not None:
        model_configname += ("_" + args.exp_id)
    model_configname += "_s"+str(args.seed)

    result_dirname = os.path.join(args.save_dir, "results", args.dataset, model_configname)
    if os.path.isdir(result_dirname):
        answer = input(result_dirname + " exists. \n Do you wish to overwrite? (y/n)")
        if 'y' in answer:
            if os.path.isdir(os.path.join(result_dirname, "tb_files")):
                for f in os.listdir(os.path.join(result_dirname, "tb_files")):
                    os.remove(os.path.join(result_dirname, "tb_files", f))
        else:
            exit()
    os.makedirs(result_dirname, exist_ok=True)
    return result_dirname


def save_config(args, results_dirname):
    argparse_dict = vars(args)
    with open(os.path.join(results_dirname, 'config.json'), 'w') as fp:
        json.dump(argparse_dict, fp)


def load_config(model_path):
    model_dirname = os.path.join(*model_path.split("/")[:-1])
    assert os.path.isdir(model_dirname)
    with open(os.path.join(model_dirname, 'config.json'), 'r') as fp:
        config = json.load(fp)
    return config, model_dirname
