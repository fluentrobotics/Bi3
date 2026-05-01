import argparse
import json
import os
import random
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.bi3.dataset import Bi3Dataset
from datasets.interaction_dataset.dataset import InteractionDataset
from datasets.nuscenes.dataset import NuscenesH5Dataset
from datasets.trajnetpp.dataset import TrajNetPPDataset
from models.autobot_ego import AutoBotEgo
from models.autobot_joint import AutoBotJoint
from process_args import load_yaml_config


ROBOT_COLOR = "#fdae61"
HUMAN_COLOR = "#74add1"
FALLBACK_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def get_args():
    parser = argparse.ArgumentParser(description="Visualize one AutoBots prediction.")
    parser.add_argument("--config", type=str, default=None, help="Optional eval YAML config.")
    parser.add_argument("--models-path", type=str, default=None, help="Trained model checkpoint.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset root. Defaults to the model config path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to sample from.")
    parser.add_argument("--index", type=int, default=None, help="Dataset index. Random if omitted.")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output image path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --index is omitted.")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after saving.")
    args = parser.parse_args()

    if args.config is not None:
        config = load_yaml_config(args.config)
        for key, value in config.items():
            dest = key.replace("-", "_")
            if dest == "eval_split":
                dest = "split"
            if hasattr(args, dest) and getattr(args, dest) is None:
                setattr(args, dest, value)
            elif dest in ["split", "output", "seed", "disable_cuda"] and getattr(args, dest) == parser.get_default(dest):
                setattr(args, dest, value)

    if args.models_path is None:
        parser.error("--models-path is required unless provided by --config")
    return args


def get_dataset(dataset_name, dataset_path, split_name, model_type, config):
    if "Nuscenes" in dataset_name:
        return NuscenesH5Dataset(dset_path=dataset_path, split_name=split_name, model_type=model_type,
                                 use_map_img=config.use_map_image, use_map_lanes=config.use_map_lanes)
    if "interaction-dataset" in dataset_name:
        return InteractionDataset(dset_path=dataset_path, split_name=split_name,
                                  use_map_lanes=config.use_map_lanes, evaluation=False)
    if "trajnet++" in dataset_name:
        return TrajNetPPDataset(dset_path=dataset_path, split_name=split_name)
    if "bi3" in dataset_name:
        return Bi3Dataset(dset_path=dataset_path, split_name=split_name, model_type=model_type)
    if "Argoverse" in dataset_name:
        return ArgoH5Dataset(dset_path=dataset_path, split_name=split_name,
                             use_map_lanes=config.use_map_lanes)
    raise NotImplementedError("Unsupported dataset {}".format(dataset_name))


def load_model_config(models_path):
    model_dirname = os.path.dirname(os.path.abspath(models_path))
    config_path = os.path.join(model_dirname, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Could not find model config at {}".format(config_path))
    with open(config_path, "r") as fp:
        config = json.load(fp)
    return config


def bi3_site_npy_files(dataset_path, site):
    root = Path(dataset_path)
    files = []
    search_roots = [root / split for split in ["train", "val", "test"] if (root / split).is_dir()]
    if not search_roots:
        search_roots = [root]

    for search_root in search_roots:
        files.extend(
            path for path in search_root.glob("bi3_{}*.npy".format(site))
            if not path.name.endswith("_agent_types.npy")
        )
    return sorted(files)


def square_bounds_from_files(npy_files):
    mins = []
    maxs = []
    for npy_file in npy_files:
        data = np.load(npy_file)
        valid = np.isfinite(data[:, :, :, 0]) & np.isfinite(data[:, :, :, 1])
        if not valid.any():
            continue
        points = data[:, :, :, :2][valid]
        mins.append(points.min(axis=0))
        maxs.append(points.max(axis=0))

    if not mins:
        return None

    min_xy = np.min(np.stack(mins), axis=0)
    max_xy = np.max(np.stack(maxs), axis=0)
    center = (min_xy + max_xy) / 2.0
    span = float(np.max(max_xy - min_xy))
    if span <= 0:
        span = 1.0
    half_span = span * 0.55
    return (
        center[0] - half_span,
        center[0] + half_span,
        center[1] - half_span,
        center[1] + half_span,
    )


def plotted_square_bounds(history, future, predictions):
    arrays = [
        history[:, :, :2].reshape(-1, 2),
        future[:, :, :2].reshape(-1, 2),
        predictions.reshape(-1, 2),
    ]
    points = np.concatenate(arrays)
    points = points[np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])]
    if len(points) == 0:
        return (-1, 1, -1, 1)

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = float(np.max(max_xy - min_xy))
    if span <= 0:
        span = 1.0
    half_span = span * 0.55
    return (
        center[0] - half_span,
        center[0] + half_span,
        center[1] - half_span,
        center[1] + half_span,
    )


def fixed_bounds(dataset_name, dataset_path, site):
    if "bi3" not in dataset_name or site not in ["um", "laas"]:
        return None
    return square_bounds_from_files(bi3_site_npy_files(dataset_path, site))


def build_model(config, dataset, device):
    if "Ego" in config.model_type:
        model = AutoBotEgo(k_attr=dataset.k_attr,
                           d_k=config.hidden_size,
                           _M=dataset.num_others,
                           c=config.num_modes,
                           T=dataset.pred_horizon,
                           L_enc=config.num_encoder_layers,
                           dropout=config.dropout,
                           num_heads=config.tx_num_heads,
                           L_dec=config.num_decoder_layers,
                           tx_hidden_size=config.tx_hidden_size,
                           use_map_img=config.use_map_image,
                           use_map_lanes=config.use_map_lanes,
                           map_attr=dataset.map_attr)
    elif "Joint" in config.model_type:
        model = AutoBotJoint(k_attr=dataset.k_attr,
                             d_k=config.hidden_size,
                             _M=dataset.num_others,
                             c=config.num_modes,
                             T=dataset.pred_horizon,
                             L_enc=config.num_encoder_layers,
                             dropout=config.dropout,
                             num_heads=config.tx_num_heads,
                             L_dec=config.num_decoder_layers,
                             tx_hidden_size=config.tx_hidden_size,
                             use_map_lanes=config.use_map_lanes,
                             map_attr=dataset.map_attr,
                             num_agent_types=dataset.num_agent_types,
                             predict_yaw=dataset.predict_yaw)
    else:
        raise NotImplementedError("Unsupported model type {}".format(config.model_type))

    model_dicts = torch.load(config.models_path, map_location=device)
    model.load_state_dict(model_dicts["AutoBot"])
    model.to(device)
    model.eval()
    return model


def to_batch_tensor(array, device):
    return torch.as_tensor(array).unsqueeze(0).float().to(device)


def run_model(model, item, model_type, device):
    if "Joint" in model_type:
        ego_in, ego_out, agents_in, agents_out, roads, agent_types = item[:6]
        with torch.no_grad():
            pred, mode_probs = model(to_batch_tensor(ego_in, device),
                                     to_batch_tensor(agents_in, device),
                                     to_batch_tensor(roads, device),
                                     to_batch_tensor(agent_types, device))
        history = np.concatenate([ego_in[:, np.newaxis], agents_in], axis=1)
        future = np.concatenate([ego_out[:, np.newaxis], agents_out], axis=1)
        predictions = pred[:, :, 0, :, :2].cpu().numpy()
    else:
        ego_in, ego_out, agents_in, roads = item[:4]
        with torch.no_grad():
            pred, mode_probs = model(to_batch_tensor(ego_in, device),
                                     to_batch_tensor(agents_in, device),
                                     to_batch_tensor(roads, device))
        history = np.concatenate([ego_in[:, np.newaxis], agents_in], axis=1)
        future = ego_out[:, np.newaxis]
        predictions = pred[:, :, 0, :2].cpu().numpy()[:, :, np.newaxis]

    return history, future, predictions, mode_probs[0].cpu().numpy()


def finite_xy(points):
    return np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])


def mode_alpha(probability, max_prob):
    if max_prob <= 0:
        return 1.0
    return min(1.0, max(0.3, 0.3 + 0.7 * (probability / max_prob)))


def agent_color(label, agent_idx):
    if label.startswith("Robot"):
        return ROBOT_COLOR
    if label.startswith("Human"):
        return HUMAN_COLOR
    return FALLBACK_COLORS[agent_idx % len(FALLBACK_COLORS)]


def default_agent_labels(num_agents):
    return ["Agent {}".format(agent_idx) for agent_idx in range(num_agents)]


def plot_prediction(history, future, predictions, mode_probs, output_path, title, agent_labels=None, bounds=None):
    if agent_labels is None:
        agent_labels = default_agent_labels(history.shape[1])
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_labels = set()

    for agent_idx in range(history.shape[1]):
        agent_label = agent_labels[agent_idx] if agent_idx < len(agent_labels) else "Agent {}".format(agent_idx)
        color = agent_color(agent_label, agent_idx)
        hist_mask = finite_xy(history[:, agent_idx])
        if hist_mask.any():
            label = "{} History".format(agent_label)
            ax.plot(history[hist_mask, agent_idx, 0], history[hist_mask, agent_idx, 1],
                    color=color, linewidth=2.5, marker="o", markersize=3,
                    label=label if label not in legend_labels else None)
            legend_labels.add(label)
            ax.scatter(history[hist_mask, agent_idx, 0][-1], history[hist_mask, agent_idx, 1][-1],
                       color=color, marker="s", s=35)

        if agent_idx < future.shape[1]:
            fut_mask = finite_xy(future[:, agent_idx])
            if fut_mask.any():
                label = "{} GT".format(agent_label)
                ax.plot(future[fut_mask, agent_idx, 0], future[fut_mask, agent_idx, 1],
                        color=color, linestyle="--", linewidth=2,
                        label=label if label not in legend_labels else None)
                legend_labels.add(label)

    for agent_idx in range(predictions.shape[2]):
        agent_label = agent_labels[agent_idx] if agent_idx < len(agent_labels) else "Agent {}".format(agent_idx)
        color = agent_color(agent_label, agent_idx)
        agent_max_prob = float(np.max(mode_probs))
        for mode_idx, probability in enumerate(mode_probs):
            alpha = mode_alpha(float(probability), agent_max_prob)
            ax.plot(predictions[mode_idx, :, agent_idx, 0], predictions[mode_idx, :, agent_idx, 1],
                    color=color, alpha=alpha, linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if bounds is None:
        bounds = plotted_square_bounds(history, future, predictions)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    return fig


def main():
    args = get_args()
    config_dict = load_model_config(args.models_path)
    config_dict["models_path"] = args.models_path
    config = namedtuple("config", config_dict.keys())(*config_dict.values())

    dataset_path = args.dataset_path or config.dataset_path
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

    dataset = get_dataset(config.dataset, dataset_path, args.split, config.model_type, config)
    if len(dataset) == 0:
        raise RuntimeError("{} split is empty".format(args.split))

    index = args.index
    if index is None:
        index = random.randrange(len(dataset))
    if index < 0 or index >= len(dataset):
        raise IndexError("Index {} is outside dataset length {}".format(index, len(dataset)))

    item = dataset[index]
    model = build_model(config, dataset, device)
    history, future, predictions, mode_probs = run_model(model, item, config.model_type, device)
    site = dataset.get_site(index) if hasattr(dataset, "get_site") else None
    agent_labels = dataset.get_agent_labels(index) if hasattr(dataset, "get_agent_labels") else None
    bounds = fixed_bounds(config.dataset, dataset_path, site)
    title = "{} {} index {}".format(config.dataset, config.model_type, index)
    if site is not None:
        title += " ({})".format(site)
    plot_prediction(history, future, predictions, mode_probs, args.output, title,
                    agent_labels=agent_labels, bounds=bounds)
    print("Saved {} for {} split index {}".format(args.output, args.split, index))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
