import argparse
import json
import math
from pathlib import Path

import numpy as np


SPLIT_EXPERIMENTS = {
    "train": {
        "um": range(1, 14),
        "laas": range(1, 15),
    },
    "val": {
        "um": range(14, 16),
        "laas": range(15, 18),
    },
    "test": {
        "um": range(16, 18),
        "laas": range(18, 21),
    },
}

RAW_AGENT_TYPES = np.array(
    [
        [1.0, 0.0],  # robot
        [0.0, 1.0],  # human 1
        [0.0, 1.0],  # human 2
    ],
    dtype=np.float32,
)


def get_args():
    parser = argparse.ArgumentParser(description="Bi3 TrajNet++/AutoBots NPY Creator")
    parser.add_argument("--bi3-root", type=str, default="/home/socnav/Desktop/Bi3",
                        help="Root of the Bi3 dataset.")
    parser.add_argument("--output-root", type=str, default="/home/socnav/Desktop/Bi3/trajectory_prediction",
                        help="Output root for split subdirectories.")
    parser.add_argument("--obs-len", type=int, default=9, help="Number of observed timesteps.")
    parser.add_argument("--pred-len", type=int, default=12, help="Number of future timesteps.")
    parser.add_argument("--dt", type=float, default=0.4, help="Downsampled timestep in seconds.")
    parser.add_argument("--window-stride", type=int, default=1,
                        help="Window stride in downsampled timesteps.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files if present.")
    return parser.parse_args()


def split_for_experiment(site, experiment_idx):
    for split_name, split_spec in SPLIT_EXPERIMENTS.items():
        if experiment_idx in split_spec.get(site, range(0)):
            return split_name
    return None


def parse_source_path(json_root, json_path):
    rel_path = json_path.relative_to(json_root)
    site = rel_path.parts[0]
    experiment = rel_path.parts[1]
    condition = json_path.stem
    experiment_idx = int(experiment.replace(site, ""))
    return site, experiment, experiment_idx, condition


def rotation_matrix(theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([[ct, st], [-st, ct]])


def rotate_scene(window, primary_idx, obs_len):
    order = [primary_idx] + [idx for idx in range(window.shape[1]) if idx != primary_idx]
    ordered_window = window[:, order]
    center = ordered_window[obs_len - 1, 0]

    diff = ordered_window[obs_len - 1, 0] - ordered_window[obs_len - 2, 0]
    theta = math.atan2(diff[1], diff[0])
    rotation = -theta + math.pi / 2.0

    rotated_window = (ordered_window - center[np.newaxis, np.newaxis, :]).dot(rotation_matrix(rotation))
    return rotated_window.astype(np.float32), RAW_AGENT_TYPES[order]


def nearest_downsample(times, positions, dt):
    target_times = np.arange(times[0], times[-1] + dt * 0.5, dt)
    target_times = target_times[target_times <= times[-1]]
    next_indices = np.searchsorted(times, target_times, side="left")
    next_indices = np.clip(next_indices, 0, len(times) - 1)
    prev_indices = np.maximum(next_indices - 1, 0)

    use_prev = np.abs(times[prev_indices] - target_times) <= np.abs(times[next_indices] - target_times)
    nearest_indices = next_indices
    nearest_indices[use_prev] = prev_indices[use_prev]
    return target_times, positions[nearest_indices]


def load_bi3_positions(json_path):
    with json_path.open("r") as fp:
        timesteps = json.load(fp)

    times = np.zeros(len(timesteps), dtype=np.float64)
    positions = np.zeros((len(timesteps), 3, 2), dtype=np.float32)
    positions[:] = np.nan

    for idx, timestep in enumerate(timesteps):
        times[idx] = float(timestep["time"])
        positions[idx, 0] = timestep["robot_state"][:2]
        agent_states = timestep["agent_states"]
        if len(agent_states) >= 2:
            positions[idx, 1] = agent_states[0][:2]
            positions[idx, 2] = agent_states[1][:2]

    sort_indices = np.argsort(times)
    return times[sort_indices], positions[sort_indices]


def make_scenes(sampled_positions, obs_len, pred_len, window_stride):
    traj_len = obs_len + pred_len
    scenes = []
    agent_types = []
    skipped_windows = 0
    window_count = 0

    for start_idx in range(0, len(sampled_positions) - traj_len + 1, window_stride):
        window_count += 1
        window = sampled_positions[start_idx:start_idx + traj_len]
        if not np.isfinite(window).all():
            skipped_windows += 1
            continue

        for primary_idx in range(window.shape[1]):
            rotated_scene, rotated_agent_types = rotate_scene(window, primary_idx, obs_len)
            scenes.append(rotated_scene)
            agent_types.append(rotated_agent_types)

    if scenes:
        return np.stack(scenes), np.stack(agent_types), window_count, skipped_windows

    return (
        np.empty((0, traj_len, 3, 2), dtype=np.float32),
        np.empty((0, 3, 2), dtype=np.float32),
        window_count,
        skipped_windows,
    )


def write_ndjson(output_path, scenes, fps):
    traj_len = scenes.shape[1]
    with output_path.open("w") as fp:
        for scene_id, scene in enumerate(scenes):
            first_frame = scene_id * traj_len
            last_frame = first_frame + traj_len - 1
            scene_row = {
                "scene": {
                    "id": scene_id,
                    "p": 0,
                    "s": first_frame,
                    "e": last_frame,
                    "fps": fps,
                    "tag": [0, ["bi3"]],
                }
            }
            fp.write(json.dumps(scene_row) + "\n")

            for timestep_idx in range(traj_len):
                frame = first_frame + timestep_idx
                for agent_idx in range(scene.shape[1]):
                    track_row = {
                        "track": {
                            "f": frame,
                            "p": agent_idx,
                            "x": round(float(scene[timestep_idx, agent_idx, 0]), 6),
                            "y": round(float(scene[timestep_idx, agent_idx, 1]), 6),
                        }
                    }
                    fp.write(json.dumps(track_row) + "\n")


def prepare_output_dirs(output_root, overwrite):
    output_root.mkdir(parents=True, exist_ok=True)
    existing_outputs = []
    for split_name in SPLIT_EXPERIMENTS:
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        existing_outputs.extend(split_dir.glob("*.npy"))
        existing_outputs.extend(split_dir.glob("*.ndjson"))

    if existing_outputs and not overwrite:
        raise FileExistsError("Generated files already exist under {}; pass --overwrite.".format(output_root))

    if overwrite:
        for output_path in existing_outputs:
            output_path.unlink()


def convert_file(json_root, json_path, output_root, obs_len, pred_len, dt, window_stride):
    site, experiment, experiment_idx, condition = parse_source_path(json_root, json_path)
    split_name = split_for_experiment(site, experiment_idx)
    if split_name is None:
        return None

    times, positions = load_bi3_positions(json_path)
    _, sampled_positions = nearest_downsample(times, positions, dt)
    scenes, agent_types, window_count, skipped_windows = make_scenes(
        sampled_positions, obs_len, pred_len, window_stride
    )

    output_stem = "bi3_{}_{}".format(experiment, condition)
    split_dir = output_root / split_name
    npy_path = split_dir / "{}.npy".format(output_stem)
    agent_types_path = split_dir / "{}_agent_types.npy".format(output_stem)
    ndjson_path = split_dir / "{}.ndjson".format(output_stem)

    np.save(npy_path, scenes)
    np.save(agent_types_path, agent_types)
    write_ndjson(ndjson_path, scenes, fps=1.0 / dt)

    return {
        "source": str(json_path.relative_to(json_root.parent)),
        "split": split_name,
        "window_count": int(window_count),
        "skipped_windows": int(skipped_windows),
        "scene_count": int(len(scenes)),
    }


def prepare_data(bi3_root, output_root, obs_len, pred_len, dt, window_stride, overwrite):
    bi3_root = Path(bi3_root)
    json_root = bi3_root / "jsons"
    output_root = Path(output_root)

    if not json_root.is_dir():
        raise NotADirectoryError("Bi3 JSON directory not found: {}".format(json_root))
    if window_stride < 1:
        raise ValueError("--window-stride must be >= 1")

    prepare_output_dirs(output_root, overwrite)

    split_counts = {
        split_name: {"source_json_count": 0, "scene_count": 0, "skipped_windows": 0}
        for split_name in SPLIT_EXPERIMENTS
    }
    json_paths = sorted(json_root.glob("*/*/*.json"))
    for json_path in json_paths:
        file_summary = convert_file(json_root, json_path, output_root, obs_len, pred_len, dt, window_stride)
        if file_summary is None:
            continue

        split_summary = split_counts[file_summary["split"]]
        split_summary["source_json_count"] += 1
        split_summary["scene_count"] += file_summary["scene_count"]
        split_summary["skipped_windows"] += file_summary["skipped_windows"]
        print(
            "{}: {} scenes from {} windows".format(
                file_summary["source"], file_summary["scene_count"], file_summary["window_count"]
            )
        )

    for split_name, split_summary in split_counts.items():
        print(
            "{}: {} files, {} scenes, {} skipped windows".format(
                split_name,
                split_summary["source_json_count"],
                split_summary["scene_count"],
                split_summary["skipped_windows"],
            )
        )


if __name__ == "__main__":
    args = get_args()
    prepare_data(
        bi3_root=args.bi3_root,
        output_root=args.output_root,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        dt=args.dt,
        window_stride=args.window_stride,
        overwrite=args.overwrite,
    )
