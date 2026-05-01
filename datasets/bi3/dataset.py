import glob
import os

import numpy as np
from torch.utils.data import Dataset


def site_from_bi3_filename(filename):
    basename = os.path.basename(filename)
    if basename.startswith("bi3_laas"):
        return "laas"
    if basename.startswith("bi3_um"):
        return "um"
    return "unknown"


def agent_labels_for_scene(scene_idx):
    primary_idx = scene_idx % 3
    if primary_idx == 0:
        return ["Robot", "Human 1", "Human 2"]
    if primary_idx == 1:
        return ["Human 1", "Robot", "Human 2"]
    return ["Human 2", "Robot", "Human 1"]


class Bi3Dataset(Dataset):
    def __init__(self, dset_path, split_name="train", model_type="Autobot-Joint"):
        self.num_others = 2
        self.pred_horizon = 12
        self.num_agent_types = 2  # robot, human
        self.in_seq_len = 9
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr = 2
        self.use_joint_version = "Joint" in model_type

        split_path = os.path.join(dset_path, split_name)
        if not os.path.isdir(split_path):
            split_path = dset_path

        dset_fnames = sorted(
            fname for fname in glob.glob(os.path.join(split_path, "*.npy"))
            if not fname.endswith("_agent_types.npy")
        )
        if not dset_fnames:
            raise FileNotFoundError("No Bi3 npy files found in {}".format(split_path))

        agents_dataset = []
        agent_types_dataset = []
        scene_sites = []
        scene_agent_labels = []
        self.source_files = []
        for dset_fname in dset_fnames:
            agents_data = np.load(dset_fname)
            if len(agents_data) == 0:
                continue

            agent_types_fname = dset_fname[:-4] + "_agent_types.npy"
            if not os.path.isfile(agent_types_fname):
                raise FileNotFoundError("Missing Bi3 agent type file {}".format(agent_types_fname))

            agents_dataset.append(agents_data[:, :, :self.num_others + 1])
            agent_types_dataset.append(np.load(agent_types_fname)[:, :self.num_others + 1])
            scene_sites.extend([site_from_bi3_filename(dset_fname)] * len(agents_data))
            scene_agent_labels.extend(agent_labels_for_scene(scene_idx) for scene_idx in range(len(agents_data)))
            self.source_files.append(dset_fname)

        if not agents_dataset:
            raise RuntimeError("Bi3 split {} contains no valid scenes".format(split_name))

        self.agents_dataset = np.concatenate(agents_dataset)
        self.agent_types_dataset = np.concatenate(agent_types_dataset)
        self.scene_sites = np.array(scene_sites)
        self.scene_agent_labels = np.array(scene_agent_labels)

    def get_site(self, idx):
        return self.scene_sites[idx]

    def get_agent_labels(self, idx):
        return self.scene_agent_labels[idx].tolist()

    def __getitem__(self, idx: int):
        data = self.agents_dataset[idx]

        # Remove invalid values and add mask column to state.
        data_mask = np.ones((data.shape[0], data.shape[1], 3))
        data_mask[:, :, :2] = data
        invalid_indices = np.where(~np.isfinite(data[:, :, 0]) | ~np.isfinite(data[:, :, 1]))
        data_mask[invalid_indices] = [0, 0, 0]

        agents_in = data_mask[:self.in_seq_len]
        agents_out = data_mask[self.in_seq_len:]

        ego_in = agents_in[:, 0]
        ego_out = agents_out[:, 0]
        roads = np.ones((1, 1))  # for dataloading to work with datasets that have maps.

        if self.use_joint_version:
            agent_types = self.agent_types_dataset[idx]
            return ego_in, ego_out, agents_in[:, 1:], agents_out[:, 1:], roads, agent_types

        return ego_in, ego_out, agents_in[:, 1:], roads

    def __len__(self):
        return len(self.agents_dataset)
