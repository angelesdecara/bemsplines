"""
Author: SpikaTech
Date: 01/02/2023
Execution of the file: python -m src.inverse_problem.spline_inverse.main -c <config_file_path>
Description:
Spline inverse problem
"""
import argparse
import json
import logging

import numpy as np

from mesh import Mesh
from bem import BEM


class Spline_inverse:
    def __init__(self, dir_config):
        # Reading the config json file
        with open(dir_config) as f:
            config = json.load(f)

        self.config = config

    def run(self):
        # Read data
        data = np.load(self.config["path_data"])
        if data:
            logging.info("Data loaded")
        else:
            logging.error("Data not loaded")
            exit(1)
        print(data)
        print(data.files)
        for d in data:
            print(data[d].shape)

        mesh_torso_ = Mesh(data["torso_faces"], data["torso_vertices"], np.array([0, 1]))
        mesh_heart = Mesh(data["heart_faces"], data["heart_vertices"], np.array([1, 10]))
        tpotentials = data["torso_measures"]

        # Apply BEM
        bem = BEM([mesh_torso_, mesh_heart], tpotentials)
        transfer_matrix = bem.calc_transfer_matrix()
        print(transfer_matrix.shape)
        np.savez("data/tmutah.npz",ForwardMatrix=transfer_matrix)

        # Apply Spline
        # spline(transfer_matrix)  # TODO: do this function


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize the argument parser
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config",
        "-c",
        default="utils/config.json",
        help="Add the config file path after this flag",
    )
    args = argparse.parse_args()

    # Argument variables
    arg_config = args.config

    print("Initial args:")
    print(f"- Config: {arg_config}")

    sp_inv = Spline_inverse(arg_config)
    sp_inv.run()
