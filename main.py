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

import scipy.io
import numpy as np

from mesh import Mesh
from bem import BEM
from laplace_interpolation import Interpolate
from spline_parameters import spline_parameters
from spline import SPLINE
from validation_errors import ERRORS
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


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
            log.info("Data loaded")
        else:
            log.error("Data not loaded")
            exit(1)
        print(data)
        print(data.files)
        for d in data:
            print(data[d].shape)

        mesh_torso_ = Mesh(data["torso_faces"].T.astype(int), data["torso_vertices"].T, np.array([0, 10]))
        mesh_heart = Mesh(data["heart_faces"].T.astype(int), data["heart_vertices"].T, np.array([10, 10]))
        tpotentials = data["torso_measures"]

        # Apply BEM # Remove tpotentials
        print("shape of faces=",data["torso_faces"].shape)
        bem = BEM([mesh_torso_, mesh_heart], tpotentials)
        transfer_matrix = bem.calc_transfer_matrix()
        #np.savez("transfer_matrix",transfer_matrix)
        #mmat = {"transfer_matrix": transfer_matrix}
        #scipy.io.savemat("tmat.mat",mmat)
        log.info("Shape tranfer matrix: %s", transfer_matrix.shape)
        log.info("Tranfer matrix: \n %s", transfer_matrix)

        # If length of potentials is != length of torso_vertices, interpolate
        if (len(tpotentials) != len(data["torso_vertices"])):
            log.info("Performing interpolation")
            closed_mesh = self.config['path_closed_mesh']
            measured_mesh = self.config['path_measured_mesh']
            interpolate = Interpolate(closed_mesh, measured_mesh, tpotentials)
            closed_tpotentials = interpolate.calc_interpolation()
            print("in main", closed_tpotentials)
        else:
            closed_tpotentials = tpotentials
        
        #cpmat={"closed_torso_potentials": closed_tpotentials}
        #scipy.io.savemat("771torsopots.mat",cpmat)
        # Apply Spline

        input_spline = spline_parameters(
            self.config['number_of_knots'],self.config['interpolation_density'],self.config['minimum_derivatives_cost_reduction'],
            self.config['minimum_overall_cost_reduction'],self.config['projection_interpolation_density'],
            torso_potentials = closed_tpotentials)
        spline = SPLINE(input_spline, transfer_matrix)
        heart_potentials = spline.spline_inverse()

        return heart_potentials
        log.info("Heart potentials: \n %s", heart_potentials)

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
    expected_heart_potentials = sp_inv.run()
    np.savez("heart_expected_mod",expected_heart_potentials)

    hpotentials = scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Interventions/Control/Cage/rsm15may02-cage-0003.mat")
    observed_heart_potentials = hpotentials['ts']['potvals'][0,0]

    temporal_correlation = np.empty(observed_heart_potentials.shape[1])
    temporal_rmse = np.empty(observed_heart_potentials.shape[1])
    temporal_smap = np.empty(observed_heart_potentials.shape[1])
    for t in np.arange(observed_heart_potentials.shape[1]):
        errors = ERRORS(observed_heart_potentials[:,t], expected_heart_potentials[:,t])
        #temporal_smap[t], temporal_rmse[t], temporal_cross_correlation[t] = errors.combined_errors()
        temporal_correlation[t] = errors.calculate_correlation()
        temporal_rmse[t] = errors.calculate_rmse()
        temporal_smap[t] = errors.calculate_smape()


    spatial_correlation = np.empty(observed_heart_potentials.shape[0])
    spatial_rmse = np.empty(observed_heart_potentials.shape[0])
    spatial_smap = np.empty(observed_heart_potentials.shape[0])
    for n in np.arange(observed_heart_potentials.shape[0]):
        errors = ERRORS(observed_heart_potentials[n,:], expected_heart_potentials[n,:])
        #spatial_smap[n], spatial_rmse[n], spatial_cross_correlation[n] = errors.calculate_cross_correlation()
        spatial_rmse[n]        = errors.calculate_rmse()
        spatial_smap[n]        = errors.calculate_smape()
        spatial_correlation[t] = errors.calculate_correlation()

    fig = plt.figure()
    obs = fig.add_subplot(4, 2, 1, projection = '3d')
    obs.set_title("Observed heart potentials")
    obs.set_xlabel("time")
    obs.set_ylabel("node")
    x = np.arange(observed_heart_potentials.shape[1])
    y = np.arange(observed_heart_potentials.shape[0])
    x, y = np.meshgrid(x, y)
    z = observed_heart_potentials
    obs.plot_surface(x, y, z)

    exp = fig.add_subplot(4, 2, 2, projection = '3d')
    exp.set_title("Expected heart potentials")
    exp.set_xlabel("time")
    exp.set_ylabel("node")
    x = np.arange(expected_heart_potentials.shape[1])
    y = np.arange(expected_heart_potentials.shape[0])
    x, y = np.meshgrid(x, y)
    z = expected_heart_potentials
    exp.plot_surface(x, y, z)

    temp_rmse = fig.add_subplot(4, 2, 3)
    temp_rmse.set_title("RMSE temporal")
    temp_rmse.plot(temporal_rmse)

    spa_rmse = fig.add_subplot(4, 2, 4)
    spa_rmse.set_title("RMSE spatial")
    spa_rmse.plot(spatial_rmse)

    temp_cc = fig.add_subplot(4, 2, 5)
    temp_cc.set_title("Temporal Correlation")
    temp_cc.plot(temporal_correlation)

    spa_cc = fig.add_subplot(4, 2, 6)
    spa_cc.set_title("Spatial correlaion")
    spa_cc.plot(spatial_correlation)

    temp_smap = fig.add_subplot(4, 2, 7)
    temp_smap.set_title("SMAP temporal")
    temp_smap.plot(temporal_smap)

    spa_smap = fig.add_subplot(4, 2, 8)
    spa_smap.set_title("SMAP spatial")
    spa_smap.plot(spatial_smap)
    
    plt.show()
    #plt.savefig('foo.png')
