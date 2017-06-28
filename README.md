# Empirical LSM

A framework for running empirical land surface models.

## Requirements

empirical_lsm requires the `pals_utils` library, available from https://bitbucket.org/naught101/pals_utils

Models fluxnet data as input in the form of netcdf files. Appropriate netcdf files can be obtained from PALS (https://modelevaluation.org), or using the [FluxnetLSM](https://github.com/aukkola/FluxnetLSM) R package to convert csv files from Fluxdata.org. 

## Usage

See scripts in `scripts/offline_runs`.

Data directories can be set using the `pals_utils.data.set_config` function.
