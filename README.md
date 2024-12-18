# 4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching

[\[Preprint (arXiv)\]](http://arxiv.org/abs/----.-----)

## Requirements

- CUDA Toolkit (including nvcc)
- ROS 1 Noetic (only for reading NTU4DRadLM rosbags)
- PyTorch
- numpy
- scikit-learn
- matplotlib
- [small_gicp](https://github.com/koide3/small_gicp)
- [evo](https://github.com/MichaelGrupp/evo)

## Usage

This repository contains the following Python scripts:

- `run_odometry.py`: Runs the odometry system on the specified sequence.
- `evaluate.py`: Generates quantitative evaluation metrics for the specified method using `evo`.

Both scripts are configured using `config.ini`:

### `[config]`

General configuration.

- `dataset`: Specifies the name of the dataset used (currently only NTU4DRadLM is supported)

### `[odometry]`

Configuration specific to `run_odometry.py`.

- `sequence`: Specifies the name of the sequence used to run the odometry.
- `ablation_gicp`: Set to true if running the GICP ablated version (default is false)
- `num_particles`: Number of scan matching hypothesis particles. Set to 1 if running the single hypothesis ablated version (default is 4).
- `out_name`: Name of the output file. Default is odom_TIMESTAMP, where the timestamp is in YYYYMMDDhhmmss format.

### `[evaluation]`

Configuration specific to `evaluate.py`.

- `gt_pattern`: Filename pattern of ground truth trajectory files, relative to the dataset folder.
- `pred_pattern`: Filename pattern of generated odometry trajectory files.
- `method`: Name of the method to evaluate.
- `sequences`: Comma-separated list of sequences to evaluate.

The following placeholders are supported in filename patterns:

- `{method}`: Name of the method
- `{dataset}`: Name of the dataset
- `{seq}`: Name of the sequence within the dataset

## Reference

```
@misc{gaussian4drio,
	author = {Fernando Amodeo and Luis Merino and Fernando Caballero},
	title = {4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching},
	year = {2024},
	eprint = {arXiv:----.-----},
}
```

## Acknowledgements

This work was partially supported by the following grants: 1) INSERTION PID2021-127648OB-C31, and 2) NORDIC TED2021-132476B-I00 projects, funded by MCIN/AEI/ 10.13039/501100011033 and the "European Union NextGenerationEU / PRTR".
