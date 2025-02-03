# test_retest

## Overview
Code to generate Figures of Vrizzi, Najar et al. (preprint) available at https://osf.io/preprints/psyarxiv/3u4gp_v1 . This project provides a set of scripts and tools for statistical analysis and visualization of reliability measures (correlation and ICC) in a dataset from a reinforcement learning (RL) task. It includes functions for plotting figures, calculating reliability metrics, and summarizing statistical properties.

## RL parameter fitting methods
- ML=maximum likelihood
- MAP=maximum a posteriori
- HB=hierarchical bayesian
- HBpool=hierarchical bayesian by pooling together test-retest data

## Features
- Generation of various plots related to reliability measures
- Statistical summaries and analyses
- Support for multiple RL parameter estimation methods (ML, MAP, HB, HBpool)
- SLURM array job support for parallel execution

## Requirements

Ensure all dependencies are installed before running the script.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```
2. Set up a virtual environment (optional but recommended):
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, manually install necessary packages.

## Usage

Comment in or out the figures you want to plot. Launching the programme will automatically generate all the data files needed for the desired figures. If already there, they will be fetched.

### Running the Main Script
Execute the script using:
```sh
python main.py --task_id <TASK_ID>
```
- `TASK_ID` (optional): Specifies which RL parameter fitting method to use.
  - Default is `0` (maximum likelihood) if not provided.
  
Example:
```sh
python main.py --task_id 1
```

### Available Functions
#### Parameter-Free Plots
```python
parameter_free_plots(folder)
```
Generates plots for different reliability measures.

#### Parameter-Based Plots
```python
parameter_plots(method)
```
Generates plots based on a specific estimation method.

#### Statistical Summaries
```python
describe_stats(method)
describe_stats_reliability(method, reliability_measure)
describe_stats_alpha_con_disc(method)
```
Provides descriptive statistics and reliability analyses.

#### Extreme Upper Bound Analysis
```python
extreme_upperbound()
```
Performs an analysis on extreme cases.

## SLURM Job Support
This script supports execution as part of an SLURM array job using:
```sh
sbatch --array=0-3 job_script.sh
```
- The `task_id` is determined based on the SLURM environment variable.
- Methods mapped to IDs: `0: ML`, `1: MAP`, `2: HB`, `3: HBpool`.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-branch`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-branch`
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For questions or support, please contact [stefano.vrizzi@ens.psl.eu] or open an issue on GitHub.

