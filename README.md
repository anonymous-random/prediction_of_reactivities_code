# Using Machine Learning to Predict Individual Differences in Psychological Reactivities to Social Interactions and Major Societal Events 

This repository contains the complete analysis code.


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Cloning the Repository](#cloning-the-repository)
   - [Get the Data](#get-the-data)
   - [Installing Python](#installing-python)
   - [Installing Requirements](#installing-requirements)
3. [Usage](#usage)
   - [Main Function](#main-function)
   - [Main Config](#main-config)
   - [Computation of SHAP Interaction Values](#computation-of-shap-interaction-values)
   - [Speed up Computations](#speed-up-computations)
   - [Troubleshooting](#troubleshooting)
4. [Reproducing Results](#reproducing-results)
5. [Project Structure](#project-structure)

## Introduction

To reproduce the results, please execute the following steps:
- Clone the repository
- Download the data 
- Install Python 
- Install the requirements 
- Run the analyses as described below

## Installation

### Cloning the Repository

To begin working with the project, you first need to copy it to your local machine. This process is called "cloning". There are two ways to clone the repository: using the command line or using a graphical user interface (GUI).

#### Using the Command Line
If you are comfortable with the command line, you can use the following commands:

```bash
git clone https://github.com/your-username/psychological_reactivities_analysis.git
cd psychological_reactivities_analysis
```

#### Cloning via GitHub Website

1. **Navigate to the Repository**:
   - Open your web browser and go to the repository's page on GitHub. Use the URL: `https://github.com/your-username/psychological_reactivities_analysis`

2. **Clone the Repository**:
   - Above the file list, click the green button labeled **Code**.
   - To clone the repository using HTTPS, under "Clone with HTTPS", click the clipboard icon to copy the repository URL.
   - If you’re prompted, sign in to your GitHub account.

3. **Download and Extract the Repository**:
   - After copying the URL, you can download the repository as a ZIP file to your computer.
   - Click the **Download ZIP** button from the dropdown menu under the **Code** button.
   - Once the download is complete, extract the ZIP file to your desired location on your computer to start working with the project files.

This method does not require any special software and is perfect for those unfamiliar with command-line tools. You will have a complete copy of the repository files, ready to be used with any code editor of your choice.

### Get the Data 

1. **Download the data**:
   - Navigate to the OSF project that contains the data by clicking on the link provided in the manuscript. Download the zip file **data**.  

2. **Unzip the data**:
   - Unzip the **data** file without changing its structure.  

3. **Paste the data in the repository**:
   - Paste the unzipped **data** folder in the repository. Ensure that the data folder is placed directly within the main repository directory as shown below. 
```plaintext
psychological_reactivities_analysis/
│
├── configs/
├── data/
├── ...
```

### Installing Python

1. **Visit the Python Downloads Page**:
   - Navigate to the Python downloads page for version 3.10.8 by clicking the following link: [Python 3.10.8 Download](https://www.python.org/downloads/release/python-3108/). This page contains installers for various operating systems including Windows, macOS, and Linux.

2. **Select the Appropriate Installer**:
   - Choose the installer that corresponds to your operating system. If you are using Windows, you might need to decide between the 32-bit and 64-bit versions. Most modern computers will use the 64-bit version.

3. **Run the Installer**:
   - After downloading the installer, run it by double-clicking the file. Ensure that you check the box labeled "Add Python 3.10.8 to PATH" before clicking "Install Now". This option sets up Python in your system's PATH, making it accessible from the command line.

4. **Verify the Installation**:
   - To confirm that Python has been installed correctly, open your command line interface (Terminal for macOS and Linux, Command Prompt for Windows) and type the following command:
     ```bash
     python --version
     ```
   - This command should return "Python 3.10.8". If it does not, you may need to restart your terminal or computer.


### Installing Requirements 
To ensure your setup is correctly configured to run the code, follow these steps to install the necessary dependencies:

1. **Open your terminal**: Before proceeding, make sure you are in the project's root directory.

2. **Check your Python installation**: Ensure that Python 3.10.8 is installed. If not, download and install it from [Python's official download page](https://www.python.org/downloads/release/python-3108/).

3. **Set up a virtual environment (recommended)**: To avoid conflicts with other Python projects, create a virtual environment by running:

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:
    - On Windows, use:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux, use:
        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages**: Install all dependencies listed in the `requirements.txt` file by running:

    ```bash
    pip install -r requirements.txt
    ```

Following these steps will prepare your environment for running the project without any issues related to dependencies.

## Usage

This repository contains several scripts to perform a sequence of analyses. The `main.py` script coordinates all the tasks, while the configurations in `config_refactored.yaml` determine which specific analyses are executed with which parameters. Thus, the config is the only file that need to be changed when using this repository. 

### Main Function

The `main.py` script is the primary entry point to run all analyses in this repository.
Tu run the main function, do the following 
- In Editor (e.g., Pycharm): Set up a Run Configuration with the virtual environment and the main script of this project and run main. 
- In Terminal (e.g., Bash, Powershell): Just run the script using `python src/main.py`


### Main Config 

The `config_refactored.yaml` is the user interface for this repository. Everything that is needed to reproduce the results
must be adjusted in this config file. It determines 
- which analysis are executed, denoted by the boolean value behind the analyses 
- which are the parameters of the analyses 
To reproduce the results, please don't change any parameters except the booleans in line 8-16 for choosing different analyses. 

To specify which analyses to perform, only adjust the settings in lines 8-16 in `config_refactored.yaml`. The available analyses are:

- `preprocessing`
- `calc_mlms`
- `ml_analysis`
- `result_analysis`
- `prelim_results_analysis`
- `significance_tests`
- `cv_results_plots`
- `shap_value_analysis`
- `osf_suppl_analysis`

Analyses set to `True` in the configuration file will be performed when this script is executed. The order of execution matters, as some analyses depend on the results of previous steps. For instance, data preprocessing must be completed before running machine learning analyses, and random effects must be computed before running machine learning models.

**Recommended Execution Workflow:**

1. Preprocess the data for all analysis settings.
2. Extract random effects for all analysis settings.
3. Run the machine learning analysis (depending on the specific configuration, this may be time-consuming (e.g., for the non-linear ML models and the computation of SHAP interaction values))
4. Conduct subsequent analyses as needed, in chronological order

### Computation of SHAP Interaction Values 

As we have described in the paper, we did only compute the SHAP interaction values for specific analysis settings. 
With the following description, SHAP interaction values for every analysis setting can be computed. 

Modify the following line in the configuration file to set up the main analysis parameters (line 715, set this to "true"). This is only implemented for the random forest regression (rfr)
- `calc_ia_values: true`

Do the computations as described in the next section. If `calc_ia_values: true`, the SHAP interaction values will be computed and stored during the machine-learning-based analysis. This may be very time-consuming. 

### Speed up Computations

To speed up the computations, the machine learning-based analysis can be parallelized (on a local computer). As a default, the analysis is not parallelized.
Therefore, lines 702-713 in `config_refactored.yaml` need to be adjusted to parallelize the computations. 
I recommend these settings for local parallelization where the specific number depends on the cores available at the local device. 

- `parallelize_reps: false`
- `reps_n_jobs: null`
- `parallelize_shap: true`
- `shap_n_jobs: 5`
- `parallelize_shap_outer_cv: false`
- `shap_outer_cv_n_jobs: null`
- `parallelize_rfe: false`
- `rfe_n_jobs: null`
- `parallelize_inner_cv: true`
- `inner_cv_n_jobs: 5`
- `parallelize_shap_ia_values: false`
- `shap_ia_values_n_jobs: null`

### Troubleshooting 

If main.py is not running as expected, consider the following steps 
- Check if you installed a virtual environment. Consider re-installing the virtual environment 
- Check if you installed the `requirements.txt` file 
- Try running the code in an editor, if it does not work from the terminal 
- Check the python version and the python path 
- Check the run configuration
- Check if you accidentally changed any settings in `config_refactored.yaml

## Reproducing Results 

### General Description 
To reproduce the results, adjust the parameters in `config_refactored.yaml` and execute the analyses sequentially. Below are the abbreviations used in the configuration file explained:

- **"main"**: Main analysis
- **"suppl"**: Supplementary analysis
- **"ssc"**: Social interaction characteristics
- **"mse"**: Major societal events

#### Types of Supplementary Analyses
1. **"sep_ftf_cmc"**: Analysis where face-to-face and computer-mediated interactions are separated (Supplement 2)
   - **"ftf"**: Face-to-face social interactions
   - **"cmc"**: Computer-mediated social interactions
   - **"ftf_pa"**: Combination of face-to-face social interactions and positive affective reactivities
2. **"sep_pa_na"**: Analysis where positive and negative affective reactivities are separated (Supplement 3)
   - **"pa"**: Positive affect
   - **"na"**: Negative affect
3. **"weighting_by_rel"**: Analysis where individual random effects are weighted by their estimated reliability (Supplement 4)
   - **"random_slopes"**: Weighting the empirical Bayes estimates
   - **"ols_slopes"**: Weighting the Ordinary Least Squares (OLS) estimates (individual models, N=1)
4. **"add_wb_change"**: Supplementary analysis where the initial well-being change is added as a person-level variable (Supplement 5)


### Walk-Through 
We provide a walk-through how to reproduce the results for the main analysis of Study 1 (reactivities to social interactions).
The **data** folder contains the raw data (data/raw, data/external_country_data) as well as the preprocessed data (data/preprocessed) for the machine learning-based analysis. 
If one only wishes to run the machine learning-based analysis, one may skip Steps 2 and 3. Steps 2 and 3 will reproduce (data/preprocessed) for the specified analysis. 

#### Step 1: Analysis Configuration
Modify the following lines in your configuration file to set up the main analysis parameters (lines 31-34):
- `analysis: main`
- `suppl_type:`
- `suppl_var:`
- `study: ssc`

#### Step 2: Specify Analysis Steps for Preprocessing
Adjust the following lines to enable preprocessing (lines 8-16):
- `preprocessing: true`
- `calc_mlms: false`
- `ml_analysis: false`
- `result_analysis: false`
- `prelim_results_analysis: false`
- `significance_tests: false`
- `cv_results_plots: false`
- `shap_value_analysis: false`
- `osf_suppl_analysis: false`

Run the main function after setting up the above parameters.

#### Step 3: Enable Calculation of Mixed Linear Models (MLMs)
Update the following lines to proceed with MLMs calculation (lines 8-16):
- `preprocessing: false`
- `calc_mlms: true`
- `ml_analysis: false`
- `result_analysis: false`
- `prelim_results_analysis: false`
- `significance_tests: false`
- `cv_results_plots: false`
- `shap_value_analysis: false`
- `osf_suppl_analysis: false`

Run the main function once these settings are adjusted.

#### Step 4: Conduct Machine Learning-Based Analysis
Specify the machine-learning based analysis parameters (lines 35-38). The current configuration executes the 10x10x10 CV procedure for the variable "social_interaction" in "emotions" using "scale_means" as features and "linear_baseline_model" as prediction model:
- `social_interaction_variable: social_interaction`
- `esm_sample: emotions`
- `feature_inclusion_strategy: scale_means`
- `model: linear_baseline_model`

Update the analysis steps to enable machine learning analysis (lines 8-16):
- `preprocessing: false`
- `calc_mlms: false`
- `ml_analysis: true`
- `result_analysis: false`
- `prelim_results_analysis: false`
- `significance_tests: false`
- `cv_results_plots: false`
- `shap_value_analysis: false`
- `osf_suppl_analysis: false`

Run the main function for this configuration.

#### Step 5: Adjust for Subsequent Methods
For subsequent methods, adjust lines 8-16 as needed for the specific method being implemented.

#### Other analysis settings
All supplementary analyses can be computed equally, it just needs other adjustments of lines 31-34. 
For example, if one want to execute the whole procedure for the supplementary analysis sep_ftf_cmc for ftf (face-to-face interactions), one has to specify these parameters and repeat steps 1-5: 
- `analysis: suppl`
- `suppl_type: sep_ftf_cmc`
- `suppl_var: ftf`
- `study: ssc`
- For results of Study 2, change "ssc" to "mse"


## Project Structure

```plaintext
psychological_reactivities_analysis/
│
├── configs/
│   ├── config_country_data.yaml
│   ├── config_refactored.yaml
│   └── feature_name_mapping.yaml
│
├── data/
│   ├── external_country_data
│   ├── preprocessed
│   └── raw
│
├── logs/  # created when producing logs 
│
├── results/ # created when storing the first results
│
├── src/
│   ├── analysis/
│   │   ├── machine_learning/
│   │   │   ├── BaseMLAnalyzer.py
│   │   │   ├── CustomScaler.py
│   │   │   ├── LassoAnalyzer.py
│   │   │   ├── LinearAnalyzer.py
│   │   │   ├── LinearBaselineAnalyzer.py
│   │   │   ├── RFRAnalyzer.py
│   │   │   └── SVRAnalyzer.py
│   │   │
│   │   ├── multilevel_modeling/
│   │   │   └── MultilevelModeling.py
│   │   │
│   │   ├── result_analysis/
│   │   │   ├── CVResultPlotter.py
│   │   │   ├── PrelimResultAnalyzer.py
│   │   │   ├── result_utils.py
│   │   │   ├── ResultAnalyzer.py
│   │   │   ├── ShapValueAnalyzer.py
│   │   │   └── SignificanceTesting.py
│   │   │
│   ├── preprocessing/
│   │   ├── BasePreprocessor.py
│   │   ├── helper_functions.py
│   │   ├── PreprocessorMSE.py
│   │   └── PreprocessorSSC.py
│   │
│   ├── main.py
│   ├── main_utils.py
│   ├── slurm_script_reactivities.sh
│
└── .gitignore
└── README.md
└── requirements.txt




