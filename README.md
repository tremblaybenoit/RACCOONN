# RACCOONN: Retrievals of Atmospheric Conditions Computed from Observations by Optimizing a Neural Network. 
RACCOONN uses deep learning to estimate atmospheric thermodynamic profiles from raw radiance observations. 
A prior/background state of the atmosphere can be provided. 
The goal is to create an inverse observation operator for the assimilation of radiances in the form of thermodynamic profiles.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Experiment configuration](#experiment-configuration)
  - [Manual execution of individual steps](#manual-execution-of-individual-steps)
    - [Forward model](#forward-model-eg-experimentforward_default)
    - [Inverse model](#inverse-model-eg-experimentinverse_default)
  - [Automated workflow (recommended)](#automated-workflow-recommended)
- [Documentation](#documentation)
- [References and Acknowledgements](#references-and-acknowledgements)

## Installation

Clone the repository:

```bash
git clone https://github.com/tremblaybenoit/RACCOONN.git
```
RACCOONN is built with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/docs/intro/). 

Create a new conda environment and install pre-requisites by executing the script [`environment.sh`](scripts/environment.sh):
```bash
./scripts/environment.sh
conda activate RACCOONN
```

## Usage
### Experiment configuration
Create or edit a configuration file in the [`config/experiment`](config/experiment) folder to set your experiment parameters.

1. Start with the `defaults` section to set the default configurations:
    - `paths` (from folder [`config/paths`](config/paths)): Directories for data and outputs.
    - `hydra` (from folder [`config/hydra`](config/hydra)): Hydra settings.
    - `data` (from folder [`config/data`](config/data)): Dataset parameters (e.g., variables, i/o functions).
    - `preprocessing` (from folder [`config/preprocessing`](config/preprocessing)): Data preprocessing steps (e.g., statistics).
    - `loader` (from folder [`config/loader`](config/loader)): Wraps [`config/data`](config/data) into a Pytorch-Lightning-ready data loader.
    - `model` (from folder [`config/model`](config/model)): Wraps [`config/architecture`](config/architecture), [`config/optimizer`](config/optimizer), [`config/scheduler`](config/scheduler), and [`config/loss`](config/loss) into a complete model.
    - `architecture` (from folder [`config/architecture`](config/architecture)): Neural network architecture details.
    - `optimizer` (from folder [`config/optimizer`](config/optimizer)): Optimizer parameters.
    - `scheduler` (from folder [`config/scheduler`](config/scheduler)): Learning rate scheduler parameters.
    - `loss` (from folder [`config/loss`](config/loss)): Loss function parameters.
    - `trainer` (from folder [`config/trainer`](config/trainer)): Training parameters.
    - `callbacks` (from folder [`config/callbacks`](config/callbacks)): Callbacks during training.
    - `logger` (from folder [`config/logger`](config/logger)): Logging parameters during training.
2. Add `overrides` below the `defaults` to change specific default parameters as needed. 

**Note**: The order of the `defaults` matters, as later entries can override earlier ones.

**Example**: The following diagram illustrates the structure of [`config/experiment/inverse_default.yaml`](config/experiment/inverse_default.yaml). 
It sets the `defaults` and then performs parameter `overrides`.
```mermaid
---
title: Structure of the experiment configuration file "config/experiment/inverse_default.yaml"
---
flowchart LR
  A["/experiment: inverse_default"]
  A --> B["defaults"]
  B --> B1["/paths: default"]
  B --> B2["/hydra: default"]
  B --> B3["/data: inverse_default"]
  B --> B4["/preprocessing: inverse_default"]
  B --> B5["/loader: default"]
  B --> B6["/model: inverse_default"]
  B --> B7["/architecture: hydra_mlp"]
  B --> B8["/optimizer: adam"]
  B --> B9["/scheduler: plateau"]
  B --> B10["/loss: var"]
  B --> B11["/trainer: gpu"]
  B --> B12["/callbacks: inverse_default"]
  B --> B13["/logger: default"]

  %% Each override points to the final overrides section
  B1 --> C
  B2 --> C
  B3 --> C
  B4 --> C
  B5 --> C
  B6 --> C
  B7 --> C
  B8 --> C
  B9 --> C
  B10 --> C
  B11 --> C
  B12 --> C
  B13 --> C

  C["Overrides"]
  C --> C1["task_name"]
  C --> C2["/paths"]
  C2 --> C21["task_dir"]
  C2 --> C22["run_id"]
  C2 --> C23["data_dir"]
  C --> C3["/trainer"]
  C3 --> C31["min_epochs"]
  C3 --> C32["max_epochs"]
  C --> C4["/data"]
  C4 --> C41["dtype"]

  %% Color per config category
  classDef experiment fill:#22313F,stroke:#888,stroke-width:1px,color:#fff;
  classDef final_overrides fill:#22313F,stroke:#888,stroke-width:1px,color:#fff;
  classDef paths fill:#FFD580,stroke:#888,stroke-width:1px,color:#000;
  classDef hydra fill:#A97FFF,stroke:#888,stroke-width:1px,color:#000;
  classDef data fill:#B0E57C,stroke:#888,stroke-width:1px,color:#000;
  classDef preprocessing fill:#FFB347,stroke:#888,stroke-width:1px,color:#000;
  classDef loader fill:#FF7F7F,stroke:#888,stroke-width:1px,color:#000;
  classDef model fill:#FFB3B3,stroke:#888,stroke-width:1px,color:#000;
  classDef architecture fill:#FFD4E5,stroke:#888,stroke-width:1px,color:#000;
  classDef optimizer fill:#D4E5FF,stroke:#888,stroke-width:1px,color:#000;
  classDef scheduler fill:#E5FFD4,stroke:#888,stroke-width:1px,color:#000;
  classDef loss fill:#FFE5D4,stroke:#888,stroke-width:1px,color:#000;
  classDef trainer fill:#80B3FF,stroke:#888,stroke-width:1px,color:#000;
  classDef callbacks fill:#57D9AD,stroke:#888,stroke-width:1px,color:#000;
  classDef logger fill:#D99157,stroke:#888,stroke-width:1px,color:#000;

  %% Assign colors to boxes
  class A,B,C1 experiment;
  class C final_overrides;
  class B1,C2,C21,C22,C23 paths;
  class B2 hydra;
  class B3,C4,C41 data;
  class B4 preprocessing;
  class B5 loader;
  class B6 model;
  class B7 architecture;
  class B8 optimizer;
  class B9 scheduler;
  class B10 loss;
  class B11,C3,C31,C32 trainer;
  class B12 callbacks;
  class B13 logger;
```

### Manual execution of individual steps
Each step of the workflow can be run manually using the corresponding Python script and 
experiment configuration.

#### Forward model (e.g., [`experiment=forward_default`](config/experiment/forward_default.yaml))

1. Configure directories:

    ```bash
    python -m config.setup +experiment=forward_default
    ```

2. Train the forward model:

    ```bash
    python -m code.train +experiment=forward_default
    ```

3. Test and evaluate the forward model:

    ```bash
    python -m code.test +experiment=forward_default
    ```

4. Predict using the forward model:

    ```bash
    python -m code.predict +experiment=forward_default
    ```

#### Inverse model (e.g., [`experiment=inverse_default`](config/experiment/inverse_default.yaml))

1. Configure directories:

    ```bash
    python -m config.setup +experiment=inverse_default
    ```

2. Prepare data for the inverse model:

    ```bash
    python -m code.data.statistics +experiment=inverse_default
    python -m code.data.covariance +experiment=inverse_default
    ```

3. Train the inverse model:

    ```bash
    python -m code.train +experiment=inverse_default
    ```

4. Test and evaluate the inverse model:

    ```bash
    python -m code.test +experiment=inverse_default
    ```

5. Predict using the inverse model:

    ```bash
    python -m code.predict +experiment=inverse_default
    ```

### Automated workflow (recommended)
RACCOONN uses the [Snakemake workflow management system](https://snakemake.readthedocs.io/en/stable/) for reproducibility.

To perform a dry-run (i.e., to check the workflow prior to execution) of the Snakefile rule [`test`](Snakefile) with the [`inverse_default`](config/experiment/inverse_default.yaml) experiment configuration:

```bash
snakemake --dry-run --verbose test --config hydra-experiment=inverse_default
```

Remove `--dry-run` to actually run the workflow. 

To account for missing dependencies, add the `--rerun-incomplete` flag:

```bash
snakemake --dry-run --rerun-incomplete --verbose test --config hydra-experiment=inverse_default
```

To draw a [directed acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of the training workflow (e.g., [`code/train.mmd`](code/train.mmd) for Snakefile rule [`test`](Snakefile)):

```bash
snakemake test --rulegraph mermaid-js --config hydra-experiment=inverse_default > train.mmd
```
Replace `--rulegraph` with `--dag` to highlight completed rules with dashed boxes.

**Example**: The following graph shows the workflow for the Snakefile rule [`test`](Snakefile) for experiment [`inverse_default`](config/experiment/inverse_default.yaml). 

```mermaid
---
title: RACCOONN training workflow - Inverse model
---
flowchart TB
	id0[test]
	id1[data]
	id2[statistics_data]
	id3[covariance_R]
	id4[covariance_B]
	id5[train]
	style id0 fill:#57CAD9,stroke-width:2px,color:#333333
	style id1 fill:#D9CA57,stroke-width:2px,color:#333333
	style id2 fill:#57D9AD,stroke-width:2px,color:#333333
	style id3 fill:#D99157,stroke-width:2px,color:#333333
	style id4 fill:#D95757,stroke-width:2px,color:#333333
	style id5 fill:#5791D9,stroke-width:2px,color:#333333
	id5 --> id0
	id3 --> id0
	id1 --> id0
	id2 --> id0
	id4 --> id0
	id1 --> id2
	id1 --> id3
	id1 --> id4
	id2 --> id4
	id1 --> id5
	id2 --> id5
	id3 --> id5
	id4 --> id5
```

## Documentation
The RACCOON project documentation is available at https://raccoonn.readthedocs.io/.

## References and Acknowledgements
- The forward model is a translation from Keras to Pytorch Lightning of an emulator published in the following paper and repository (full credit goes to the original authors): 
  - Paper by Howard et al. (2025): https://www.arxiv.org/abs/2504.16192.
  - Repository: https://zenodo.org/records/13963758.
- Inspiration for the PyTorch Lightning + Hydra framework comes from the following repositories: 
  - Lightning-Hydra-Template: https://github.com/ashleve/lightning-hydra-template.
  - Anemoi framework by ECMWF: https://github.com/ecmwf/anemoi-core.
- Inspiration for the use of Neural Fields (NFs) for retrievals/inversions comes from the following paper and repository:
  - Paper by Jarolim et al. (2025): https://arxiv.org/pdf/2502.13924.
  - Repository: https://github.com/RobertJaro/pinn-me.
- Inspiration for the use of emulation for retrievals/inversions comes from the following papers:
  - Paper by Ermis et al. (2025): https://www.climatechange.ai/papers/neurips2025/63
  - Paper by Girtsou et al. (2024): https://neurips.cc/media/PosterPDFs/NeurIPS%202024/100006.png?t=1733082861.1906157.
