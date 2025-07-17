# ♻️ Trash Classification with EfficientNet-B0 on Google Cloud TPU

This project fine-tunes an **EfficientNet-B0** model on the **TrashNet** dataset for classifying waste items into 6 categories. Training is performed efficiently on **Google Cloud TPUs** using PyTorch/XLA.

## Key Features:
-   **Model:** EfficientNet-B0 (ImageNet pre-trained via `timm` library)
-   **Dataset:** TrashNet (6 classes, ~2500 images)
-   **Hardware:** Google Cloud TPU (e.g., v3-8)
-   **Framework:** PyTorch/XLA

## Setup & Run:

1.  **Google Cloud TPU VM:** Ensure you have a TPU VM (e.g., `v3-8`) running on GCP.
2.  **Clone this repository** to your TPU VM:
    ```bash
    git clone https://github.com/nurlanjalil/efficientnet-trashnet-tpu.git
    cd efficientnet-trashnet-tpu
    ```
3.  **Activate your Python Virtual Environment** (e.g., `source my_ml_env/bin/activate`).
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Dataset:** Download the [TrashNet dataset](https://www.kaggle.com/datasets/asdasf/trashnet). Extract `trashnet_dataset` and place it in the root of this repository. **Verify `DATA_DIR` in the notebook (Cell 2) matches this path.**
6.  **SSH Tunnel for Jupyter Lab:** **(MOVED UP)**
    From your **local machine's terminal**, use an SSH tunnel to connect to your VM:
    ```bash
    gcloud compute tpus tpu-vm ssh YOUR_TPU_VM_NAME --zone=YOUR_TPU_ZONE -- -L 8888:localhost:8888
    ```
    Keep this local terminal session open.
7.  **Start Jupyter Lab on your VM:** **(MOVED DOWN)**
    From your project directory in the **VM terminal** (with your venv active):
    ```bash
    jupyter lab --no-browser --port=8888 --ip=0.0.0.0
    ```
    *(This command will print the URL with the token you need.)*
8.  **Access Jupyter Lab in Browser:** **(MOVED DOWN)**
    Open `http://localhost:8888` in your local web browser. You will be prompted for a token. Paste the token displayed in your VM terminal from the previous step.
9.  **Run Notebook:** Open `garbage_classification_tpu.ipynb` and execute all cells.

## 📈 Results

| Metric               | Value     |
|----------------------|-----------|
| Final Val Accuracy   | **89.4%** |
| Final Val Loss       | **0.32**  |

- A sample metrics plot is available in `training_metrics_plot_efficientnet_b0.png`
- Tested on unseen images — consistent results


## Model Checkpoint:

The best-performing model checkpoint is provided as `best_model_efficientnet_b0_final.pt`.

## Acknowledgements:

Special thanks to the [Google Cloud TPU Research Cloud (TRC)](https://www.tensorflow.org/tfrc) program for providing access to TPU resources.
