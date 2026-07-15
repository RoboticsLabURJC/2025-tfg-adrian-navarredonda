## PilotNetDefault

Directory **PilotNetDefault** contains everything required for the **basic PilotNet training setup**, where **only the RGB image is used as the input channel**.

- **/experiments** – Stores the experiments generated during training.
- **/utils** – Contains the utilities required by the network, including the dataset generation pipeline and the PilotNet architecture implementation.
- **run_training.sh** – Script used to launch the training process. It contains the arguments passed to the training code.
- **train_final** – Main training script implementing the full pipeline: training, validation, and testing. This script is executed through the `.sh` launcher.
- **run_carla_autopilot_rgb** – Script used to test a trained model in inference mode inside CARLA. The visualization is displayed in a third-person videogame-style camera perspective.
- **videocam** – Script used to test a trained model in inference. It alternates between a dynamic heatmap visualization and a red trajectory trail following the vehicle. The circuit number must be passed as an argument.

### Usage

Run training:

./run_training.sh

Test a model in CARLA (third-person camera):

python3 run_carla_autopilot_rgb.py

Run inference visualization with circuit argument:

python3 videocam.py 4