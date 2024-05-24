class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './workspace'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'  # Directory for tensorboard files.
        self.pretrained_nets_dir = './pretrained_networks'  # Directory for pre-trained networks.
        self.save_predictions_path = self.workspace_dir + '/saved_images'  # Directory for saving network predictions for evaluation.

        self.zurichraw2rgb_dir = './_DATASETS/zurich-raw-to-rgb'  # Zurich RAW 2 RGB path
        self.synburstval_dir = './_DATASETS/SyntheticBurstVal'  # SyntheticBurst validation set path
        self.burstsr_dir = './_DATASETS/burstsr_cropped'  # BurstSR dataset path
