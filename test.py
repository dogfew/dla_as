import os
import warnings

import hydra
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.eer import compute_eer
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore")

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@hydra.main(config_path="src/configs", config_name="config.yaml")
def main(config):
    config = ConfigParser(config)
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    print(config["resume"])
    checkpoint = torch.load(config["resume"], map_location=device)
    print("Checkpoint!")
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    args = config["test_settings"]
    output_dir = args.out_dir
    dataloaders = get_dataloaders(config)
    with torch.no_grad():
        for test_type in ['test']:
            all_probs = []
            all_targets = []
            for batch_num, batch in enumerate(tqdm(dataloaders[test_type])):
                batch = Trainer.move_batch_to_device(batch, device)
                batch.update(model(**batch))
                all_probs.extend(batch['logits'][:, 1].detach().cpu().tolist())
                all_targets.extend(batch['targets'].detach().cpu().tolist())

            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            eer, thres = compute_eer(
                bonafide_scores=all_probs[all_targets == 1],
                other_scores=all_probs[all_targets == 0])
            print(f"EER:   {eer}\nThres: {thres}")




if __name__ == "__main__":
    main()
