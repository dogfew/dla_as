import os
import warnings

import hydra
import librosa
import numpy as np
import pandas as pd
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


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@torch.no_grad()
@hydra.main(config_path="src/configs", config_name="config_rawnet2.yaml")
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
    dataloaders = get_dataloaders(config)
    dataset = dataloaders["test"].dataset
    with torch.no_grad():
        for test_type in ["test"]:
            if args["skip_test"]:
                print("Skipping test!")
                continue
            all_probs = []
            all_targets = []
            for batch_num, batch in enumerate(tqdm(dataloaders[test_type])):
                batch = Trainer.move_batch_to_device(batch, device)
                batch.update(model(**batch))
                all_probs.extend(batch["logits"][:, 1].detach().cpu().tolist())
                all_targets.extend(batch["targets"].detach().cpu().tolist())

            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            eer, thres = compute_eer(
                bonafide_scores=all_probs[all_targets == 1],
                other_scores=all_probs[all_targets == 0],
            )
            print(f"EER:   {eer}\nThres: {thres}")
    rows = {}
    dirpath = ROOT_PATH / args["audio_dir"]
    if not os.path.exists(dirpath):
        return
    for audio_file in os.listdir(dirpath):
        if audio_file.endswith(".flac") or audio_file.endswith(".wav"):
            audio, _ = librosa.load(dirpath / audio_file, sr=16_000)
            if model.__class__.__name__ == "LightCNN":
                audio_tensor = dataset._process_audio(
                    {"path": dirpath / audio_file, "type": ""}
                )["mel"].to(device)
            else:
                audio, _ = librosa.load(dirpath / audio_file, sr=16_000)
                audio_tensor = torch.tensor(audio, device=device)
            out = model(audio_tensor.unsqueeze(dim=0))["logits"]
            prob_fake, prob_real = torch.softmax(out, dim=1).flatten().cpu().tolist()
            rows[audio_file] = {
                "audio_name": audio_file,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
            }
    else:
        print(pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True))


if __name__ == "__main__":
    main()
