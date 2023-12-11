import os

import PIL
import librosa
import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torchvision.transforms import (
    ToTensor,
)
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import (
    inf_loop,
    MetricTracker,
    ROOT_PATH,
)
from src.utils import optional_autocast
from src.utils.eer import compute_eer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        eval_interval=5,
        mixed_precision=False,
        scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics=metrics,
            optimizer=optimizer,
            config=config,
            device=device,
            lr_scheduler=scheduler,
        )
        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        print(self.evaluation_dataloaders)
        self.config = config
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_interval = eval_interval
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            *metrics,
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(*metrics, writer=self.writer)
        self.scaler = GradScaler(enabled=self.mixed_precision)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_name in ["waves", "mels", "targets"]:
            if tensor_name in batch:
                batch[tensor_name] = batch[tensor_name].to(device)

        return batch

    def _clip_grad_norm(self, optimizer):
        self.scaler.unscale_(optimizer)
        if self.config["trainer"].get("grad_norm_clip") is not None:
            try:
                clip_grad_value_(
                    parameters=self.model.parameters(),
                    clip_value=self.config["trainer"]["grad_max_abs"],
                )
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config["trainer"]["grad_norm_clip"],
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                return False
        return True

    def _train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        all_probs = []
        all_targets = []
        for i, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.len_epoch,
            )
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            all_probs.append(batch["probs"][:, 1].cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            if i >= self.len_epoch:
                break
        try:
            all_targets = np.concatenate(all_targets)
            all_probs = np.concatenate(all_probs)
            self.train_metrics.update(
                "roc_auc",
                roc_auc_score(all_targets, all_probs),
            )
            eer, thres = compute_eer(
                bonafide_scores=all_probs[all_targets == 1],
                other_scores=all_probs[all_targets == 0],
            )
            self.train_metrics.update("eer", eer)
        except ValueError as e:
            print(e)
        last_train_metrics = self.debug(
            batch,
            i,
            epoch,
        )
        self.scheduler.step()
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            if epoch % self.eval_interval == 0 and part == "test":
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(
                    **{f"{part}_{name}": value for name, value in val_log.items()}
                )
        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.len_epoch)
        return log

    @torch.no_grad()
    def debug(self, batch, batch_idx, epoch):
        self.logger.debug(
            "Train Epoch: {} {} Loss: {:.4f}".format(
                epoch, self._progress(batch_idx), self.train_metrics.avg("loss")
            )
        )
        self._log_scalars(self.train_metrics)
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
        self._log_spectrogram(batch, mode="train")
        if self.writer is not None:
            self.writer.add_scalar(
                "epoch",
                epoch,
            )
            self.writer.add_scalar(
                "learning rate",
                self.optimizer.state_dict()["param_groups"][0]["lr"],
            )
            self.writer.add_scalar("scaler factor", self.scaler.get_scale())

            # audio_generator_example = (
            #     batch["wave_fake_detached"][0]
            #     .detach()
            #     .cpu()
            #     .to(torch.float32)
            #     .numpy()
            #     .flatten()
            # )
            # audio_true_example = (
            #     batch["wave_true"][0].detach().cpu().to(torch.float32).numpy().flatten()
            # )
            # self.writer.add_audio(
            #     "generated",
            #     audio_generator_example,
            #     sample_rate=22050,
            # )
            # self.writer.add_audio(
            #     "true",
            #     audio_true_example,
            #     sample_rate=22050,
            # )
        return last_train_metrics

    def process_batch(
        self,
        batch,
        metrics: MetricTracker,
    ):
        batch = self.move_batch_to_device(batch, self.device)
        with optional_autocast(enabled=self.mixed_precision):
            self.optimizer.zero_grad(set_to_none=True)
            batch.update(
                self.model(
                    **batch,
                )
            )
            batch.update(self.criterion(**batch))
        self.scaler.scale(batch["loss"]).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_metrics.update(
            "grad_norm",
            self.get_grad_norm(),
        )

        for item in batch:
            if item in self.train_metrics.keys():
                metrics.update(
                    item,
                    batch[item].item(),
                )
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(
            self.train_dataloader,
            "n_samples",
        ):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(
            current,
            total,
            100.0 * current / total,
        )

    @torch.no_grad()
    def _log_predictions(
        self,
        *args,
        **kwargs,
    ):
        if self.writer is None:
            return
        rows = {}
        dirpath = ROOT_PATH / "test_data"

        for audio_file in os.listdir(dirpath):
            if audio_file.endswith(".flac") or audio_file.endswith(".wav"):
                audio, _ = librosa.load(dirpath / audio_file, sr=16_000)
                audio_tensor = torch.tensor(audio, device=self.device)
                out = self.model(audio_tensor.unsqueeze(dim=0))["logits"]
                prob_real = torch.softmax(out, dim=1)[:, 1].flatten()[0].item()
                rows[audio_file] = {
                    "audio_name": audio_file,
                    "audio": wandb.Audio(audio, sample_rate=16_000),
                    "prob_real": prob_real,
                }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                with optional_autocast(enabled=self.mixed_precision):
                    batch = self.move_batch_to_device(batch, self.device)
                    batch.update(
                        self.model(
                            **batch,
                        )
                    )
                    batch.update(self.criterion(**batch))

                    all_probs.append(batch["probs"][:, 1].cpu().numpy())
                    all_targets.append(batch["targets"].cpu().numpy())
                for item in batch:
                    if item in self.train_metrics.keys():
                        self.evaluation_metrics.update(
                            item,
                            batch[item].item(),
                        )
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        try:
            self.evaluation_metrics.update(
                "roc_auc",
                roc_auc_score(all_targets, all_probs),
            )
            eer, thres = compute_eer(
                bonafide_scores=all_probs[all_targets == 1],
                other_scores=all_probs[all_targets == 0],
            )
            self.evaluation_metrics.update("eer", eer)
            # np.save("bonafide.npy", all_probs[all_targets == 1])
            # np.save("other.npy", all_probs[all_targets == 0])
        except ValueError as e:
            print(e)
            self.evaluation_metrics.update(
                "roc_auc",
                np.nan,
            )
            self.evaluation_metrics.update("eer", np.nan)
        if self.writer is not None:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins="auto")
        self._log_scalars(self.evaluation_metrics, test=True)
        self._log_spectrogram(batch, "test")
        result = self.evaluation_metrics.result()
        self._log_predictions()
        return result

    @staticmethod
    def make_image(buff):
        return ToTensor()(PIL.Image.open(buff))

    @torch.no_grad()
    def _log_spectrogram(self, batch, mode="train"):
        if "mels" not in batch or self.writer is None:
            return
        spectrogram = batch[f"mels"][0].detach().cpu().to(torch.float64)
        spectrogram = torch.nan_to_num(spectrogram)
        buf = plot_spectrogram_to_buf(spectrogram)
        self.writer.add_image(
            f"mel_{mode}",
            Trainer.make_image(buf),
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        torch.nan_to_num(
                            p.grad,
                            nan=0,
                        ).detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return float(total_norm.item())

    @torch.no_grad()
    def _log_scalars(self, metric_tracker: MetricTracker, test=False):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                metric_name + "_test" * test,
                metric_tracker.avg(metric_name),
            )
