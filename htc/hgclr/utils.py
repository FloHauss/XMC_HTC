import json
import os
import time
import torch
import random
import numpy as np


def _synchronise_cuda():
    """Wait for queued GPU work before taking a wall-clock timestamp."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CostTracker:
    """Tracks wall-clock time, GPU memory, and throughput for training and inference."""

    def __init__(self):
        self._t0_training = None
        self._t0_epoch = None
        self._t0_train_phase = None
        self._t0_val_phase = None
        self._t0_inference = None
        self._epoch_records = []
        self._training_total_sec = None
        self._inference_record = None
        self._model_params = {}

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def record_model(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._model_params = {'model_params_total': total, 'model_params_trainable': trainable}

    @property
    def model_parameters(self):
        return dict(self._model_params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def start_training(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _synchronise_cuda()
        self._t0_training = time.time()

    def start_epoch(self):
        _synchronise_cuda()
        self._t0_epoch = time.time()
        self._t0_train_phase = time.time()

    def end_train_phase(self, num_samples):
        _synchronise_cuda()
        self._train_time = time.time() - self._t0_train_phase
        self._train_throughput = num_samples / self._train_time if self._train_time > 0 else 0.0
        self._t0_val_phase = time.time()

    def end_val_phase(self, epoch, scores):
        _synchronise_cuda()
        val_time = time.time() - self._t0_val_phase
        gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        self._epoch_records.append({
            'epoch': epoch,
            'train_time_sec': round(self._train_time, 3),
            'val_time_sec': round(val_time, 3),
            'train_throughput_samples_per_sec': round(self._train_throughput, 2),
            'gpu_memory_allocated_mb': round(gpu_mb, 2),
            'macro_f1': round(scores['macro_f1'], 6),
            'micro_f1': round(scores['micro_f1'], 6),
        })

    def end_training(self):
        _synchronise_cuda()
        self._training_total_sec = time.time() - self._t0_training

    def training_summary(self):
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        samples = [r['gpu_memory_allocated_mb'] for r in self._epoch_records]
        avg_mb = sum(samples) / len(samples) if samples else 0.0
        return {
            'total_time_sec': round(self._training_total_sec, 3),
            'epochs_completed': len(self._epoch_records),
            'peak_gpu_memory_mb': round(peak_mb, 2),
            'avg_gpu_memory_mb': round(avg_mb, 2),
            'per_epoch': self._epoch_records,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def start_inference(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _synchronise_cuda()
        self._t0_inference = time.time()

    def end_inference(self, num_samples, checkpoint_extra=''):
        _synchronise_cuda()
        elapsed = time.time() - self._t0_inference
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        self._inference_record = {
            'checkpoint': checkpoint_extra,
            'total_time_sec': round(elapsed, 3),
            'num_samples': num_samples,
            'throughput_samples_per_sec': round(num_samples / elapsed, 2) if elapsed > 0 else 0.0,
            'peak_gpu_memory_mb': round(peak_mb, 2),
        }

    def inference_summary(self):
        return self._inference_record

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, run_name='', dataset=''):
        d = {'run_name': run_name, 'dataset': dataset}
        d.update(self._model_params)
        if self._epoch_records:
            d['training'] = self.training_summary()
        if self._inference_record:
            d['inference'] = self._inference_record
        return d

    def save(self, path, extra=None):
        existing = {}
        if os.path.exists(path):
            with open(path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass
        if self._epoch_records:
            existing.update(self.to_dict(
                run_name=existing.get('run_name', ''),
                dataset=existing.get('dataset', ''),
            ))
        if self._inference_record:
            existing['inference'] = self._inference_record
        if extra:
            existing.update(extra)
        with open(path, 'w') as f:
            json.dump(existing, f, indent=2)

    def print_training_summary(self):
        s = self.training_summary()
        print(f'[cost] training_time={s["total_time_sec"]:.1f}s  epochs={s["epochs_completed"]}'
              f'  peak_gpu={s["peak_gpu_memory_mb"]:.0f}MB  avg_gpu={s["avg_gpu_memory_mb"]:.0f}MB')

    def print_inference_summary(self):
        s = self._inference_record
        print(f'[cost] inference_time={s["total_time_sec"]:.1f}s'
              f'  throughput={s["throughput_samples_per_sec"]:.1f} samples/sec'
              f'  peak_gpu={s["peak_gpu_memory_mb"]:.0f}MB')


# seed_torch(3)
# print('Set seed to 3.')
