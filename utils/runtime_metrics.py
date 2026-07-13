import time
from collections import defaultdict

from prettytable import PrettyTable
import torch


def synchronized_perf_counter(device=None):
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def log_training_time_per_epoch(logger, epoch, elapsed_seconds):
    logger.info("Training time / epoch: {:.4f} s (epoch {})".format(elapsed_seconds, epoch))


def measure_forward_latency(forward_fn, device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = forward_fn()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()
        output = forward_fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return output, elapsed_ms


class QueryLatencyMeter:
    _GROUP_LABELS = {
        1: "ONE_AVER",
        2: "TWO_AVER",
        3: "THREE_AVER",
        4: "FOUR_AVER",
    }

    def __init__(self):
        self._task_stats = {}
        self._group_stats = defaultdict(lambda: {"total_ms": 0.0, "batches": 0, "queries": 0})

    def update(self, task_name, modality_count, elapsed_ms, num_queries):
        if num_queries <= 0:
            return
        if task_name not in self._task_stats:
            self._task_stats[task_name] = {
                "modality_count": int(modality_count),
                "total_ms": 0.0,
                "batches": 0,
                "queries": 0,
            }

        stats = self._task_stats[task_name]
        stats["total_ms"] += float(elapsed_ms)
        stats["batches"] += 1
        stats["queries"] += int(num_queries)

        group_stats = self._group_stats[int(modality_count)]
        group_stats["total_ms"] += float(elapsed_ms)
        group_stats["batches"] += 1
        group_stats["queries"] += int(num_queries)

    def log(self, logger):
        if not self._task_stats:
            return

        table = PrettyTable(["task", "batches", "queries", "total_ms", "ms/batch", "ms/query"])
        for task_name, stats in self._task_stats.items():
            self._add_latency_row(table, task_name, stats)

        self._format_latency_table(table)
        logger.info(
            "\nInference latency / query (ms) "
            "(model forward only; excludes data loading, gallery extraction, similarity and ranking):\n"
            + str(table)
        )

        group_table = PrettyTable(["group", "batches", "queries", "total_ms", "ms/batch", "ms/query"])
        for modality_count in sorted(self._group_stats):
            label = self._GROUP_LABELS.get(modality_count, "{}_MODAL_AVER".format(modality_count))
            self._add_latency_row(group_table, label, self._group_stats[modality_count])

        self._format_latency_table(group_table)
        logger.info(
            "\nInference latency / query average by modality count (ms):\n"
            + str(group_table)
        )

    @staticmethod
    def _add_latency_row(table, task_name, stats):
        total_ms = stats["total_ms"]
        batches = max(stats["batches"], 1)
        queries = max(stats["queries"], 1)
        table.add_row([
            task_name,
            stats["batches"],
            stats["queries"],
            total_ms,
            total_ms / batches,
            total_ms / queries,
        ])

    @staticmethod
    def _format_latency_table(table):
        for field in ["total_ms", "ms/batch", "ms/query"]:
            table.custom_format[field] = lambda f, v: "{:.3f}".format(v)
