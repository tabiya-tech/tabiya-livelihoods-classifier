"""Progress reporting: stage banners and a plain-language training progress callback."""

import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class StageReporter:
    """Prints a banner whenever a script moves to its next stage."""

    def __init__(self, total_stages: int) -> None:
        self.total_stages = total_stages
        self.current_stage = 0
        self.start_time = time.time()

    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60

    def start(self, description: str) -> None:
        self.current_stage += 1
        banner_line = "=" * 72
        print(
            f"\n{banner_line}\n[Stage {self.current_stage}/{self.total_stages}] {description}"
            f"   (elapsed {self.elapsed_minutes():.1f} min, {time.strftime('%H:%M:%S')})\n{banner_line}",
            flush=True,
        )

    def finish(self) -> None:
        print(f"\nAll done in {self.elapsed_minutes():.1f} min.", flush=True)


class TrainingProgressCallback(TrainerCallback):
    """Prints a summary at the start and end of every pass over the data (epoch), and the
    validation score after each one, with an estimate of the time left."""

    def __init__(self) -> None:
        self.training_start_time = 0.0
        self.latest_train_loss: float | None = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.training_start_time = time.time()
        print(
            f"Training for up to {int(args.num_train_epochs)} passes over the data "
            f"({state.max_steps} update steps); it stops early if the validation score stalls.",
            flush=True,
        )

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        pass_number = int(state.epoch or 0) + 1
        print(f"\n--- Pass {pass_number}/{int(args.num_train_epochs)} starting ({time.strftime('%H:%M:%S')}) ---", flush=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs) -> None:
        if logs and "loss" in logs:
            self.latest_train_loss = float(logs["loss"])

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs) -> None:
        if not metrics or "eval_macro_strict_f1" not in metrics:
            return
        minutes_so_far = (time.time() - self.training_start_time) / 60
        share_of_steps_done = state.global_step / max(state.max_steps, 1)
        minutes_left_if_no_early_stop = minutes_so_far / share_of_steps_done - minutes_so_far
        loss_text = f"{self.latest_train_loss:.4f}" if self.latest_train_loss is not None else "n/a"
        best_text = f"{state.best_metric:.3f}" if state.best_metric is not None else "n/a"
        print(
            f"--- Pass {round(state.epoch or 0)} done: validation F1 {metrics['eval_macro_strict_f1']:.3f} "
            f"(best so far {best_text}), training loss {loss_text}, {minutes_so_far:.1f} min so far, "
            f"at most {minutes_left_if_no_early_stop:.1f} min left ---",
            flush=True,
        )
