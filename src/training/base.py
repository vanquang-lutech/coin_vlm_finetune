from abc import ABC, abstractmethod
from src.utils import get_logger
from omegaconf import DictConfig
from .callbacks import GradNormCallback, MemoryCallback

logger = get_logger(__name__)

class BaseTrainer(ABC):
    def __init__(self, config, model, processor, train_loader, val_loader):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        logger.info("Starting training...")
        args = self._build_args()
        self.trainer = self._build_trainer(args)

        logger.info("Starting training with args: %s", args)
        result = self._run_training()
        return result
    
    def get_model(self):
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
        return self.trainer.model
    
    def _build_args(self):
        from transformers import TrainingArguments
        training_args = self.config.training
        return TrainingArguments(
            output_dir = training_args.output_dir,
            num_train_epochs = training_args.num_train_epochs,
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            gradient_accumulation_steps= training_args.gradient_accumulation_steps,
            learning_rate= training_args.learning_rate,
            lr_scheduler_type= training_args.lr_scheduler_type,
            warmup_ratio= training_args.warmup_ratio,
            weight_decay= training_args.weight_decay,
            max_grad_norm= training_args.max_grad_norm,
            bf16= training_args.bf16,
            fp16= training_args.fp16,
            eval_strategy= training_args.eval_strategy,
            eval_steps= training_args.eval_steps,
            save_strategy= training_args.save_strategy,
            save_steps= training_args.save_steps,
            save_total_limit= training_args.save_total_limit,
            load_best_model_at_end= training_args.load_best_model_at_end,
            metric_for_best_model= training_args.metric_for_best_model,
            logging_steps= training_args.logging_steps,
            dataloader_num_workers= training_args.dataloader_num_workers,
            dataloader_pin_memory= training_args.dataloader_pin_memory,
            seed= training_args.seed,
            remove_unused_columns= False, # dataset return image + label dict, no need to remove columns
            report_to= training_args.report_to,
            run_name= training_args.run_name,
        )

    def _run_training(self):
        result = self.trainer.train()
        logger.info(
            "Training complete. "
            "Steps: %d | Train loss: %.4f",
            result.global_step,
            result.training_loss,
        )
        self.trainer.save_model()  # Save final model
        self.processor.save_pretrained(self.trainer.args.output_dir)  # Save processor config
        logger.info(f"Model and processor saved to {self.trainer.args.output_dir}")
        return result

    @abstractmethod
    def _build_trainer(self, args):
        pass
    
    def _get_callbacks(self):
        callbacks = [GradNormCallback(), MemoryCallback()]

        if self.config.training.early_stopping_patience.enabled:
            from transformers import EarlyStoppingCallback
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience.patience,
                    early_stopping_threshold=self.config.training.early_stopping_patience.threshold,
                )
            )
        return callbacks

