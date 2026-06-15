from src.utils import get_logger, report_to_list
from src.data.dataset import CoinDataset
from src.data.collator import CoinDataCollator
from .base import BaseTrainer

logger = get_logger(__name__)

class UnslothTrainer(BaseTrainer):
    def _build_args(self):
        from unsloth import is_bf16_supported
        from trl import SFTConfig
        training_config = self.config.training

        return SFTConfig(
            output_dir = training_config.output_dir,
            num_train_epochs = training_config.num_train_epochs,
            per_device_train_batch_size = training_config.per_device_train_batch_size,
            per_device_eval_batch_size = training_config.per_device_eval_batch_size,
            gradient_accumulation_steps= training_config.gradient_accumulation_steps,
            learning_rate = training_config.learning_rate,
            lr_scheduler_type = training_config.lr_scheduler_type,
            warmup_steps = training_config.warmup_steps,
            weight_decay = training_config.weight_decay,
            max_grad_norm = training_config.max_grad_norm,
            bf16 = is_bf16_supported(),
            fp16 = not is_bf16_supported(),
            eval_strategy = training_config.eval_strategy,
            eval_steps = training_config.eval_steps,
            save_strategy = training_config.save_strategy,
            save_steps = training_config.save_steps,
            save_total_limit = training_config.save_total_limit,
            load_best_model_at_end = training_config.load_best_model_at_end,
            metric_for_best_model = training_config.metric_for_best_model,
            greater_is_better = training_config.get("greater_is_better", None),
            logging_steps = training_config.logging_steps,
            dataloader_num_workers = training_config.dataloader_num_workers,
            dataloader_pin_memory = training_config.dataloader_pin_memory,
            seed = training_config.seed,
            remove_unused_columns= False, # dataset return image + label dict, no need to remove columns
            dataset_kwargs={"skip_prepare_dataset": True}, # handle data preparation in the Dataset class, so skip it in the trainer
            dataset_text_field="", # dataset will return dict with "image" and "label" keys, no need to specify text field
            max_seq_length = training_config.max_seq_length,
            report_to= report_to_list(training_config.report_to),
            run_name= training_config.run_name,
        )
    
    def _build_trainer(self, args):
        from trl import SFTTrainer
        from unsloth import FastVisionModel

        FastVisionModel.for_training(self.model)
        collator = CoinDataCollator(self.processor, self.config)
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.processor,
            args = args,
            train_dataset = self.train_loader,
            eval_dataset = self.val_loader,
            data_collator = collator,
            callbacks = self._get_callbacks(),
        )
        return trainer