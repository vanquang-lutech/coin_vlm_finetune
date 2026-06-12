from transformers import Trainer, TrainingArguments
from src.utils import get_logger
from src.data.collator import CoinDataCollator
from .base import BaseTrainer

logger = get_logger(__name__)

class FullFinetuneTrainer(BaseTrainer):
    def _build_args(self):
        training_config = self.config.training
        if not training_config.get("deepspeed", None):
            logger.warning("DeepSpeed config not found. For full finetuning, it's recommended to use DeepSpeed for better performance and memory efficiency.")
        
        return TrainingArguments(
            output_dir= training_config.output_dir,
            num_train_epochs= training_config.num_train_epochs, 
            per_device_train_batch_size= training_config.per_device_train_batch_size,
            per_device_eval_batch_size= training_config.per_device_eval_batch_size,
            gradient_accumulation_steps= training_config.gradient_accumulation_steps,
            learning_rate= training_config.learning_rate,
            lr_scheduler_type= training_config.lr_scheduler_type,
            warmup_steps= training_config.warmup_steps,
            weight_decay= training_config.weight_decay,
            max_grad_norm= training_config.max_grad_norm,
            bf16= training_config.bf16,
            fp16= training_config.fp16,
            eval_strategy= training_config.eval_strategy,
            eval_steps= training_config.eval_steps,
            save_strategy= training_config.save_strategy,
            save_steps= training_config.save_steps,
            save_total_limit= training_config.save_total_limit,
            load_best_model_at_end= training_config.load_best_model_at_end,
            metric_for_best_model= training_config.metric_for_best_model,
            logging_steps= training_config.logging_steps,
            report_to= training_config.report_to,
            run_name= training_config.run_name,
            dataloader_num_workers= training_config.dataloader_num_workers,
            dataloader_pin_memory= training_config.dataloader_pin_memory,
            seed= training_config.seed,
            remove_unused_columns= False, # dataset return image + label dict, no need to remove columns
            deepspeed= training_config.deepspeed, # DeepSpeed config path or dict

        )
    
    def _build_trainer(self, args):
        collactor = CoinDataCollator(self.processor, self.config)
        trainer = Trainer(
            model = self.model,
            args = args,
            train_dataset = self.train_loader,
            eval_dataset = self.val_loader,
            data_collator = collactor,
            callbacks = self._get_callbacks(),
        )
        return trainer