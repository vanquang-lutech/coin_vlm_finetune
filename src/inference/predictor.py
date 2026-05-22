import logging
from pathlib import Path
import torch
from PIL import Image

from src.evaluate.metrics import parse_response

logger = logging.getLogger(__name__)

class CoinPredictor:
    def __init__(self, config, model=None, processor=None, checkpoint_path=None):
        self.config = config 
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        
        elif checkpoint_path is not None:
            self.model, self.processor = self._load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(
                "Must provide either (model + processor) or checkpoint_path"
            )
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def predict(self, image):
        pil_image = self._load_image(image)
        messages = self._build_messages()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,    
        )

        response = self._generate([pil_image], [text])[0]
        parsed = parse_response(response)

        return {
            **parsed,
            "raw": response,
            "parse_ok": parsed is not None,
        }
    
    def predict_batch(self, images, batch_size):
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            pil_images = [self._load_image(img) for img in batch_images]
            text = self.processor.apply_chat_template(
                self._build_messages(),
                tokenize=False,
                add_generation_prompt=True,
            )

            texts = [text] * len(pil_images)
            responses = self._generate(pil_images, texts)

            for response in responses:
                parsed = parse_response(response)
                results.append({
                    **parsed,
                    "raw": response,
                    "parse_ok": parsed is not None,
                })
            logger.info("Processed batch %d-%d / %d", i, min(i + batch_size, len(images)), len(images))
        return results


    def _load_from_checkpoint(self, checkpoint_path):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Loading model and processor from checkpoint: %s", checkpoint_path)
        processor = AutoProcessor.from_pretrained(checkpoint_path)

        if (checkpoint_path / "adapter_config.json").exists():
            from peft import PeftModel
            logger.info("Detected LoRA adapter, loading base model + adapter...")
            base_model = AutoModelForImageTextToText.from_pretrained(
                self.config.model.model_name,
                torch_dtype = torch.bfloat16,
                device_map  = "auto",
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            logger.info("Loading full model from checkpoint...")
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype = torch.bfloat16,
                device_map  = "auto",
            )
            
        model.eval()
        return model, processor
        
    def _generate(self, images, texts):
        generation_config = self.config.get("generation", {})
        inputs = self.processor(
            images = images, 
            text = texts,
            return_tensors = "pt",
            padding = True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = generation_config.get("max_new_tokens", 256),
                do_sample = generation_config.get("do_sample", False),
                temperature = generation_config.get("temperature", 0.1),
                top_p = generation_config.get("top_p", 1.0),
                pad_token_id = self.processor.tokenizer.pad_token_id,
                eos_token_id = self.processor.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )
        

    def _load_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return Image.open(path).convert("RGB")
    
    def _build_messages(self):
        prompt = self.config.get("prompt", None)
        if prompt is None:
            prompt = self.config.data.prompt
        return [
            {
                "role": "system",
                "content": prompt.system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.user},
                ],
            },
        ]

