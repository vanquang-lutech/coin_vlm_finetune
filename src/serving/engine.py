"""vLLM-backed inference engine for serving the merged + quantized coin VLM.

This mirrors the generation contract of src.inference.predictor.CoinPredictor
(same prompt, same chat template with thinking disabled, same parse_response
post-processing) but runs the merged/quantized checkpoint through vLLM's
AsyncLLMEngine so the FastAPI app can serve many requests concurrently.

The checkpoint is whatever scripts/export.py produced: a plain merged 16-bit
dir, or an AWQ W4A16 dir (set serving.quantization: awq). vLLM does its own
image preprocessing, so we only apply the chat template to get the prompt text
(with the vision placeholder tokens) and hand the raw PIL image to vLLM via
multi_modal_data.
"""

import io
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.preprocessing import from_config as build_enhancer
from src.evaluate.metrics import parse_response
from src.utils import get_logger

logger = get_logger(__name__)


class VLLMCoinEngine:
    def __init__(self, config):
        from transformers import AutoProcessor
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        self.config = config
        serving = config.serving
        model_path = serving.model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Serving model_path does not exist: {model_path}. "
                "Run scripts/export.py (merge_lora / awq) first."
            )

        # Processor only used to apply the chat template (insert vision
        # placeholders + system/user turns). Image pixels go straight to vLLM.
        logger.info("Loading processor (chat template) from: %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=serving.get("trust_remote_code", True)
        )

        # Resolution override: vLLM does its own image preprocessing using the
        # processor saved in the checkpoint (training-time min/max_pixels). The
        # HF inference path (CoinPredictor) instead overrides these from the
        # inference config. Replicate that here via mm_processor_kwargs so the
        # served resolution matches scripts/inference.py exactly.
        mm_processor_kwargs = self._resolve_mm_processor_kwargs()

        quantization = serving.get("quantization", None) or None
        logger.info(
            "Initializing vLLM AsyncLLMEngine (model=%s, quantization=%s)...",
            model_path, quantization,
        )
        engine_args = AsyncEngineArgs(
            model=model_path,
            quantization=quantization,
            dtype=serving.get("dtype", "auto"),
            max_model_len=serving.get("max_model_len", 4096),
            gpu_memory_utilization=serving.get("gpu_memory_utilization", 0.90),
            max_num_seqs=serving.get("max_num_seqs", 16),
            trust_remote_code=serving.get("trust_remote_code", True),
            tensor_parallel_size=serving.get("tensor_parallel_size", 1),
            enforce_eager=serving.get("enforce_eager", False),
            limit_mm_per_prompt={"image": serving.get("limit_mm_per_prompt_image", 1)},
            mm_processor_kwargs=mm_processor_kwargs or None,
            seed=serving.get("seed", 42),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._sampling_params = self._build_sampling_params()

        # Image enhancement to match the training-time dataset build (CLAHE +
        # unsharp). Without this, the model sees a different image distribution
        # at serve time than it trained on.
        self._preprocess = serving.get("preprocess", None)
        self._enhance_on = bool(self._preprocess and self._preprocess.get("enhance", False))
        self._enhance_mode = (self._preprocess.get("mode", "full")
                              if self._preprocess else "full")
        self._enhancer = build_enhancer(self._preprocess) if self._enhance_on else None
        if self._enhance_on:
            logger.info(
                "Input enhancement ENABLED (mode=%s, clahe_clip=%.2f tile=%s, "
                "unsharp sigma=%.2f amount=%.2f).",
                self._enhance_mode,
                self._enhancer.clahe_clip_limit, self._enhancer.clahe_tile_size,
                self._enhancer.unsharp_sigma, self._enhancer.unsharp_amount,
            )
        else:
            logger.warning(
                "Input enhancement DISABLED. The model was trained on CLAHE+unsharp "
                "enhanced images; serving raw images may reduce accuracy."
            )
        logger.info("vLLM engine ready.")

    def _enhance(self, pil_image: Image.Image) -> Image.Image:
        """Apply the training-time enhancement. CoinEnhancer works on BGR uint8,
        so convert PIL(RGB) -> BGR -> enhance -> RGB -> PIL."""
        if not self._enhance_on:
            return pil_image
        bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        if self._enhance_mode == "smart":
            bgr, _ = self._enhancer.smart_enhance(bgr)
        else:
            bgr = self._enhancer.enhance(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _resolve_mm_processor_kwargs(self) -> dict:
        """Convert config processor min/max_pixels (in vision-token units) to the
        raw-pixel min_pixels/max_pixels vLLM's image processor expects, matching
        CoinPredictor._apply_processor_overrides.

        raw_pixels = token_count * (patch_size * merge_size) ** 2
        """
        processor_config = self.config.get("processor", None)
        if processor_config is None:
            processor_config = self.config.model.get("processor", None)
        if processor_config is None:
            logger.info("No processor min/max_pixels override; vLLM uses the checkpoint default.")
            return {}

        try:
            image_processor = self.processor.image_processor
            patch_size = image_processor.patch_size
            merge_size = getattr(image_processor, "merge_size", 1) or 1
        except AttributeError:
            logger.warning(
                "Processor has no image_processor; skipping resolution override."
            )
            return {}

        token_px = patch_size * merge_size
        min_pixels = int(processor_config.min_pixels * token_px * token_px)
        max_pixels = int(processor_config.max_pixels * token_px * token_px)
        logger.info(
            "Resolution override -> min_pixels=%d, max_pixels=%d raw px "
            "(%d/%d tokens, token_px=%dx%d).",
            min_pixels, max_pixels,
            processor_config.min_pixels, processor_config.max_pixels,
            token_px, token_px,
        )
        return {"min_pixels": min_pixels, "max_pixels": max_pixels}

    def _build_sampling_params(self):
        from vllm import SamplingParams

        gen = self.config.get("generation", {})
        # Greedy unless do_sample is explicitly enabled, matching the predictor.
        do_sample = gen.get("do_sample", False)
        temperature = gen.get("temperature", 0.1) if do_sample else 0.0
        return SamplingParams(
            max_tokens=gen.get("max_new_tokens", 128),
            temperature=temperature,
            top_p=gen.get("top_p", 1.0),
            repetition_penalty=gen.get("repetition_penalty", 1.0),
        )

    def _resolve_prompt(self):
        """Same precedence as CoinPredictor: inference-time prompt first, then
        the training (model) prompt as a fallback."""
        prompt = self.config.get("prompt", None)
        if prompt is not None:
            return prompt
        model_prompt = self.config.model.get("prompt", None)
        if model_prompt is not None:
            return model_prompt
        raise ValueError(
            "Prompt not found. Set `prompt:` in inference.yaml or `model.prompt` "
            "in the model config."
        )

    def _build_prompt_text(self):
        prompt = self._resolve_prompt()
        messages = [
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.user},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return Image.open(path).convert("RGB")

    async def predict(self, image) -> dict:
        """Run a single coin image through vLLM and return the parsed result."""
        pil_image = self._to_pil(image)
        pil_image = self._enhance(pil_image)  # match training-time CLAHE+unsharp
        prompt_text = self._build_prompt_text()

        vllm_prompt = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": pil_image},
        }

        request_id = str(uuid.uuid4())
        final_output = None
        async for output in self.engine.generate(
            vllm_prompt, self._sampling_params, request_id
        ):
            final_output = output

        response = final_output.outputs[0].text if final_output else ""
        parsed = parse_response(response)
        return {
            **(parsed or {}),
            "raw": response,
            "parse_ok": parsed is not None,
        }
