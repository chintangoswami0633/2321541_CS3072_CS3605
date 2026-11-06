"""
Realistic Body Image Generation Module
=======================================
Uses Stable Diffusion XL with ControlNet OpenPose for realistic body visualizations.

This is a proof-of-concept for the dissertation. It requires:
- GPU with 10GB+ VRAM for SDXL (or CPU, but very slow)
- PyTorch, diffusers, controlnet_aux, mediapipe libraries

Author: Chintan (2321541)
"""

import os
import warnings
from typing import Dict

# PIL for image processing
from PIL import Image, ImageFilter, ImageDraw

# Try to import torch and diffusers
try:
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("PyTorch/diffusers not installed. Realistic generation unavailable.")

# ControlNet OpenPose — optional, graceful fallback if not installed
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    from controlnet_aux import OpenposeDetector
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    print("controlnet_aux not installed. Pose locking disabled — install with: pip install controlnet_aux")

# MediaPipe body segmentation — optional, graceful fallback
try:
    import mediapipe as mp
    import numpy as np
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("mediapipe not installed. Torso masking disabled — install with: pip install mediapipe")

warnings.filterwarnings('ignore')


class RealisticPhysiqueGenerator:
    """
    Generates realistic body images using Stable Diffusion XL with ControlNet OpenPose.

    Translates body measurements into text prompts, generates images,
    and applies face masking for privacy.
    """

    MODEL_ID      = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_ID = "thibaud/controlnet-openpose-sdxl-1.0"
    POSE_MODEL_ID = "lllyasviel/ControlNet"

    _cached_pipeline = None  # Class-level cache for model reuse
    _cached_device = None
    _img2img_pipeline_cache = None  # Class-level cache for img2img pipeline
    _img2img_device_cache: str | None = None
    
    def __init__(self, device: str = None, output_dir: str = 'visualizations/realistic/'):
        """
        Initialize the generator.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detect
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = device
        if self.device is None:
            if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        self.pipeline = None
        self.is_loaded = False
        
        print(f"  RealisticPhysiqueGenerator initialized")
        print(f"  Device: {self.device}")
        print(f"  Diffusers available: {DIFFUSERS_AVAILABLE}")
        
    def load_model(self):
        """Load the SDXL pipeline (with ControlNet if available) with class-level caching."""
        if not DIFFUSERS_AVAILABLE:
            print("  Cannot load model - diffusers not installed")
            return False

        # Return cached pipeline if device matches
        if (RealisticPhysiqueGenerator._cached_pipeline is not None and
                RealisticPhysiqueGenerator._cached_device == self.device):
            print("  Using cached model (instant load)")
            self.pipeline = RealisticPhysiqueGenerator._cached_pipeline
            self.pose_detector = getattr(RealisticPhysiqueGenerator, '_cached_pose_detector', None)
            self.is_loaded = True
            return True

        if self.is_loaded:
            return True

        print(f"\n  Loading SDXL model...")
        print(f"  This may take several minutes on first run...")

        try:
            dtype = torch.float16 if self.device == 'cuda' else torch.float32

            if CONTROLNET_AVAILABLE:
                print("  Loading ControlNet OpenPose for pose locking...")
                controlnet = ControlNetModel.from_pretrained(
                    self.CONTROLNET_ID,
                    torch_dtype=dtype,
                )
                self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    self.MODEL_ID,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    use_safetensors=True,
                ).to(self.device)
                self.pose_detector = OpenposeDetector.from_pretrained(self.POSE_MODEL_ID)
                RealisticPhysiqueGenerator._cached_pose_detector = self.pose_detector
                print("  ControlNet OpenPose loaded")
            else:
                # Fallback: plain SDXL img2img without pose locking
                self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=dtype,
                    variant="fp16" if self.device == 'cuda' else None,
                    use_safetensors=True,
                ).to(self.device)
                self.pose_detector = None
                print("  SDXL loaded (no ControlNet — install controlnet_aux for pose locking)")

            if self.device == 'cuda':
                self.pipeline.enable_attention_slicing()

            RealisticPhysiqueGenerator._cached_pipeline = self.pipeline
            RealisticPhysiqueGenerator._cached_device   = self.device

            self.is_loaded = True
            print("  Model loaded successfully (cached for future use)")
            return True

        except Exception as e:
            print(f"  Error loading model: {str(e)}")
            return False
    
    @staticmethod
    def _calculate_bf_reduction_factor(total_weeks: int, diet_controlled: bool = True) -> float:
        """
        Compute the maximum body fat reduction factor achievable over total_weeks.

        Based on McDonald (2008) and Aragon & Schoenfeld (2013) diminishing returns:
        - No diet: very limited fat loss regardless of duration
        - ≤12 weeks: linear phase (~1.25% BF per week)
        - ≤24 weeks: slower linear phase after initial adaptation
        - >24 weeks: logarithmic plateau (diminishing returns)
        """
        if not diet_controlled:
            return min(0.08, total_weeks * 0.002)
        if total_weeks <= 12:
            return total_weeks * 0.0125                                    # up to 15% at 12wk
        if total_weeks <= 24:
            return 0.15 + (total_weeks - 12) * 0.0085                     # up to 25.2% at 24wk
        return min(0.45, 0.15 + 0.10 + ((total_weeks - 24) / 12) * 0.07)  # plateau toward 45%

    @staticmethod
    def _calculate_muscle_growth_plateau(current_week: int, total_weeks: int) -> float:
        """
        Plateau multiplier for muscle growth (Aragon & Schoenfeld 2013).
        Early weeks: full rate. After ~20 weeks: diminishing returns.
        """
        if current_week <= 20:
            return 1.0
        if current_week <= 40:
            return 1.0 - ((current_week - 20) / 20) * 0.4   # 1.0 → 0.6
        return 0.25

    def _get_transformation_descriptors(self, week: int, total_weeks: int) -> dict:
        """
        Return per-week physique descriptors calibrated to realistic natural gym progress.

        The descriptor tier is capped by total_weeks so a 4-week plan can never
        produce "visible abs" language regardless of week number.
        """
        progress = week / max(total_weeks, 1)

        # Hard cap on how advanced the descriptor can be based on plan length
        # 4 weeks max = tier 0, 8 weeks max = tier 1, 12 weeks max = tier 2, etc.
        if total_weeks <= 4:
            max_tier = 0
        elif total_weeks <= 8:
            max_tier = 1
        elif total_weeks <= 16:
            max_tier = 2
        elif total_weeks <= 32:
            max_tier = 3
        else:
            max_tier = 4

        # Determine tier from progress ratio
        if progress < 0.25:
            tier = 0
        elif progress < 0.50:
            tier = 1
        elif progress < 0.75:
            tier = 2
        elif progress < 0.90:
            tier = 3
        else:
            tier = 4

        tier = min(tier, max_tier)

        tiers = [
            {   # tier 0 — very early / short plan
                "physique_desc": "natural physique, slight muscle tone, soft midsection, no visible abs, realistic starting point with minimal change",
                "chest_desc":    "very subtle chest improvement, barely noticeable pectoral development, chest almost identical to starting point",
                "arm_desc":      "average arm size, no visible bicep peak",
                "abs_desc":      "soft midsection, no ab definition whatsoever",
                "fat_desc":      "19-22% body fat, smooth skin over midsection",
                "intensity":     "very subtle early gym progress, barely noticeable change from baseline",
            },
            {   # tier 1 — 2-3 months
                "physique_desc": "early gym progress, slightly improved muscle tone, midsection beginning to tighten, soft definition emerging in arms",
                "chest_desc":    "mild chest development, slightly fuller pectorals, natural modest improvement, no dramatic chest size, realistic modest pec growth",
                "arm_desc":      "slightly larger arms, faint bicep shape",
                "abs_desc":      "midsection slightly tighter, no visible abs",
                "fat_desc":      "17-20% body fat, slight reduction in waist softness",
                "intensity":     "realistic 2-3 month natural beginner progress, subtle but visible improvement",
            },
            {   # tier 2 — 4-6 months
                "physique_desc": "moderate muscle development, noticeable definition in arms and shoulders, midsection tighter, upper abs faintly visible",
                "chest_desc":    "moderate chest development, natural pectoral definition, realistic chest for someone who has trained consistently, not oversized, proportionate to body",
                "arm_desc":      "defined biceps with faint peak, arms noticeably more toned",
                "abs_desc":      "upper abs faintly visible, midsection clearly tighter than starting point",
                "fat_desc":      "15-18% body fat, visible improvement in body composition",
                "intensity":     "realistic 4-6 month natural intermediate progress",
            },
            {   # tier 3 — 7-9 months
                "physique_desc": "well-developed athletic physique, clear definition throughout, lean midsection, full muscle bellies",
                "chest_desc":    "well-developed but natural chest, proportionate pectorals, realistic chest for dedicated training, not a powerlifter chest, normal athletic chest size",
                "arm_desc":      "prominent bicep peak, full tricep development",
                "abs_desc":      "upper 4-pack abs visible, obliques beginning to show",
                "fat_desc":      "13-16% body fat, lean athletic look",
                "intensity":     "realistic 7-9 month dedicated natural progress",
            },
            {   # tier 4 — 10-12 months
                "physique_desc": "lean athletic physique, excellent muscle definition, impressive natural transformation, full year of training",
                "chest_desc":    "well-developed but natural chest, proportionate pectorals, realistic chest for dedicated training, not a powerlifter chest, normal athletic chest size",
                "arm_desc":      "well-developed arms with clear bicep peak and tricep definition",
                "abs_desc":      "clear 6-pack abs, lean midsection, visible obliques",
                "fat_desc":      "10-13% body fat, lean and defined",
                "intensity":     "realistic 10-12 month natural dedicated gym progress",
            },
        ]
        return tiers[tier]

    def _get_denoising_strength(self, week: int, total_weeks: int) -> float:
        """
        Timeline-aware denoising strength.

        Wider range = more visible progression between weeks.
        Identity anchoring in the prompt compensates at higher strengths.

        Ceiling by total_weeks:
          ≤4  weeks: 0.28
          ≤8  weeks: 0.32
          ≤12 weeks: 0.38
          ≤24 weeks: 0.46
          >24 weeks: 0.55

        Floor: 0.22 so SD always makes at least a small change.
        """
        progress = week / max(total_weeks, 1)

        if total_weeks <= 4:
            ceiling = 0.28
        elif total_weeks <= 8:
            ceiling = 0.32
        elif total_weeks <= 12:
            ceiling = 0.38
        elif total_weeks <= 24:
            ceiling = 0.46
        else:
            ceiling = 0.55

        strength = 0.22 + progress * (ceiling - 0.22)
        return round(max(0.22, min(strength, ceiling)), 3)

    def _get_guidance_scale(self, week: int, total_weeks: int) -> float:
        """
        Guidance scale — fixed at 7.0 for all weeks and all timelines.
        Ramping guidance alongside strength causes exponential distortion.
        """
        return 7.0

    @staticmethod
    def _get_week_generation_params(week: int, total_weeks: int) -> dict:
        """
        Compatibility shim — returns strength and guidance.
        Called by image_generator.py for per-frame params.
        """
        progress = week / max(total_weeks, 1)

        if total_weeks <= 4:
            ceiling = 0.28
        elif total_weeks <= 8:
            ceiling = 0.32
        elif total_weeks <= 12:
            ceiling = 0.38
        elif total_weeks <= 24:
            ceiling = 0.46
        else:
            ceiling = 0.55

        strength = round(max(0.22, min(0.22 + progress * (ceiling - 0.22), ceiling)), 3)
        return {"strength": strength, "guidance": 7.5}

    def _extract_skin_tone_description(self, reference_image) -> str:
        """
        Sample the torso region of the reference photo and return a skin tone
        description string for use in the prompt. Avoids background and clothing
        by sampling the central 25-75% width, 20-70% height band.
        """
        import numpy as np
        img_array = np.array(reference_image.convert("RGB"))
        h, w = img_array.shape[:2]
        torso = img_array[int(h * 0.20):int(h * 0.70), int(w * 0.25):int(w * 0.75)]
        r = int(np.median(torso[:, :, 0]))
        g = int(np.median(torso[:, :, 1]))
        b = int(np.median(torso[:, :, 2]))
        brightness = (r + g + b) / 3
        if brightness > 200:
            return "very fair caucasian skin tone, pale complexion"
        elif brightness > 170:
            return "fair caucasian skin tone, light complexion"
        elif brightness > 140:
            return "medium caucasian skin tone, natural complexion"
        elif brightness > 110:
            return "medium-tan skin tone, olive complexion"
        elif brightness > 80:
            return "tan brown skin tone, South Asian or Mediterranean complexion"
        elif brightness > 55:
            return "dark brown skin tone, darker complexion"
        else:
            return "deep dark skin tone, very dark complexion"

    def _build_prompt(self, week: int, total_weeks: int,
                      gender: str, skin_tone_desc: str) -> str:
        """Build the positive prompt for a given week."""
        d = self._get_transformation_descriptors(week, total_weeks)
        return (
            f"professional fitness progress photo, {gender}, "
            f"(same person as reference photo:1.5), (same background:1.4), (same clothing:1.3), "
            f"(same pose:1.3), (same lighting:1.2), "
            f"{d['physique_desc']}, "
            f"{d['chest_desc']}, "
            f"proportionate chest relative to arms and shoulders, chest development in line with overall physique, balanced muscle development across all muscle groups, "
            f"arms and shoulders are the most visibly developed muscle groups, overall balanced physique improvement, chest is secondary to arm and shoulder development, "
            f"{d['arm_desc']}, "
            f"{d['abs_desc']}, "
            f"{d['fat_desc']}, "
            f"{skin_tone_desc}, consistent skin tone throughout, "
            f"no skin discolouration, no skin lightening, "
            f"same pose, frontal view, "
            f"natural stance arms at sides, "
            f"photorealistic DSLR photo, studio lighting, "
            f"natural gym transformation, {d['intensity']}"
        )

    def _build_negative_prompt(self, week: int, total_weeks: int) -> str:
        """Build the negative prompt for a given week."""
        base = (
            "bodybuilder, competition physique, unrealistic muscle, "
            "steroids, photoshopped, CGI, cartoon, anime, illustration, "
            "painting, digital art, "
            "ugly, deformed, bad anatomy, extra limbs, missing limbs, "
            "skin darkening, skin lightening, skin discolouration, "
            "leathery skin, orange skin, grey skin, blotchy skin, "
            "over-saturated, over-contrasted, blown out highlights, "
            "noisy skin, grainy texture, rough skin, skin lesions, "
            "different person, different background, different clothing, "
            "side view, back view, rotated torso, leaning body, "
            "arms raised, wide stance, different pose, "
            "extreme vascularity, veins covering entire body, "
            "full body tattoos, face visible, unblurred face, "
            "oversized chest, massive pectorals, huge chest, overly developed pecs, "
            "chest too big, unnatural chest size, barrel chest, pectoral implants, "
            "disproportionate chest, chest larger than shoulders"
        )
        # Only suppress dramatic results for short plans or very early weeks (first 25%)
        if total_weeks <= 8 or (week / max(total_weeks, 1)) < 0.25:
            base += (
                ", six pack abs, visible abs, shredded, ripped, very lean, "
                "visible striations, extremely muscular, "
                "fitness model, athlete physique, competition ready, "
                "vascular, veins showing, deep muscle cuts, "
                "dramatic transformation, impressive physique"
            )
        return base

    @staticmethod
    def _estimate_background_complexity(image) -> float:
        """
        Estimate background complexity of a reference photo (0.0 = plain, 1.0 = complex).

        Samples the border pixels (top/bottom/left/right strips) and computes
        normalised colour variance. A plain white/grey studio backdrop has near-zero
        variance; a gym with equipment, mirrors, and coloured walls scores high.
        """
        import numpy as np
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        h, w = img_np.shape[:2]

        strip = max(1, int(min(h, w) * 0.08))  # 8% border strip

        border = np.concatenate([
            img_np[:strip, :].reshape(-1, 3),          # top
            img_np[-strip:, :].reshape(-1, 3),         # bottom
            img_np[:, :strip].reshape(-1, 3),          # left
            img_np[:, -strip:].reshape(-1, 3),         # right
        ], axis=0)

        # Normalised std across all border pixels and channels
        variance = float(np.std(border) / 128.0)
        return min(1.0, variance)

    def _match_skin_tone(self, generated_img: Image.Image, reference_img: Image.Image,
                         blend_weight: float = 0.65) -> Image.Image:
        """
        Transfer skin tone from reference photo to generated image using LAB colour matching.

        Matches A and B channels (colour) only — L (luminance/shading) is preserved
        so muscle definition from the generation is kept intact.

        blend_weight=0.65 means 65% reference colour, 35% generated colour.
        Falls back gracefully if cv2 is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return generated_img

        try:
            gen_arr = np.array(generated_img.convert("RGB"))
            ref_arr = np.array(reference_img.convert("RGB"))

            if gen_arr.shape != ref_arr.shape:
                ref_arr = cv2.resize(ref_arr, (gen_arr.shape[1], gen_arr.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

            gen_lab = cv2.cvtColor(gen_arr, cv2.COLOR_RGB2LAB).astype(float)
            ref_lab = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2LAB).astype(float)
            corrected = gen_lab.copy()

            for ch in [1, 2]:  # A and B only — preserve L
                g_mean = gen_lab[:, :, ch].mean()
                g_std  = gen_lab[:, :, ch].std()
                r_mean = ref_lab[:, :, ch].mean()
                r_std  = ref_lab[:, :, ch].std()
                normalised = (gen_lab[:, :, ch] - g_mean) / (g_std + 1e-6)
                matched    = normalised * r_std + r_mean
                corrected[:, :, ch] = (
                    blend_weight * matched + (1.0 - blend_weight) * gen_lab[:, :, ch]
                )

            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB))
        except Exception:
            return generated_img

    def _apply_face_mask(self, image: Image.Image, blur_radius: int = 35) -> Image.Image:
        """
        Blur the face region for privacy.
        Uses OpenCV face/eye detection first; falls back to blurring a centred
        rectangle in the top 15% of the image if no face is detected.
        """
        width, height = image.size
        region = self._detect_face_region(image, width, height)

        # Fallback: blur a centred band across the top 15% of the image.
        # Full-width blur at 22% was too aggressive — covered neck/shoulders.
        # Centred at 20–80% width keeps the sides (background) unblurred.
        if region is None:
            region = (int(width * 0.10), 0, int(width * 0.90), int(height * 0.15))

        x1, y1, x2, y2 = region
        face_region  = image.crop((x1, y1, x2, y2))
        blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        result       = image.copy()
        result.paste(blurred_face, (x1, y1))
        return result

    def _detect_face_region(
        self, image: Image.Image, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Return (x1, y1, x2, y2) of the region to blur, or None if no face found."""
        try:
            import cv2
            import numpy as np

            img_np = np.array(image.convert("RGB"))
            gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            valid_faces = []
            for min_neighbors in (3, 5):
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=min_neighbors, minSize=(30, 30)
                )
                valid_faces = [
                    (fx, fy, fw, fh) for (fx, fy, fw, fh) in faces
                    if (fy + fh / 2) < height * 0.55 and fw >= width * 0.06
                ]
                if valid_faces:
                    break

            if not valid_faces:
                eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
                eyes = eye_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
                )
                valid_eyes = [e for e in eyes if (e[1] + e[3] / 2) < height * 0.45]
                if valid_eyes:
                    ex  = min(e[0] for e in valid_eyes)
                    ey  = min(e[1] for e in valid_eyes)
                    ex2 = max(e[0] + e[2] for e in valid_eyes)
                    ey2 = max(e[1] + e[3] for e in valid_eyes)
                    valid_faces = [(ex, ey, ex2 - ex, ey2 - ey)]

            if valid_faces:
                fx, fy, fw, fh = max(valid_faces, key=lambda r: r[2] * r[3])
                pad_x        = int(fw * 0.30)          # horizontal padding
                pad_top      = int(fh * 0.40)          # above forehead (hair)
                pad_bottom   = int(fh * 0.10)          # just past chin — no neck/chest
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_top)
                x2 = min(width, fx + fw + pad_x)
                y2 = min(int(height * 0.40), fy + fh + pad_bottom)
                return x1, y1, x2, y2
        except Exception:
            pass

        return None
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply post-processing enhancements to improve image quality.
        
        Args:
            image: PIL Image
            
        Returns:
            Enhanced image
        """
        from PIL import ImageEnhance
        
        # Slight sharpness increase for better definition
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Slight contrast increase for better muscle definition
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.02)
        
        # Slight color saturation for more natural skin tone
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def _add_watermark(self, image: Image.Image) -> Image.Image:
        """Add a subtle 'AI Generated' watermark to the bottom-right corner."""
        from PIL import ImageDraw, ImageFont
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        w, h = watermarked.size
        text = "AI Generated"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = w - text_w - 12
        y = h - text_h - 12
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 180))
        return watermarked

    def generate_timeline(
        self,
        reference_image: Image.Image,
        timeline_weeks: list,
        total_weeks: int,
        gender: str,
        user_seed: int,
    ) -> list:
        """
        Generate a timeline of physique images from a reference photo.

        Each week is generated independently from the ORIGINAL reference image
        (no chaining) using a fixed seed derived from the user's session so that
        background, lighting, and pose stay consistent across all frames.

        Returns a list of (week, PIL.Image) tuples, starting with (0, reference_image)
        as the unmodified "Week 0 — you today" anchor frame.

        Args:
            reference_image: The user's uploaded photo (PIL Image, RGB).
            timeline_weeks:  List of week numbers to generate, e.g. [7, 13, 20, 26].
            total_weeks:     Total training duration (used to scale descriptors).
            gender:          'male' or 'female'.
            user_seed:       Fixed integer seed — use hash(user_id) % (2**32) so all
                             weeks for the same user share the same seed, keeping
                             background, lighting, and pose consistent.
        """
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

        if not DIFFUSERS_AVAILABLE:
            return [(0, reference_image.copy())]

        # generate_timeline uses SD 1.5 (Realistic Vision) — NOT SDXL.
        # Do NOT call self.load_model() here: that loads SDXL (~7GB VRAM) which
        # would leave insufficient VRAM for the SD 1.5 pipeline on most GPUs.
        device = self.device
        dtype  = torch.float16 if device == "cuda" else torch.float32

        # If SDXL was previously loaded into VRAM, offload it to free space
        if (
            RealisticPhysiqueGenerator._cached_pipeline is not None
            and RealisticPhysiqueGenerator._cached_device == "cuda"
            and device == "cuda"
        ):
            try:
                RealisticPhysiqueGenerator._cached_pipeline.to("cpu")
                torch.cuda.empty_cache()
                print("  SDXL offloaded to CPU to free VRAM for img2img")
            except Exception:
                pass

        # Use SD 1.5-based Realistic Vision — much faster than SDXL (2GB vs 6.9GB,
        # ~3-5x faster per frame).
        FAST_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"

        # Use class-level cache so we don't reload on every call
        if (
            not hasattr(RealisticPhysiqueGenerator, "_img2img_pipeline_cache")
            or RealisticPhysiqueGenerator._img2img_pipeline_cache is None
            or RealisticPhysiqueGenerator._img2img_device_cache != device
        ):
            print(f"  Loading Realistic Vision img2img pipeline on {device}...")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                FAST_MODEL_ID,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
            # DPMSolver++ gives equal quality at 12 steps vs DDIM at 20 (~40% faster)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++"
            )
            if device == "cuda":
                pipe.enable_attention_slicing()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("  xformers memory-efficient attention enabled")
                except Exception:
                    pass
            RealisticPhysiqueGenerator._img2img_pipeline_cache = pipe
            RealisticPhysiqueGenerator._img2img_device_cache   = device
            print(f"  img2img pipeline loaded on {device}")
        pipe = RealisticPhysiqueGenerator._img2img_pipeline_cache

        # Resize reference to 512×768 for SD
        from PIL import Image as _PIL
        target_w, target_h = 512, 768
        orig_w, orig_h = reference_image.size
        scale   = max(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        ref_resized = reference_image.resize((new_w, new_h), _PIL.LANCZOS)
        left = (new_w - target_w) // 2
        top  = (new_h - target_h) // 2
        ref_resized = ref_resized.crop((left, top, left + target_w, top + target_h))

        skin_tone_desc = self._extract_skin_tone_description(reference_image)

        # Week 0 = reference resized to match generated frames — gives users a
        # clear before/after anchor at the same resolution and crop as the AI frames.
        # Apply face mask to the reference frame too for consistent privacy.
        ref_week0 = self._apply_face_mask(ref_resized.copy())
        timeline_outputs = [(0, ref_week0)]

        for week in timeline_weeks:
            denoising  = self._get_denoising_strength(week, total_weeks)
            guidance   = self._get_guidance_scale(week, total_weeks)
            prompt     = self._build_prompt(week, total_weeks, gender, skin_tone_desc)
            neg_prompt = self._build_negative_prompt(week, total_weeks)

            # Per-frame seed: user_seed + week keeps identity stable (same base)
            # while giving each frame a slightly different noise pattern so
            # progression is visually distinct rather than identical-looking.
            frame_seed = (user_seed + week) & 0xFFFFFFFF
            g = torch.Generator(device=device).manual_seed(frame_seed)

            # Clamp denoising strength to a safe ceiling that preserves identity.
            # Stronger identity anchoring in the prompt (same person:1.5) compensates
            # at higher strengths, allowing more visible progression.
            safe_denoising = min(denoising, 0.55)

            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=ref_resized,
                strength=safe_denoising,
                guidance_scale=7.5,
                num_inference_steps=20,  # 20 steps gives the prompt enough influence to hold pose/bg
                generator=g,
            )
            generated = result.images[0]

            # Higher blend weight keeps skin tone anchored to reference
            generated = self._match_skin_tone(generated, ref_resized, blend_weight=0.80)
            generated = self._apply_face_mask(generated)
            generated = self._enhance_image(generated)
            generated = self._add_watermark(generated)

            timeline_outputs.append((week, generated))

        return timeline_outputs

    
    
    

