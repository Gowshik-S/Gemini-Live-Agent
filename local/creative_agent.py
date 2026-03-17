"""
Rio Local — Creative Agent (Priority 9)

Handles creative tasks: image generation via Imagen 3,
text generation via Gemini, and interleaved text+image output.

Usage::

    agent = CreativeAgent(api_key="...")
    result = await agent.generate_image("A sunset over mountains")
    result = await agent.generate_text("Write a haiku about coding")
    result = await agent.execute_step(
        "Create a logo for a coding assistant called Rio",
        expected_outcome="A professional logo image",
    )
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import structlog

from constants import MODEL_FLASH, MODEL_IMAGEN

log = structlog.get_logger(__name__)

# Attempt imports
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# Output directory for generated images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "creative")


def load_prompt_from_markdown(filename: str) -> str:
    """Load system prompt from a markdown file in the rio directory."""
    rio_dir = Path(__file__).resolve().parent.parent
    md_path = rio_dir / filename
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""

def get_creative_agent_prompt() -> str:
    """Load the Creative Agent prompt from markdown, or fallback to default."""
    import re
    md_content = load_prompt_from_markdown("creative_agent.md")
    if md_content:
        pattern = r'CREATIVE_AGENT_SYSTEM_PROMPT\s*=\s*"""(.*?)"""'
        match = re.search(pattern, md_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return md_content.strip()
        
    return "You are a creative assistant. Be creative, engaging, and concise."


class CreativeAgent:
    """Creative content generation agent using Gemini + Imagen.

    Supports:
      - Image generation via Imagen 3
      - Creative text generation via Gemini Flash
      - Combined text+image workflows
      - Image description/analysis
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client: Optional[genai.Client] = None
        self._log = log.bind(component="creative_agent")

        if _GENAI_AVAILABLE and self._api_key:
            self._client = genai.Client(api_key=self._api_key)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    @property
    def available(self) -> bool:
        return _GENAI_AVAILABLE and self._client is not None

    # ------------------------------------------------------------------
    # Image Generation
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        style: str = "",
        save: bool = True,
    ) -> dict:
        """Generate an image using Imagen 3.

        Args:
            prompt: Description of the image to generate.
            style: Optional style modifier (e.g., "photorealistic", "digital art").
            save: Whether to save the image to disk.

        Returns:
            {"success": bool, "image_path": str, "image_base64": str, ...}
        """
        if not self.available:
            return {"success": False, "error": "Creative agent not available"}

        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"

        self._log.info("creative.generate_image", prompt=full_prompt[:100])

        try:
            response = await self._client.aio.models.generate_images(
                model=MODEL_IMAGEN,
                prompt=full_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                ),
            )

            if not response.generated_images:
                return {"success": False, "error": "No image generated"}

            image = response.generated_images[0]
            image_bytes = image.image.image_bytes

            result = {
                "success": True,
                "prompt": full_prompt,
                "size_kb": round(len(image_bytes) / 1024, 1),
            }

            if save:
                filename = f"rio_image_{int(time.time())}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                result["image_path"] = filepath
                self._log.info("creative.image_saved", path=filepath)

            # Include base64 for sending via WS if needed
            result["image_base64"] = base64.b64encode(image_bytes).decode("ascii")

            return result

        except Exception as exc:
            self._log.exception("creative.generate_image_failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Text Generation
    # ------------------------------------------------------------------

    async def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        max_tokens: int = 2048,
    ) -> dict:
        """Generate creative text using Gemini Flash.

        Args:
            prompt: The creative prompt.
            system_instruction: Optional system instruction for persona/style.
            max_tokens: Maximum output tokens.

        Returns:
            {"success": bool, "text": str, ...}
        """
        if not self.available:
            return {"success": False, "error": "Creative agent not available"}

        self._log.info("creative.generate_text", prompt=prompt[:100])

        try:
            config = types.GenerateContentConfig(
                temperature=0.9,  # Higher creativity
                max_output_tokens=max_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = await self._client.aio.models.generate_content(
                model=MODEL_FLASH,
                contents=prompt,
                config=config,
            )

            text = response.text or ""
            return {
                "success": True,
                "text": text,
                "tokens": len(text.split()),
            }

        except Exception as exc:
            self._log.exception("creative.generate_text_failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Describe Image
    # ------------------------------------------------------------------

    async def describe_image(self, image_bytes: bytes) -> dict:
        """Analyze and describe an image using Gemini Flash.

        Args:
            image_bytes: PNG or JPEG bytes.

        Returns:
            {"success": bool, "description": str}
        """
        if not self.available:
            return {"success": False, "error": "Creative agent not available"}

        try:
            response = await self._client.aio.models.generate_content(
                model=MODEL_FLASH,
                contents=[types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(
                        mime_type="image/png", data=image_bytes)),
                    types.Part(text="Describe this image in detail."),
                ])],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )

            return {
                "success": True,
                "description": response.text or "",
            }

        except Exception as exc:
            self._log.exception("creative.describe_image_failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Step execution (for orchestrator integration)
    # ------------------------------------------------------------------

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
    ) -> dict:
        """Generate a short video using Veo 2.

        Args:
            prompt: Description of the video to generate.
            duration_seconds: Target duration (5-10 seconds).
            aspect_ratio: "16:9" or "9:16".

        Returns:
            {"success": bool, "video_path": str, ...}
        """
        if not self.available:
            return {"success": False, "error": "Creative agent not available"}

        self._log.info(
            "creative.generate_video",
            prompt=prompt[:100],
            duration=duration_seconds,
        )

        try:
            # Veo 2 video generation via google-genai
            response = await self._client.aio.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    duration_seconds=duration_seconds,
                    number_of_videos=1,
                ),
            )

            # Poll for completion (video gen is async)
            import time as _time
            max_wait = 120  # 2 minutes max
            poll_interval = 5
            elapsed = 0
            while not response.done and elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                response = await response.refresh()

            if not response.done:
                return {"success": False, "error": "Video generation timed out (2 min)"}

            if not response.generated_videos:
                return {"success": False, "error": "No video generated"}

            video = response.generated_videos[0]
            video_bytes = video.video.video_bytes

            filename = f"rio_video_{int(time.time())}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(video_bytes)

            self._log.info("creative.video_saved", path=filepath)
            return {
                "success": True,
                "video_path": filepath,
                "size_kb": round(len(video_bytes) / 1024, 1),
                "prompt": prompt,
            }

        except Exception as exc:
            self._log.exception("creative.generate_video_failed")
            return {"success": False, "error": str(exc)}

    async def execute_step(
        self,
        goal: str,
        expected_outcome: str = "",
        scratchpad: Optional[dict] = None,
    ) -> dict:
        """Execute a creative step as part of an orchestrated task.

        Analyzes the goal and decides whether to generate an image,
        text, or both.

        Returns:
            {"success": bool, "result": str, ...}
        """
        if not self.available:
            return {"success": False, "error": "Creative agent not available"}

        goal_lower = goal.lower()

        # Determine what kind of creative task this is
        is_image = any(kw in goal_lower for kw in (
            "image", "picture", "logo", "icon", "draw", "paint",
            "illustration", "photo", "design", "graphic", "visual",
            "diagram", "sketch", "generate an image",
        ))
        is_video = any(kw in goal_lower for kw in (
            "video", "clip", "animation", "motion", "animate",
            "generate a video", "short film",
        ))
        is_text = any(kw in goal_lower for kw in (
            "write", "compose", "poem", "story", "haiku", "essay",
            "explain", "summarize", "describe", "create a", "draft",
        ))

        results = []

        if is_video:
            vid_result = await self.generate_video(goal)
            if vid_result.get("success"):
                results.append(f"Video generated: {vid_result.get('video_path', 'in memory')}")
            else:
                results.append(f"Video generation failed: {vid_result.get('error', '')}")

        if is_image:
            img_result = await self.generate_image(goal)
            if img_result.get("success"):
                results.append(f"Image generated: {img_result.get('image_path', 'in memory')}")
            else:
                results.append(f"Image generation failed: {img_result.get('error', '')}")

        if is_text or (not is_image and not is_video):
            # Default to text generation
            system = get_creative_agent_prompt()
            txt_result = await self.generate_text(goal, system_instruction=system)
            if txt_result.get("success"):
                results.append(txt_result["text"])
            else:
                results.append(f"Text generation failed: {txt_result.get('error', '')}")

        combined = "\n\n".join(results)
        success = bool(results) and not all("failed" in r.lower() for r in results)

        return {
            "success": success,
            "result": combined[:2000],
            "type": "image+text" if (is_image and is_text) else ("image" if is_image else "text"),
        }
