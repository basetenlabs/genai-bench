import re
from typing import Optional, Tuple

from genai_bench.scenarios.base import MultiModality, Scenario, parse_params_str


class ImageModality(Scenario):
    """
    Image input and text output
    e.g.
    I(256,256) or
    I(2048,2048,2)
    The third number represents the
    number of images
    """

    scenario_type = MultiModality.IMAGE
    validation_pattern = r"^I\(\d+,\d+(?:,\d+)?\)$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        num_input_images: int = 1,
        max_output_token: Optional[int] = None,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.num_input_images = num_input_images
        self.max_output_token = max_output_token

    def sample(self) -> Tuple[Tuple[int, int], int, int | None]:
        return (
            (
                self.num_input_dimension_width,
                self.num_input_dimension_height,
            ),
            self.num_input_images,
            self.max_output_token,
        )

    def to_string(self) -> str:  # TODO: include max_output_token in the string
        if self.num_input_images == 1:
            return (
                f"I({self.num_input_dimension_width},{self.num_input_dimension_height})"
            )
        else:
            return (
                f"I({self.num_input_dimension_width},"
                f"{self.num_input_dimension_height},"
                f"{self.num_input_images})"
            )

    @classmethod
    def parse(cls, params_str: str) -> "ImageModality":
        num_input_dimension_width, num_input_dimension_height, *optional = (
            parse_params_str(params_str)[0]
        )
        if not optional:
            return cls(
                num_input_dimension_width=num_input_dimension_width,
                num_input_dimension_height=num_input_dimension_height,
            )
        else:
            return cls(
                num_input_dimension_width=num_input_dimension_width,
                num_input_dimension_height=num_input_dimension_height,
                num_input_images=optional[0],
            )


class DeterministicImageScenario(Scenario):
    """
    Deterministic image + text scenario for VLM benchmarking.

    All text tokens are unique per request (no prefix caching).
    Images are sampled from the dataset without replacement.

    Format: ID(width, height, input_tokens, output_tokens[, num_images])
      - 4-param form: ID(w, h, input_tokens, output_tokens) — defaults to 1 image
      - 5-param form: ID(w, h, input_tokens, output_tokens, N) — N images per request
    Example: ID(1024,1024,1500,200)     — single image (default)
             ID(1024,1024,1500,200,3)   — three images per request
    """

    scenario_type = MultiModality.DETERMINISTIC_IMAGE
    # 4 or 5 positional ints inside ID(...)
    validation_pattern = r"^ID\(\d+,\d+,\d+,\d+(?:,\d+)?\)$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        num_input_tokens: int,
        num_output_tokens: int,
        num_images: int = 1,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.num_images = num_images

    def sample(self) -> Tuple[Tuple[int, int], int, int, int]:
        return (
            (self.num_input_dimension_width, self.num_input_dimension_height),
            self.num_images,
            self.num_input_tokens,
            self.num_output_tokens,
        )

    def to_string(self) -> str:
        base = (
            f"ID({self.num_input_dimension_width},"
            f"{self.num_input_dimension_height},"
            f"{self.num_input_tokens},"
            f"{self.num_output_tokens}"
        )
        # Only emit num_images when > 1, so the canonical form for the common
        # single-image case matches the legacy 4-param shape exactly.
        if self.num_images != 1:
            return f"{base},{self.num_images})"
        return f"{base})"

    @classmethod
    def parse(cls, params_str: str) -> "DeterministicImageScenario":
        params = parse_params_str(params_str)[0]
        if len(params) == 4:
            w, h, input_tokens, output_tokens = params
            num_images = 1
        elif len(params) == 5:
            w, h, input_tokens, output_tokens, num_images = params
        else:
            raise ValueError(
                f"Invalid ID() format: {params_str}. "
                f"Expected 4 or 5 params: (w,h,input_tokens,output_tokens) "
                f"or (w,h,input_tokens,output_tokens,num_images)"
            )
        return cls(
            num_input_dimension_width=w,
            num_input_dimension_height=h,
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            num_images=num_images,
        )


class PrefixImageScenario(Scenario):
    """
    Prefix-cached image + text scenario for VLM KV cache benchmarking.

    All requests share a cached text prefix with unique suffixes.
    Images are sampled from the dataset without replacement.

    Format: IP(width, height, prefix_tokens, suffix_tokens[, num_images])/output_tokens
      - 4-param form: IP(w, h, prefix_tokens, suffix_tokens)/output — defaults to 1 image
      - 5-param form: IP(w, h, prefix_tokens, suffix_tokens, N)/output — N images per request
    Example: IP(1024,1024,1200,300)/200     — single image (default)
             IP(1024,1024,1200,300,3)/200   — three images per request
    """

    scenario_type = MultiModality.PREFIX_IMAGE
    # 4 or 5 positional ints inside IP(...)/output
    validation_pattern = r"^IP\(\d+,\d+,\d+,\d+(?:,\d+)?\)/\d+$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        prefix_tokens: int,
        suffix_tokens: int,
        output_tokens: int,
        num_images: int = 1,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.output_tokens = output_tokens
        self.num_images = num_images

    def sample(self) -> Tuple[Tuple[int, int], int, int, int, int]:
        return (
            (self.num_input_dimension_width, self.num_input_dimension_height),
            self.num_images,
            self.prefix_tokens,
            self.suffix_tokens,
            self.output_tokens,
        )

    def to_string(self) -> str:
        base = (
            f"IP({self.num_input_dimension_width},"
            f"{self.num_input_dimension_height},"
            f"{self.prefix_tokens},"
            f"{self.suffix_tokens}"
        )
        if self.num_images != 1:
            base = f"{base},{self.num_images}"
        return f"{base})/{self.output_tokens}"

    @classmethod
    def parse(cls, params_str: str) -> "PrefixImageScenario":
        # Try 5-param form first, then fall back to 4-param form.
        match_5 = re.match(r"\((\d+),(\d+),(\d+),(\d+),(\d+)\)/(\d+)", params_str)
        match_4 = re.match(r"\((\d+),(\d+),(\d+),(\d+)\)/(\d+)", params_str)
        if match_5:
            w, h, prefix_tokens, suffix_tokens, num_images, output_tokens = (
                int(x) for x in match_5.groups()
            )
        elif match_4:
            w, h, prefix_tokens, suffix_tokens, output_tokens = (
                int(x) for x in match_4.groups()
            )
            num_images = 1
        else:
            raise ValueError(
                f"Invalid prefix image format: {params_str}. "
                f"Expected (w,h,prefix_tokens,suffix_tokens)/output_tokens "
                f"or (w,h,prefix_tokens,suffix_tokens,num_images)/output_tokens"
            )
        return cls(
            num_input_dimension_width=w,
            num_input_dimension_height=h,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            output_tokens=output_tokens,
            num_images=num_images,
        )
