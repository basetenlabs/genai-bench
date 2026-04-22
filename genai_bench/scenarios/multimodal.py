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

    Format: ID(width, height, input_tokens, output_tokens)
    Example: ID(1024,1024,1500,200)
    """

    scenario_type = MultiModality.DETERMINISTIC_IMAGE
    validation_pattern = r"^ID\(\d+,\d+,\d+,\d+\)$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        num_input_tokens: int,
        num_output_tokens: int,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens

    def sample(self) -> Tuple[Tuple[int, int], int, int, int]:
        return (
            (self.num_input_dimension_width, self.num_input_dimension_height),
            1,  # num_images — TODO: extend format to ID(w,h,N,input,output) for multi-image support
            self.num_input_tokens,
            self.num_output_tokens,
        )

    def to_string(self) -> str:
        return (
            f"ID({self.num_input_dimension_width},"
            f"{self.num_input_dimension_height},"
            f"{self.num_input_tokens},"
            f"{self.num_output_tokens})"
        )

    @classmethod
    def parse(cls, params_str: str) -> "DeterministicImageScenario":
        w, h, input_tokens, output_tokens = parse_params_str(params_str)[0]
        return cls(
            num_input_dimension_width=w,
            num_input_dimension_height=h,
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
        )


class PrefixImageScenario(Scenario):
    """
    Prefix-cached image + text scenario for VLM KV cache benchmarking.

    All requests share a cached text prefix with unique suffixes.
    Images are sampled from the dataset without replacement.

    Format: IP(width, height, prefix_tokens, suffix_tokens)/output_tokens
    Example: IP(1024,1024,1200,300)/200
    """

    scenario_type = MultiModality.PREFIX_IMAGE
    validation_pattern = r"^IP\(\d+,\d+,\d+,\d+\)/\d+$"

    def __init__(
        self,
        num_input_dimension_width: int,
        num_input_dimension_height: int,
        prefix_tokens: int,
        suffix_tokens: int,
        output_tokens: int,
    ):
        self.num_input_dimension_width = num_input_dimension_width
        self.num_input_dimension_height = num_input_dimension_height
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.output_tokens = output_tokens

    def sample(self) -> Tuple[Tuple[int, int], int, int, int, int]:
        return (
            (self.num_input_dimension_width, self.num_input_dimension_height),
            1,  # num_images — TODO: extend format to IP(w,h,N,prefix,suffix)/output for multi-image support
            self.prefix_tokens,
            self.suffix_tokens,
            self.output_tokens,
        )

    def to_string(self) -> str:
        return (
            f"IP({self.num_input_dimension_width},"
            f"{self.num_input_dimension_height},"
            f"{self.prefix_tokens},"
            f"{self.suffix_tokens})/"
            f"{self.output_tokens}"
        )

    @classmethod
    def parse(cls, params_str: str) -> "PrefixImageScenario":
        match = re.match(r"\((\d+),(\d+),(\d+),(\d+)\)/(\d+)", params_str)
        if not match:
            raise ValueError(
                f"Invalid prefix image format: {params_str}. "
                f"Expected format: (w,h,prefix_tokens,suffix_tokens)/output_tokens"
            )
        w = int(match.group(1))
        h = int(match.group(2))
        prefix_tokens = int(match.group(3))
        suffix_tokens = int(match.group(4))
        output_tokens = int(match.group(5))
        return cls(
            num_input_dimension_width=w,
            num_input_dimension_height=h,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            output_tokens=output_tokens,
        )
