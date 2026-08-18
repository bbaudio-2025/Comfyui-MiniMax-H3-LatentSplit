"""ComfyUI custom node pack: split MiniMax H3 video+audio latents by time.

Splits an already-denoised MiniMax H3 AV latent into overlapping time segments
so each segment can be re-sampled / re-denoised separately, and stitches the
re-sampled segments back together.
"""

from comfy_api.latest import ComfyExtension

from .nodes import (
    MiniMaxH3LatentSplit,
    MiniMaxH3LatentFromBatch,
    MiniMaxH3CondFromBatch,
    MiniMaxH3ConcatLatents,
    MiniMaxH3LatentAppend,
)

__all__ = ["MiniMaxH3LatentSplit", "MiniMaxH3LatentFromBatch", "MiniMaxH3CondFromBatch", "MiniMaxH3ConcatLatents", "MiniMaxH3LatentAppend"]


class MiniMaxH3LatentSplitExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3LatentSplit,
            MiniMaxH3LatentFromBatch,
            MiniMaxH3CondFromBatch,
            MiniMaxH3ConcatLatents,
            MiniMaxH3LatentAppend,
        ]


async def comfy_entrypoint() -> MiniMaxH3LatentSplitExtension:
    return MiniMaxH3LatentSplitExtension()