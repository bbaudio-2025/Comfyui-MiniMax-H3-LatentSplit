# ComfyUI MiniMax H3 Latent Split

Split MiniMax H3 video+audio latents (`NestedTensor`) by time into overlapping segments, re-sample / re-denoise each segment individually, then stitch the results back into one seamless latent.

The goal: generate **long, high-resolution videos on low-VRAM GPUs**. Instead of sampling the whole timeline in one pass, the video is split into short overlapping segments that are re-sampled one at a time, so peak VRAM is bounded by a single segment rather than the whole video. Overlapping cross-fades plus a real-content keyframe anchor keep the segments temporally consistent.

## Key advantages

- **Low-VRAM long/high-res generation**: only one segment is re-sampled at a time (≈5.7 s by default), so peak VRAM depends only on a single segment's length and resolution, not on the total duration. Combined with re-encoding the conditioning at an upscaled size, you can re-sample long videos at higher resolution chunk by chunk.
- **Inter-chunk consistency**, via two stacked mechanisms:
  - **Overlap cross-fade** (Concat / Append): the overlapping frames of neighboring chunks are linearly blended to avoid seam flicker;
  - **Real-content keyframe anchor** (Anchor): the frame-0 keyframe of each chunk is replaced by the *re-sampled boundary frame* of the previous chunk. H3 keyframes are frozen rows in the packed sequence (never denoised), so the seam frame is hard-pinned to the previous chunk's actual output, removing the detail divergence that low-denoise chunking produces at seams — this is the key difference from plain "slice + paste" plugins.
- **Semantically correct timeline**: every chunk boundary is snapped to the model's keyframe token grid (17-frame steps), so each chunk is a valid, phase-aligned standalone video that can be decoded or re-sampled on its own — no phase-shift artifacts.
- **Audio stays in sync**: the audio latent (40 Hz) is cut in lockstep with the video frames at the 5/3 ratio; wherever a chunk starts in the video, the audio starts there too.
- **Automatic conditioning re-anchoring**: on split, the original `minimax_keyframes` are trimmed and re-anchored to each segment (out-of-segment keyframes are dropped), so every chunk gets a correct local condition.
- **Zero padding handled transparently**: segments are batch-padded to a uniform shape with a `noise_mask` marking the padded tails, so re-sampling keeps them untouched; concatenation trims them automatically.

## How it works

A MiniMax H3 latent is `{"samples": NestedTensor((video, audio))}`:

- video `[B, 24, T_v, H/16, W/16]`
- audio `[B, 32, 2, T_a]`, with `T_a = round(video_frames × 5/3)`

Video tokens cover frames on the periodic grid `(1, 4, 4, 4, 4)`: every 5 tokens cover 17 frames, and **the first token of every block is a 1-frame keyframe token**. Chunk boundaries must therefore land on token indices that are multiples of 5 (17-frame grid), otherwise the chunk is treated as a phase-shifted video and comes out broken. This plugin guarantees:

- every chunk starts on the keyframe grid (`overlap` and `chunk_length` must be multiples of 17; the UI enforces step=17);
- the last chunk always ends exactly on the total frame count — no leftover "sliver" tail chunk;
- the realized overlap is always a whole multiple of 17 frames (0, 17, 34, …).

## Nodes

| Node | Purpose |
| --- | --- |
| **Split MiniMax H3 Latent** | Splits a denoised AV latent into overlapping segments (batched output + per-segment conditioning + `segments_info` metadata). Optionally re-anchors keyframes; supports the `resize_conditioning` fallback. |
| **Extract MiniMax H3 Latent** | Pulls one segment from the batch, trims it to its true length, and outputs its conditioning, start frame and true frame count. |
| **MiniMax H3 Conditioning From Batch** | Extracts one segment's re-anchored conditioning on its own. |
| **Concat MiniMax H3 Latents** | Stitches segments back into a full AV latent with overlap cross-fading and automatic padding trim. `segments_info` is optional (if left unconnected, segments are assumed to tile end-to-end in order — correct for overlap-0 splits). |
| **Append MiniMax H3 Latents** | For auto-loop flows: appends the current segment to an accumulated latent each iteration, equivalent to one Concat step per round. |
| **Anchor MiniMax H3 Latent** | Replaces the current chunk's frame-0 keyframe with the previous iteration's re-sampled boundary frame, hard-anchoring the seam (see "consistency" above). |

## Typical workflows

### Auto-loop sequential re-sampling (recommended: low VRAM + consistency anchor)

```
source image → MiniMax H3 Image to Video (first stage, full length) → Split (chunk_length=136, overlap=17)
                                                                        │
     each iteration (index = 0, 1, 2, …):                              ▼
        Extract(index) ─┬─ latent ──────────────────────────► KSampler ─┐
                         └─ conditioning → Anchor ────────────────────────┘
                              ↑ previous_result = previous iteration's Append output
        Append(segment_source=accumulated, segment_add=re-sampled chunk, index, segments_info)
```

- Iteration 0: leave `previous_result` and `segment_source` unconnected (handled automatically);
- the final Append output is the complete stitched latent;
- generate the first stage at a small size/short length, then re-sample chunk by chunk at the upscaled resolution with bounded VRAM.

### Independent parallel re-sampling (chunks independent, can run in parallel)

```
Split → for each index: Extract → KSampler → collect
  → Concat(segments=all re-sampled chunks, segments_info=the same Split's output)
```

## Parameters

- **chunk_length** (default `136`, ≈5.7 s @ 24 fps; must be a multiple of 17)
- **overlap** (default `17`, one grid step; must be a multiple of 17, 17 is recommended. Larger values blend smoother but re-sample the overlap region twice)
- **anchor_strength** (default `0.999`): how much of the re-sampled boundary content the frozen anchor row keeps. 1.0 = exact content (hardest anchor); 0.999 = model default; lower values mix in more noise and weaken the anchor; 0.0 = pure noise (no anchoring).

## Notes & pitfalls

- **Do not rely on `resize_conditioning` for upscaling** — it is off by default. Bilinearly resizing keyframe latents in latent space breaks the model's 2×2 patch structure and paints a grid pattern into the output. The correct approach is to re-encode your conditioning **at the upscaled size** (run a new "MiniMax H3 Image to Video" with the upscaled width/height). `resize_conditioning` is only a last-resort fallback when re-encoding is not possible.
- `chunk_length` / `overlap` are forced to multiples of 17, so the value you set is the value actually used (no "displayed value ≠ real value" confusion).
- If subtle differences remain after stitching, raise `overlap`, or stack Plan B (global noise-field slicing) on top of Anchor.

## Installation

Place this folder in ComfyUI's `custom_nodes/` directory and restart ComfyUI. Nodes appear under `model/latent/minimax` and `model/conditioning/minimax`.

Dependencies: ComfyUI (with MiniMax H3 support) + PyTorch.  

## Extra

This project was vibe-coded by Deepseek V4 flash🐋, If you run into any problems, it's best to search with AI.😂
