"""MiniMax H3 latent splitting / recombination nodes.

These nodes split an already-denoised MiniMax H3 AV latent (a NestedTensor of
video [B, 24, T, H/16, W/16] + audio [B, 32, 2, T40]) into overlapping time
segments so each can be re-sampled separately, then re-combine the results.

Frame/token mapping (mirrors comfy.ldm.minimax.model):
  * video latent token k covers FRAME_PER_TOKEN[k % 5] = (1, 4, 4, 4, 4) pixel
    frames (periodic grid, 17 frames per 5 tokens)
  * audio latent frames run at FRAME_RESCALE = 5/3 per pixel frame (40 vs 24 Hz)

Every split boundary is snapped to a video-token boundary so the segments cut
cleanly; the audio stream is cut at the matching round(f * FRAME_RESCALE).
"""

import math

import torch
import torch.nn.functional as F

import comfy.nested_tensor
from comfy_api.latest import io

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

H3_COND_BATCH = io.Custom("H3_COND_BATCH")
H3_SPLIT_BATCH = io.Custom("H3_SPLIT_BATCH")
H3_SPATIAL_INFO = io.Custom("H3_SPATIAL_INFO")


# ---------------------------------------------------------------------------
# frame <-> token helpers
# ---------------------------------------------------------------------------

def frames_for_tokens(n):
    """Pixel frames covered by the first `n` video latent tokens."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f):
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def tokens_within_frames(f):
    """Largest token count fully inside `f` pixel frames."""
    n, acc = 0, 0
    while True:
        nxt = acc + FRAME_PER_TOKEN[n % 5]
        if nxt > f:
            return n
        acc = nxt
        n += 1


def snap_frame_boundary(f, max_tokens, phase=None):
    """Nearest video-token boundary to pixel frame f.

    When `phase` is given, only boundaries at token indices that are a multiple
    of `phase` are considered. The H3 model treats the first token of every video
    as a 1-frame keyframe token (FRAME_PER_TOKEN[k%5], token 0 covers 1 frame),
    so segment boundaries MUST be snapped to the keyframe grid (phase=5): a chunk
    that starts at any other token is re-sampled/decoded on a phase-shifted
    timeline and comes out broken. Returns (token_index, exact_pixel_frames)."""
    step = phase if phase is not None else 1
    best_k, best_f, best_d = 0, 0, f
    for k in range(0, max_tokens + 1, step):
        acc = frames_for_tokens(k)
        d = abs(acc - f)
        if d < best_d:
            best_k, best_f, best_d = k, acc, d
    return best_k, best_f


def audio_range(f0, f1):
    """Audio latent token range [a0, a1) for the pixel-frame span [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def compute_segments(tv, chunk_length, overlap):
    """Compute per-chunk (video_token_start, frame_start, video_token_end, frame_end).

    Chunk i spans pixel frames [i*hop, min(i*hop + chunk_length, frame_count))
    with hop = chunk_length - overlap. Every boundary is snapped to a keyframe
    token (index % 5 == 0, one FRAME_PER_TOKEN grid step = 17 frames), so each
    chunk is a valid standalone video for re-sampling/decoding. The realized
    overlap is therefore a whole number of grid steps (0, 17, 34, ... frames).
    The last chunk always ends on the exact total frame count, and any chunk
    whose nominal end reaches the total becomes the last one (clipped at the
    exact total) instead of being snapped down and leaving a tiny remainder
    chunk with zero overlap at the seam.
    """
    frame_count = frames_for_tokens(tv)
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_length <= overlap:
        raise ValueError("overlap must be smaller than chunk_length")

    hop = chunk_length - overlap
    bounds = []
    prev_end_k = 0
    i = 0
    while True:
        s = i * hop
        e = min(s + chunk_length, frame_count)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = snap_frame_boundary(s, tv, phase=5)
            # never leave a gap after the previous chunk
            if k0 > prev_end_k:
                k0, f0 = prev_end_k, frames_for_tokens(prev_end_k)
        if e >= frame_count:
            # nominal end already reaches the total: this is the last chunk
            k1, f1 = tv, frame_count
        else:
            k1, f1 = snap_frame_boundary(e, tv, phase=5)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_for_tokens(k1)
            if k1 >= tv:
                k1, f1 = tv, frame_count
        bounds.append((k0, f0, k1, f1))
        if k1 >= tv:
            break
        prev_end_k = k1
        i += 1
    return bounds, frame_count
    return bounds, frame_count


def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)


# ---------------------------------------------------------------------------
# spatial tiling helpers (experimental)
# ---------------------------------------------------------------------------

def compute_spatial_grid(h, w, th, tw, ol_h, ol_w):
    """Tile a latent of size (h, w) with tiles (th, tw) and overlap (ol_h, ol_w).

    Returns (row_offsets, col_offsets, true_row_dims, true_col_dims). The stride
    is th - ol_h / tw - ol_w; the last row/col is clamped to the source size and
    every tile's own true extent is clamped too. All values are in latent units."""
    if th <= 0 or tw <= 0:
        raise ValueError("tile dimensions must be positive")
    if ol_h >= th or ol_w >= tw:
        raise ValueError("overlap must be smaller than the tile size")
    sh = th - ol_h
    sw = tw - ol_w
    nrows = 1 if h <= th else math.ceil((h - ol_h) / sh)
    if (nrows - 1) * sh + th < h:
        nrows += 1
    ncols = 1 if w <= tw else math.ceil((w - ol_w) / sw)
    if (ncols - 1) * sw + tw < w:
        ncols += 1
    rows = [i * sh for i in range(nrows)]
    cols = [j * sw for j in range(ncols)]
    trows = [min(th, h - r) for r in rows]
    tcols = [min(tw, w - c) for c in cols]
    return rows, cols, trows, tcols


def spatial_fade_mask(th, tw, tr, tc, oh, ow, done_top, done_left, overlap_mode="earlier", fade_width=None):
    """Per-tile video noise mask [th, tw]: 1 = re-sample freely, 0 = frozen.

    Padding rows/cols (beyond the tile's true extent) are frozen. Overlap strips
    shared with an already-processed neighbor (done_top / done_left) are frozen
    by fade_width ALONE: 0 (default) = whole strip frozen (keeps the neighbor's
    content), None = whole-strip fade 0->1 toward the tile interior, otherwise
    only the first fade_width latent units fade 0->1 and the rest is free.

    overlap_mode is accepted for signature compatibility but does NOT affect the
    mask; it only selects which side wins the band when stitching (see
    blend_weights / the append node)."""
    mask = torch.ones(th, tw, dtype=torch.float32)
    if tr < th:
        mask[tr:, :] = 0.0
    if tc < tw:
        mask[:, tc:] = 0.0
    if done_left and ow > 0:
        if fade_width == 0:
            mask[:, :ow] = 0.0
        else:
            fw = min(fade_width if fade_width is not None else ow, ow)
            w = torch.linspace(0.0, 1.0, fw)
            mask[:, :fw] = torch.minimum(mask[:, :fw], w[None, :])
    if done_top and oh > 0:
        if fade_width == 0:
            mask[:oh, :] = 0.0
        else:
            fw = min(fade_width if fade_width is not None else oh, oh)
            w = torch.linspace(0.0, 1.0, fw)
            mask[:fw, :] = torch.minimum(mask[:fw, :], w[:, None])
    return mask


def blend_weights(t, overlap_blend, overlap_mode):
    """Weight given to the NEW tile's content across an overlap band.

    t runs 0..1 from the done-seam toward the tile interior. overlap_mode 'later'
    hands the band to the new tile; 'earlier' to the accumulated content.
    overlap_blend selects the transition shape: linear cross-fade, smoothstep
    (eased), overwrite (whole band from the overlap_mode side) or midpoint (hard
    switch in the middle)."""
    if overlap_blend == "overwrite":
        return torch.ones_like(t) if overlap_mode == "later" else torch.zeros_like(t)
    if overlap_blend == "midpoint":
        step = (t >= 0.5).to(t.dtype)
    elif overlap_blend == "smoothstep":
        step = t * t * (3.0 - 2.0 * t)
    else:
        step = t
    if overlap_mode == "later":
        return step
    return 1.0 - step


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    """Spatially crop every keyframe's video latent to a tile of the source frame.

    Keyframes whose latent already matches the source spatial size are cropped to
    the tile's latent region; others are passed through unchanged. Audio
    keyframes are untouched (audio is not spatial)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if (lt is not None and lt.shape[3] == src_h and lt.shape[4] == src_w):
                    nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


# ---------------------------------------------------------------------------
# conditioning re-anchoring
# ---------------------------------------------------------------------------

def trim_keyframe(kf, f0, f1):
    """Copy a keyframe cut to the portion fully inside pixel frames [f0, f1).

    The video latent is trimmed to whole tokens strictly inside the segment and
    resolved_frame_index is re-anchored to the segment origin. Returns None when
    nothing of the keyframe survives.
    """
    idx = kf["resolved_frame_index"]
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    has_v = latent is not None
    has_a = audio_latent is not None

    if not has_v and not has_a:
        if idx < f0 or idx >= f1:
            return None
        return {"resolved_frame_index": idx - f0}

    out = {}
    if has_v:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if has_a:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    if "latent" not in out and "audio_latent" not in out:
        return None
    return out


def reanchor_conditioning(cond, f0, f1, spatial=None):
    """Return a conditioning list whose minimax_keyframes are cut/re-anchored to
    the pixel-frame segment [f0, f1). minimax_refs are left untouched (they are
    positioned before the target timeline and need no re-anchoring).

    When `spatial` (latent_h, latent_w) is given, keyframe video latents whose
    spatial size differs are resized to it: the model patches keyframe rows onto
    the target's spatial grid, so a spatially-upscaled latent (done by another
    plugin) requires the keyframes to be resized too or sampling errors out."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [trim_keyframe(kf, f0, f1) for kf in kfs]
            trimmed = [kf for kf in trimmed if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear", align_corners=False
                            ).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


# ---------------------------------------------------------------------------
# stitching helpers
# ---------------------------------------------------------------------------

def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    w = w.view(shape)
    return a + (b - a) * w


def _true_len(mask, dim):
    """Length of the unpadded head of a mask whose tail is zero-padded."""
    other = [d for d in range(mask.ndim) if d != dim]
    m = mask.any(dim=other) if len(other) > 1 else mask.any(dim=other[0])
    idx = (m != 0).nonzero()
    if idx.numel() == 0:
        return 0
    return int(idx.max().item()) + 1


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

class MiniMaxH3LatentSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentSplit",
            display_name="Split MiniMax H3 Latent",
            category="model/latent/minimax",
            description=(
                "Split an already-denoised MiniMax H3 AV latent (video+audio) into "
                "overlapping time segments so each can be re-sampled separately. "
                "Boundaries are snapped to the model's keyframe-token grid (every "
                "segment starts at a 1-frame keyframe token), so each piece is a "
                "valid standalone video and can be decoded on its own. Outputs a "
                "batch of segments padded to a uniform shape (the padded tail is "
                "masked so re-sampling keeps it). Optionally re-anchors "
                "minimax_keyframes to each segment. When the latent was spatially "
                "upscaled by another plugin, re-create the conditioning at the "
                "upscaled size (resize_conditioning is OFF by default): resizing "
                "keyframe latents in latent space paints a grid pattern into the "
                "output."
            ),
            search_aliases=["h3 split", "latent split", "split latent", "minimax split"],
            inputs=[
                io.Latent.Input("latent", tooltip="Denoised MiniMax H3 AV latent to split."),
                io.Conditioning.Input("conditioning", optional=True,
                                      tooltip="Conditioning used to generate this latent. When connected, its minimax_keyframes are re-anchored (and out-of-segment ones dropped) for every segment."),
                io.Int.Input("chunk_length", default=136, min=17, max=100000, step=17,
                             tooltip="Target pixel frames per chunk (at 24 fps). MUST be a multiple of 17 (one keyframe grid step) so the realized length exactly equals this value. 136 = ~5.7s, 153 = ~6.4s."),
                io.Int.Input("overlap", default=17, min=0, max=100000, step=17,
                             tooltip="Pixel frames of overlap between consecutive chunks. MUST be a multiple of 17; recommended 17 (one keyframe grid step) for clean cross-chunk continuity. Larger overlap blends smoother but re-samples the overlap region twice."),
                io.Boolean.Input("resize_conditioning", default=False,
                                 tooltip="Resize keyframe video latents to the target latent's spatial size when they differ. OFF by default: re-encode your conditioning at the target (upscaled) size instead - e.g. run a new 'MiniMax H3 Image to Video' with the upscaled width/height - because bilinearly resizing keyframe latents in latent space breaks the model's 2x2 patch structure and paints a grid pattern into the output. Only enable this as a fallback to avoid a hard error when you cannot re-encode the conditioning."),
            ],
            outputs=[
                io.Latent.Output("latents",
                                 tooltip="Batch of segments (N, 24, T, H, W). All are padded with zeros to the longest segment; a noise_mask marks the padded tails so re-sampling keeps them. H3 samples one latent at a time - pull each segment with 'Extract MiniMax H3 Latent' before re-sampling."),
                H3_COND_BATCH.Output("cond_batch",
                                     tooltip="Per-segment conditioning (re-anchored keyframes). None unless 'conditioning' was connected. Extract with 'MiniMax H3 Conditioning From Batch'."),
                H3_SPLIT_BATCH.Output("segments_info",
                                      tooltip="Segment metadata (frame starts/counts, video & audio token ranges) consumed by the extract/concat companion nodes."),
                io.Int.Output("frame_starts", is_output_list=True,
                              tooltip="Start pixel frame of each segment in the original timeline."),
                io.Int.Output("frame_counts", is_output_list=True,
                              tooltip="True pixel-frame length of each segment (before padding)."),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Both length inputs must be multiples of 17 so the set value is exactly
        the realized value (the model's keyframe grid step = 17 frames)."""
        for name in ("chunk_length", "overlap"):
            v = kwargs.get(name)
            if isinstance(v, int) and v % 17 != 0:
                return (f"'{name}' must be a multiple of 17 (the model's keyframe "
                        f"grid step); got {v}. Use values like 17, 34, 51, ... for "
                        f"overlap and 119, 136, 153, ... for chunk_length.")
        return True

    @classmethod
    def execute(cls, latent, chunk_length, overlap, conditioning=None, resize_conditioning=False) -> io.NodeOutput:
        if chunk_length % 17 != 0:
            raise ValueError(f"chunk_length must be a multiple of 17 (the model's keyframe grid step); got {chunk_length}")
        if overlap % 17 != 0:
            raise ValueError(f"overlap must be a multiple of 17 (the model's keyframe grid step); got {overlap}")
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MiniMaxH3LatentSplit expects a MiniMax H3 AV latent (nested video [B,24,T,H,W] + audio [B,32,2,T])")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("MiniMaxH3LatentSplit expects a single-video latent (batch 1)")

        tv = video.shape[2]
        ta = audio.shape[-1]
        bounds, frame_count = compute_segments(tv, chunk_length, overlap)

        video_segs = []
        audio_segs = []
        for k0, f0, k1, f1 in bounds:
            video_segs.append(video[:, :, k0:k1].contiguous())
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, ta)
            audio_segs.append(audio[:, :, :, a0:a1].contiguous())

        # pad every segment to a uniform shape (the user requirement) and build
        # a noise_mask so the padded tail is kept (not denoised) when re-sampling.
        max_vt = max(v.shape[2] for v in video_segs)
        max_at = max(a.shape[-1] for a in audio_segs)
        batch_video = torch.zeros((len(video_segs), video.shape[1], max_vt, video.shape[3], video.shape[4]),
                                  device=video.device, dtype=video.dtype)
        batch_audio = torch.zeros((len(audio_segs), audio.shape[1], audio.shape[2], max_at),
                                  device=audio.device, dtype=audio.dtype)
        batch_vmask = torch.zeros_like(batch_video)
        batch_amask = torch.zeros_like(batch_audio)
        for i, v in enumerate(video_segs):
            batch_video[i, :, :v.shape[2]] = v
            batch_vmask[i, :, :v.shape[2]] = 1.0
        for i, a in enumerate(audio_segs):
            batch_audio[i, :, :, :a.shape[-1]] = a
            batch_amask[i, :, :, :a.shape[-1]] = 1.0

        out_latent = {
            "samples": comfy.nested_tensor.NestedTensor((batch_video, batch_audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((batch_vmask, batch_amask)),
        }

        starts = [f0 for _, f0, _, _ in bounds]
        counts = [f1 - f0 for _, f0, _, f1 in bounds]
        segments_meta = {
            "frame_count": frame_count,
            "starts": starts,
            "counts": counts,
            "video_tokens": [[k0, k1] for k0, _, k1, _ in bounds],
            "audio_tokens": [list(audio_range(f0, f1)) for _, f0, _, f1 in bounds],
        }

        conds = None
        if conditioning is not None:
            spatial = (video.shape[3], video.shape[4]) if resize_conditioning else None
            conds = [reanchor_conditioning(conditioning, f0, f1, spatial) for _, f0, _, f1 in bounds]

        return io.NodeOutput(out_latent, conds, segments_meta, starts, counts)


class MiniMaxH3LatentFromBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentFromBatch",
            display_name="Extract MiniMax H3 Latent",
            category="model/latent/minimax",
            description=(
                "Pull one segment out of a Split MiniMax H3 Latent batch and trim it "
                "to its true length (dropping the zero padding). The result is a clean "
                "batch-1 AV latent ready for re-sampling, plus the segment's matching "
                "conditioning, frame start and true length."
            ),
            search_aliases=["h3 extract", "segment from batch", "minimax extract"],
            inputs=[
                io.Latent.Input("latents", tooltip="Batch produced by 'Split MiniMax H3 Latent'."),
                H3_COND_BATCH.Input("cond_batch",
                                    tooltip="cond_batch output of the same split. The conditioning of the selected segment is returned on the 'conditioning' output."),
                H3_SPLIT_BATCH.Input("segments", tooltip="Metadata output of the same split."),
                io.Int.Input("index", default=0, min=0, max=100000,
                             tooltip="Segment index (0-based). Negative counts from the end."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="The selected segment, trimmed to its true length."),
                io.Conditioning.Output("conditioning", display_name="conditioning",
                                       tooltip="Conditioning of the selected segment (re-anchored keyframes)."),
                io.Int.Output("frame_start", tooltip="Start pixel frame of this segment in the original timeline."),
                io.Int.Output("frame_count", tooltip="True pixel-frame length of this segment."),
            ],
        )

    @classmethod
    def execute(cls, latents, cond_batch, segments, index) -> io.NodeOutput:
        n = len(segments["starts"])
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise ValueError("index {} out of range for {} segments".format(index, n))
        samples = latents["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MiniMaxH3LatentFromBatch expects a MiniMax H3 AV latent batch")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        k0, k1 = segments["video_tokens"][index]
        a0, a1 = segments["audio_tokens"][index]
        seg_video = video[index:index + 1, :, :k1 - k0].contiguous()
        seg_audio = audio[index:index + 1, :, :, :a1 - a0].contiguous()
        out = {
            "samples": comfy.nested_tensor.NestedTensor((seg_video, seg_audio)),
        }
        conditioning = None
        if cond_batch is not None:
            conditioning = cond_batch[index]
        return io.NodeOutput(out, conditioning, segments["starts"][index], segments["counts"][index])


class MiniMaxH3CondFromBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CondFromBatch",
            display_name="MiniMax H3 Conditioning From Batch",
            category="model/conditioning/minimax",
            description=(
                "Extract the conditioning of one segment from a 'Split MiniMax H3 "
                "Latent' cond_batch output. Keyframes have already been re-anchored "
                "to that segment."
            ),
            search_aliases=["h3 cond", "conditioning from batch", "minimax cond"],
            inputs=[
                H3_COND_BATCH.Input("cond_batch", tooltip="cond_batch output of 'Split MiniMax H3 Latent'."),
                io.Int.Input("index", default=0, min=0, max=100000,
                             tooltip="Segment index (0-based). Negative counts from the end."),
            ],
            outputs=[
                io.Conditioning.Output("conditioning", display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, cond_batch, index) -> io.NodeOutput:
        if cond_batch is None:
            raise ValueError("No conditioning was produced - connect 'conditioning' to the split node first")
        n = len(cond_batch)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise ValueError("index {} out of range for {} conditioning entries".format(index, n))
        return io.NodeOutput(cond_batch[index])


class MiniMaxH3ConcatLatents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=io.Latent.Input("segment", tooltip="One re-sampled segment, in timeline order."),
            prefix="segment_",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="MiniMaxH3ConcatLatents",
            display_name="Concat MiniMax H3 Latents",
            category="model/latent/minimax",
            description=(
                "Stitch independently re-sampled segments back into one MiniMax H3 AV "
                "latent. Overlapping tails are cross-faded and the padding is trimmed "
                "automatically (segments may keep or drop the split's noise_mask). "
                "The segments_info input is the split's output of the same name - it "
                "tells this node where each segment sits in the original timeline. It "
                "is optional: when left unconnected the segments are assumed to tile "
                "end-to-end in order (correct when the split used overlap 0)."
            ),
            search_aliases=["h3 concat", "concat latent", "stitch latent", "minimax concat"],
            inputs=[
                io.Autogrow.Input("segments", template=template),
                H3_SPLIT_BATCH.Input("segments_info", optional=True,
                                     tooltip="Segment metadata ('segments_info' output of the same 'Split MiniMax H3 Latent'). Leave unconnected to stitch segments end-to-end in order (overlap-0 splits)."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Full-length MiniMax H3 AV latent."),
            ],
        )

    @classmethod
    def execute(cls, segments, segments_info=None) -> io.NodeOutput:
        ordered = [segments[k] for k in sorted(segments, key=lambda k: int(k.rsplit("_", 1)[1]))
                   if segments[k] is not None]
        if not ordered:
            raise ValueError("MiniMaxH3ConcatLatents needs at least one segment")

        video_ref = ordered[0]["samples"].tensors[0]
        if video_ref.shape[0] != 1:
            raise ValueError("MiniMaxH3ConcatLatents expects batch-1 segments")
        c = video_ref.shape[1]
        h = video_ref.shape[3]
        w = video_ref.shape[4]

        lens_v, lens_a = [], []
        for lat in ordered:
            samples = lat["samples"]
            if not is_h3_av_latent(samples):
                raise ValueError("MiniMaxH3ConcatLatents expects MiniMax H3 AV latent segments")
            video = samples.tensors[0]
            audio = samples.tensors[1]
            if video.shape[3] != h or video.shape[4] != w:
                raise ValueError("all segments must share the same spatial size")
            lv = video.shape[2]
            la = audio.shape[-1]
            nm = lat.get("noise_mask")
            if nm is not None and nm.is_nested and len(nm.tensors) == 2:
                lv = _true_len(nm.tensors[0], dim=2)
                la = _true_len(nm.tensors[1], dim=3)
                lv = min(lv, video.shape[2])
                la = min(la, audio.shape[-1])
            lens_v.append(lv)
            lens_a.append(la)

        if segments_info is None:
            # no metadata: assume the segments tile end-to-end in timeline order
            g = [0]
            for lv in lens_v[:-1]:
                g.append(g[-1] + lv)
            starts = [frames_for_tokens(x) for x in g]
            segments_info = {
                "starts": starts,
                "video_tokens": [[g[i], g[i] + lens_v[i]] for i in range(len(g))],
                "audio_tokens": [list(audio_range(starts[i], frames_for_tokens(g[i] + lens_v[i])))
                                 for i in range(len(g))],
            }

        expected = len(segments_info["starts"])
        if len(ordered) != expected:
            raise ValueError("connected {} segments but the split produced {}".format(len(ordered), expected))

        # global token offset of each segment
        g = [tokens_for_frames(s) for s in segments_info["starts"]]
        ag = [round(s * FRAME_RESCALE) for s in segments_info["starts"]]

        total_v = max(g[i] + lens_v[i] for i in range(expected))
        total_a = max(ag[i] + lens_a[i] for i in range(expected))
        result_v = torch.zeros((1, c, total_v, h, w), device=video_ref.device, dtype=video_ref.dtype)
        result_a = torch.zeros((1, 32, 2, total_a), device=video_ref.device, dtype=video_ref.dtype)

        for i, lat in enumerate(ordered):
            video = lat["samples"].tensors[0]
            audio = lat["samples"].tensors[1]
            lv = lens_v[i]
            la = lens_a[i]
            v = video[:, :, :lv]
            a = audio[:, :, :, :la]
            gi = g[i]
            agi = ag[i]

            if i > 0:
                ov = (g[i - 1] + lens_v[i - 1]) - gi
                ova = (ag[i - 1] + lens_a[i - 1]) - agi
            else:
                ov = ova = 0

            if ov > 0:
                ov = min(ov, v.shape[2])
                tail = result_v[:, :, gi:gi + ov].clone()
                result_v[:, :, gi:gi + ov] = _crossfade(tail, v[:, :, :ov], dim=2)
                v = v[:, :, ov:]
            write_v = gi + max(ov, 0)
            if v.shape[2] > 0:
                result_v[:, :, write_v:write_v + v.shape[2]] = v

            if ova > 0:
                ova = min(ova, a.shape[-1])
                tail = result_a[:, :, :, agi:agi + ova].clone()
                result_a[:, :, :, agi:agi + ova] = _crossfade(tail, a[:, :, :, :ova], dim=3)
                a = a[:, :, :, ova:]
            write_a = agi + max(ova, 0)
            if a.shape[-1] > 0:
                result_a[:, :, :, write_a:write_a + a.shape[-1]] = a

        out = {"samples": comfy.nested_tensor.NestedTensor((result_v, result_a))}
        return io.NodeOutput(out)


class MiniMaxH3LatentAppend(io.ComfyNode):
    """Append one segment onto an accumulated latent, one per loop iteration."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentAppend",
            display_name="Append MiniMax H3 Latents",
            category="model/latent/minimax",
            description=(
                "Append one re-sampled segment onto an accumulated MiniMax H3 AV "
                "latent for auto-loop flows that stitch one segment per iteration. "
                "segment_source is the result of the previous additions (segments "
                "0..index-1). segment_add is the current segment, placed at the "
                "timeline position given by segments_info and index; its overlap with "
                "the previous segment is cross-faded exactly like 'Concat MiniMax H3 "
                "Latents', so after the final iteration the accumulated latent equals "
                "the concat of every segment."
            ),
            search_aliases=["h3 append", "append latent", "loop append", "minimax append"],
            inputs=[
                io.Latent.Input("segment_source",
                                tooltip="Accumulated latent from previous iterations (segments 0..index-1)."),
                io.Latent.Input("segment_add",
                                tooltip="Current segment (index) to append, e.g. the 'Extract MiniMax H3 Latent' output after re-sampling."),
                io.Int.Input("index", default=0, min=0, max=100000, step=1,
                             tooltip="Timeline index (0-based) of segment_add in the split."),
                H3_SPLIT_BATCH.Input("segments_info",
                                     tooltip="Segment metadata ('segments_info' output of the same 'Split MiniMax H3 Latent')."),
            ],
            outputs=[
                io.Latent.Output("latent",
                                 tooltip="Accumulated latent with segment_add appended. Feed back into segment_source on the next iteration."),
            ],
        )

    @classmethod
    def execute(cls, segment_source, segment_add, index, segments_info) -> io.NodeOutput:
        if segments_info is None:
            raise ValueError("MiniMaxH3LatentAppend needs the split's 'segments_info' input")
        starts = segments_info.get("starts")
        vtoks = segments_info.get("video_tokens")
        atoks = segments_info.get("audio_tokens")
        if not starts or not vtoks or not atoks:
            raise ValueError("MiniMaxH3LatentAppend: segments_info is missing starts/video_tokens/audio_tokens")
        if index < 0 or index >= len(starts):
            raise ValueError(f"index {index} out of range (split produced {len(starts)} segments)")

        add = segment_add["samples"]
        if not is_h3_av_latent(add):
            raise ValueError("MiniMaxH3LatentAppend expects a MiniMax H3 AV latent segment")
        av = add.tensors[0]
        aa = add.tensors[1]
        if av.shape[0] != 1:
            raise ValueError("MiniMaxH3LatentAppend expects batch-1 segments")
        lv = av.shape[2]
        la = aa.shape[-1]
        nm = segment_add.get("noise_mask")
        if nm is not None and nm.is_nested and len(nm.tensors) == 2:
            lv = min(_true_len(nm.tensors[0], dim=2), lv)
            la = min(_true_len(nm.tensors[1], dim=3), la)
        if lv <= 0 or la <= 0:
            raise ValueError(f"MiniMaxH3LatentAppend: segment {index} is empty after padding trim")

        gi = vtoks[index][0]
        agi = atoks[index][0]

        if segment_source is None:
            # nothing accumulated yet: only valid for index 0 (gi == 0); any
            # later index is caught by the gap check below
            src_v = torch.zeros((1, av.shape[1], 0, av.shape[3], av.shape[4]),
                                device=av.device, dtype=av.dtype)
            src_a = torch.zeros((1, 32, 2, 0), device=aa.device, dtype=aa.dtype)
        else:
            if not is_h3_av_latent(segment_source["samples"]):
                raise ValueError("MiniMaxH3LatentAppend expects segment_source to be a MiniMax H3 AV latent")
            src_v = segment_source["samples"].tensors[0]
            src_a = segment_source["samples"].tensors[1]
            if src_v.shape[1] != 24 or src_v.shape[3] != av.shape[3] or src_v.shape[4] != av.shape[4]:
                raise ValueError("segment_source and segment_add must share the same spatial size")

        if gi > src_v.shape[2]:
            raise ValueError(f"segment_source does not yet reach segment {index} (gap at video token {gi}); "
                             "feed it the accumulated result of the previous iterations")
        if agi > src_a.shape[-1]:
            raise ValueError(f"segment_source does not yet reach segment {index} (audio gap)")

        total_v = max(src_v.shape[2], gi + lv)
        total_a = max(src_a.shape[-1], agi + la)
        result_v = torch.zeros((1, src_v.shape[1], total_v, src_v.shape[3], src_v.shape[4]),
                               device=src_v.device, dtype=src_v.dtype)
        result_a = torch.zeros((1, 32, 2, total_a), device=src_a.device, dtype=src_a.dtype)
        result_v[:, :, :src_v.shape[2]] = src_v
        result_a[:, :, :, :src_a.shape[-1]] = src_a

        v = av[:, :, :lv]
        a = aa[:, :, :, :la]

        ov = (src_v.shape[2] - gi) if index > 0 else 0
        if ov > 0:
            ov = min(ov, lv)
            tail = result_v[:, :, gi:gi + ov].clone()
            result_v[:, :, gi:gi + ov] = _crossfade(tail, v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
        write_v = gi + max(ov, 0)
        if v.shape[2] > 0:
            result_v[:, :, write_v:write_v + v.shape[2]] = v

        ova = (src_a.shape[-1] - agi) if index > 0 else 0
        if ova > 0:
            ova = min(ova, la)
            tail = result_a[:, :, :, agi:agi + ova].clone()
            result_a[:, :, :, agi:agi + ova] = _crossfade(tail, a[:, :, :, :ova], dim=3)
            a = a[:, :, :, ova:]
        write_a = agi + max(ova, 0)
        if a.shape[-1] > 0:
            result_a[:, :, :, write_a:write_a + a.shape[-1]] = a

        out = {"samples": comfy.nested_tensor.NestedTensor((result_v, result_a))}
        return io.NodeOutput(out)


class MiniMaxH3LatentAnchor(io.ComfyNode):
    """Anchor the current segment to the previous iteration's re-sampled content.

    The frame-0 video keyframe (resolved_frame_index 0) of the segment's
    conditioning is replaced by the actual re-sampled boundary frame taken from
    `previous_result`. Keyframes are frozen rows in the H3 packed sequence, so
    the chunk's first frame is hard-pinned to the content the previous chunk
    ended with instead of the original (first-stage) keyframe, removing the
    detail mismatch at the seam."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentAnchor",
            display_name="Anchor MiniMax H3 Latent",
            category="model/latent/minimax",
            description=(
                "Replace the current segment's frame-0 video keyframe with the "
                "previous iteration's re-sampled boundary frame, so the chunk starts "
                "from the same content the previous chunk ended with (keyframes are "
                "frozen rows in the H3 packed sequence, so the seam frame is pinned). "
                "Wire it between the segment 'conditioning' and the re-sampler, "
                "feeding 'previous_result' from the previous iteration's 'Append "
                "MiniMax H3 Latents' output. A no-op for index 0 or when "
                "'previous_result' is unconnected."
            ),
            search_aliases=["h3 anchor", "anchor keyframe", "seam keyframe", "boundary anchor", "keyframe seam"],
            inputs=[
                io.Conditioning.Input("conditioning",
                                      tooltip="Current segment's conditioning ('conditioning' output of 'Extract MiniMax H3 Latent')."),
                io.Latent.Input("previous_result", optional=True,
                                tooltip="Accumulated latent of the previous iteration ('Append MiniMax H3 Latents' output). Its frame at the current segment's start becomes the new frame-0 keyframe. Leave unconnected for index 0."),
                H3_SPLIT_BATCH.Input("segments_info",
                                     tooltip="Segment metadata ('segments_info' output of the same 'Split MiniMax H3 Latent')."),
                io.Int.Input("index", default=0, min=0, max=100000, step=1,
                             tooltip="Index (0-based) of the current segment."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="How much of the re-sampled boundary content the frozen anchor row keeps: 1.0 = exact content (hardest anchor), 0.999 = model default, lower values mix in noise and weaken the anchor, 0.0 = pure noise (no anchoring)."),
            ],
            outputs=[
                io.Conditioning.Output("conditioning",
                                       tooltip="Conditioning with the frame-0 keyframe replaced by the re-sampled boundary frame."),
            ],
        )

    @classmethod
    def execute(cls, conditioning, previous_result=None, segments_info=None, index=0, anchor_strength=0.999) -> io.NodeOutput:
        if index == 0 or previous_result is None:
            return io.NodeOutput(conditioning)

        starts = segments_info.get("starts")
        if not starts:
            raise ValueError("MiniMaxH3LatentAnchor: segments_info is missing starts")
        if index < 0 or index >= len(starts):
            raise ValueError(f"index {index} out of range (split produced {len(starts)} segments)")

        f0 = starts[index]
        prev = previous_result["samples"]
        if not is_h3_av_latent(prev):
            raise ValueError("MiniMaxH3LatentAnchor expects previous_result to be a MiniMax H3 AV latent")
        pv = prev.tensors[0]

        t = tokens_for_frames(f0)
        if t >= pv.shape[2]:
            raise ValueError("previous_result does not extend to the current segment's start frame")
        if f0 != frames_for_tokens(t):
            raise ValueError(f"segment start {f0} is not a video-token boundary (token start {frames_for_tokens(t)})")

        anchor_kf = {"resolved_frame_index": 0, "latent": pv[:, :, t:t + 1].contiguous()}
        aug = max(0.0, min(1.0, float(anchor_strength)))

        out = []
        for tensor, d in conditioning:
            nd = dict(d)
            kfs = nd.get("minimax_keyframes")
            if kfs:
                kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
                nd["minimax_keyframes"] = [anchor_kf] + kept
            else:
                nd["minimax_keyframes"] = [anchor_kf]
            nd["minimax_visual_cond_noise_aug"] = aug
            out.append([tensor, nd])
        return io.NodeOutput(out)


class MiniMaxH3SpatialSplit(io.ComfyNode):
    """EXPERIMENTAL. Split a MiniMax H3 AV latent into a spatial tile grid.

    Each tile keeps the FULL temporal length and is a spatial crop of the source
    (latent-space, so it is a valid standalone video for re-sampling). Tiles are
    zero-padded to a uniform size; the padded border is frozen via the tile's
    noise_mask. Overlap strips shared with the neighbor that is processed FIRST
    (row-major order: left/top neighbors) fade the mask from frozen at the seam
    to free at the tile interior, so the second tile keeps the first tile's
    re-sampled content at the seam. Audio is carried unchanged in every tile and
    frozen (mask 0) so spatial tiling never re-samples audio."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpatialSplit",
            display_name="Split MiniMax H3 Latent (Spatial)",
            category="model/latent/minimax",
            description=(
                "EXPERIMENTAL. Split a denoised MiniMax H3 AV latent into a 2D grid of "
                "spatially-cropped tiles (each tile keeps the full temporal length) so "
                "each tile can be re-sampled separately and stitched back, bounding "
                "peak VRAM by one tile. Process tiles row-major with 'Extract MiniMax "
                "H3 Spatial Tile' + 'Append MiniMax H3 Spatial Tile'. The tile noise "
                "mask fades overlap strips that are shared with the already-processed "
                "left/top neighbor (frozen at the seam, free toward the interior), so "
                "seams keep the earlier neighbor's re-sampled content. Audio is frozen "
                "in every tile (never re-sampled)."
            ),
            search_aliases=["h3 spatial split", "spatial split", "tile latent", "h3 tile"],
            inputs=[
                io.Latent.Input("latent", tooltip="Denoised MiniMax H3 AV latent to tile."),
                io.Conditioning.Input("conditioning", optional=True,
                                      tooltip="Conditioning used to generate this latent. Its minimax_keyframes are spatially cropped per tile."),
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="Tile width in PIXELS. Must be a multiple of 32."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="Tile height in PIXELS. Must be a multiple of 32."),
                io.Int.Input("overlap", default=64, min=0, max=100000, step=32,
                             tooltip="Overlap in PIXELS between neighbouring tiles (both axes). Must be a multiple of 32 and smaller than the tile size. 32-96 px recommended; the shared strip fades in the mask."),
                io.Int.Input("fade_width", default=0, min=0, max=100000, step=32,
                             tooltip="Width in PIXELS of the mask fade from the frozen seam toward the free interior (applies to each shared overlap strip, in both earlier and later modes). 0 (default) freezes the whole overlap strip with no fade - a hard test of the noise_mask freeze. Set to 'overlap' to restore the whole-strip fade."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier",
                               tooltip="How tiles are stitched together (the append node follows this via 'auto'). 'earlier' (default): the already-stitched content wins each shared overlap band. 'later': the re-sampled tile wins. This does NOT affect the noise mask - the freeze/fade of each strip is controlled by fade_width alone."),
            ],
            outputs=[
                io.Latent.Output("tiles", tooltip="Tile batch (N, 24, T, th, tw) with a noise_mask: padding and done-neighbor overlap strips are frozen."),
                H3_COND_BATCH.Output("cond_batch", tooltip="Per-tile conditioning with spatially-cropped keyframes. None unless 'conditioning' was connected."),
                H3_SPATIAL_INFO.Output("tiles_info", tooltip="Tile grid metadata consumed by the spatial extract/append nodes."),
                io.Int.Output("grid_rows", tooltip="Number of tile rows."),
                io.Int.Output("grid_cols", tooltip="Number of tile columns."),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        for name in ("tile_width", "tile_height", "overlap"):
            v = kwargs.get(name)
            if isinstance(v, int) and v % 32 != 0:
                return (f"'{name}' must be a multiple of 32 pixels (the model's 2x2 "
                        f"latent patch grid); got {v}.")
        tw = kwargs.get("tile_width")
        th = kwargs.get("tile_height")
        ov = kwargs.get("overlap")
        fw = kwargs.get("fade_width")
        if isinstance(tw, int) and isinstance(th, int) and isinstance(ov, int):
            if ov >= tw or ov >= th:
                return "overlap must be smaller than both tile dimensions."
        if isinstance(fw, int) and isinstance(ov, int) and fw > ov:
            return "fade_width must not exceed overlap."
        return True

    @classmethod
    def execute(cls, latent, tile_width, tile_height, overlap, overlap_mode="earlier", conditioning=None, fade_width=0) -> io.NodeOutput:
        if tile_width % 32 != 0 or tile_height % 32 != 0 or overlap % 32 != 0:
            raise ValueError("tile_width, tile_height and overlap must be multiples of 32 pixels")
        if overlap >= tile_width or overlap >= tile_height:
            raise ValueError("overlap must be smaller than both tile dimensions")
        if fade_width < 0 or fade_width > overlap:
            raise ValueError("fade_width must be between 0 and overlap")

        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MiniMaxH3SpatialSplit expects a MiniMax H3 AV latent")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("MiniMaxH3SpatialSplit expects a single-video latent (batch 1)")
        _, c, t, h, w = video.shape
        ta = audio.shape[-1]

        th = tile_height // 16
        tw = tile_width // 16
        ol = overlap // 16
        fw = fade_width // 16
        rows, cols, trows, tcols = compute_spatial_grid(h, w, th, tw, ol, ol)
        nrows, ncols = len(rows), len(cols)
        n = nrows * ncols

        batch_v = torch.zeros((n, c, t, th, tw), device=video.device, dtype=video.dtype)
        batch_a = torch.zeros((n, 32, 2, ta), device=audio.device, dtype=audio.dtype)
        mask_v = torch.zeros((n, c, t, th, tw), device=video.device, dtype=video.dtype)
        mask_a = torch.zeros((n, 32, 2, ta), device=audio.device, dtype=audio.dtype)

        for i in range(nrows):
            for j in range(ncols):
                k = i * ncols + j
                tr = trows[i]
                tc = tcols[j]
                batch_v[k, :, :, :tr, :tc] = video[0, :, :, rows[i]:rows[i] + tr, cols[j]:cols[j] + tc]
                batch_a[k] = audio[0]
                m = spatial_fade_mask(th, tw, tr, tc, min(ol, tr), min(ol, tc),
                                      done_top=(i > 0), done_left=(j > 0), overlap_mode=overlap_mode,
                                      fade_width=fw)
                mask_v[k] = m[None, None, None]

        out_latent = {
            "samples": comfy.nested_tensor.NestedTensor((batch_v, batch_a)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mask_v, mask_a)),
        }

        conds = None
        if conditioning is not None:
            conds = [crop_keyframes_to_tile(conditioning, h, w, rows[i], cols[j], trows[i], tcols[j])
                     for i in range(nrows) for j in range(ncols)]

        tiles_info = {
            "rows": rows,
            "cols": cols,
            "tile_h": th,
            "tile_w": tw,
            "overlap": ol,
            "tile_rows": trows,
            "tile_cols": tcols,
            "n_cols": ncols,
            "orig_h": h,
            "orig_w": w,
            "overlap_mode": overlap_mode,
        }
        return io.NodeOutput(out_latent, conds, tiles_info, nrows, ncols)


class MiniMaxH3SpatialExtract(io.ComfyNode):
    """EXPERIMENTAL. Pull one spatial tile out of a Split MiniMax H3 Latent
    (Spatial) batch and trim it to its true extent. Returns the tile latent
    together with its noise_mask (frozen padding + done-overlap fades), so a
    re-sampler keeps the shared seams. When `accumulated` (the running result of
    the spatial append loop) is connected, the overlap strips shared with the
    already-processed left/top neighbor are pre-filled with the neighbor's
    re-sampled content, which the frozen mask then keeps."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpatialExtract",
            display_name="Extract MiniMax H3 Spatial Tile",
            category="model/latent/minimax",
            description=(
                "EXPERIMENTAL. Pull one tile out of a 'Split MiniMax H3 Latent "
                "(Spatial)' batch, trimmed to its true extent, together with its "
                "noise_mask (padding and done-overlap strips are frozen). Optionally "
                "pre-fill the done-overlap strips from the running result of the "
                "'Append MiniMax H3 Spatial Tile' loop so the re-sampler keeps the "
                "neighbor's completed content at the seam."
            ),
            search_aliases=["h3 spatial extract", "spatial tile extract", "extract tile"],
            inputs=[
                io.Latent.Input("tiles", tooltip="Tile batch produced by 'Split MiniMax H3 Latent (Spatial)'."),
                H3_COND_BATCH.Input("cond_batch", optional=True,
                                    tooltip="cond_batch output of the same split. The selected tile's conditioning is returned on the 'conditioning' output."),
                H3_SPATIAL_INFO.Input("tiles_info", tooltip="tiles_info output of the same split."),
                io.Int.Input("index", default=0, min=0, max=100000,
                             tooltip="Tile index (0-based, row-major). Negative counts from the end."),
                io.Latent.Input("accumulated", optional=True,
                                tooltip="Running result of the 'Append MiniMax H3 Spatial Tile' loop (previous iterations). Its content in the done-overlap strips is copied into the tile before re-sampling. Leave unconnected for the first tile / to freeze strips to the original content."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="The tile, trimmed to its true extent, with a noise_mask attached."),
                io.Conditioning.Output("conditioning", display_name="conditioning",
                                       tooltip="Conditioning of the selected tile (spatially-cropped keyframes)."),
            ],
        )

    @classmethod
    def execute(cls, tiles, cond_batch=None, tiles_info=None, index=0, accumulated=None) -> io.NodeOutput:
        ncols = tiles_info["n_cols"]
        n = len(tiles_info["rows"]) * ncols
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise ValueError("index {} out of range for {} tiles".format(index, n))
        i = index // ncols
        j = index % ncols

        samples = tiles["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MiniMaxH3SpatialExtract expects a MiniMax H3 AV latent batch")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        mask = tiles.get("noise_mask")

        tr = tiles_info["tile_rows"][i]
        tc = tiles_info["tile_cols"][j]
        v = video[index:index + 1, :, :, :tr, :tc].contiguous()
        a = audio[index:index + 1].contiguous()

        if accumulated is not None:
            acc = accumulated["samples"]
            if not is_h3_av_latent(acc):
                raise ValueError("MiniMaxH3SpatialExtract expects accumulated to be a MiniMax H3 AV latent")
            acc_v = acc.tensors[0]
            r0 = tiles_info["rows"][i]
            c0 = tiles_info["cols"][j]
            ov = min(tiles_info["overlap"], tr, tc)
            if j > 0 and ov > 0:
                v[:, :, :, :, :ov] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + ov]
            if i > 0 and ov > 0:
                v[:, :, :, :ov, :] = acc_v[:, :, :, r0:r0 + ov, c0:c0 + tc]

        out = {"samples": comfy.nested_tensor.NestedTensor((v, a))}
        if mask is not None and mask.is_nested and len(mask.tensors) == 2:
            mv = mask.tensors[0][index:index + 1, :, :, :tr, :tc].contiguous()
            ma = mask.tensors[1][index:index + 1].contiguous()
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))

        conditioning = None
        if cond_batch is not None:
            conditioning = cond_batch[index]
        return io.NodeOutput(out, conditioning)


class MiniMaxH3SpatialAppend(io.ComfyNode):
    """EXPERIMENTAL. Stitch one re-sampled spatial tile into the accumulated
    latent (the running result of the spatial loop). The tile is placed at its
    grid position; overlap bands shared with the previous row/column tile are
    cross-faded against what is already there. Audio is carried from the base
    unchanged (spatial tiling never re-samples audio)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpatialAppend",
            display_name="Append MiniMax H3 Spatial Tile",
            category="model/latent/minimax",
            description=(
                "EXPERIMENTAL. Stitch one re-sampled spatial tile into the running "
                "result of the spatial loop. The tile is written at its grid position "
                "and the overlap bands shared with the previous row/column tile are "
                "cross-faded against the existing content. Wire 'source' to the "
                "original latent and 'accumulated' to this node's previous output "
                "(None on the first iteration); after the last tile the output is the "
                "full re-sampled latent."
            ),
            search_aliases=["h3 spatial append", "spatial tile append", "splice tile"],
            inputs=[
                io.Latent.Input("segment_add", tooltip="The re-sampled tile (output of the re-sampler; 'Extract MiniMax H3 Spatial Tile' for the first iteration / after carry)."),
                io.Latent.Input("source", tooltip="The original full latent (the input to the spatial split)."),
                io.Latent.Input("accumulated", optional=True,
                                tooltip="Previous output of this node (None on the first iteration)."),
                io.Int.Input("index", default=0, min=0, max=100000,
                             tooltip="Tile index (0-based, row-major) of segment_add."),
                H3_SPATIAL_INFO.Input("tiles_info", tooltip="tiles_info output of the same split."),
                io.Combo.Input("overlap_mode", options=["auto", "earlier", "later"], default="auto",
                               tooltip="Who wins each shared overlap band when stitching. 'auto' (default) uses the split's overlap_mode. 'earlier': the accumulated (already stitched) content dominates the band. 'later': the re-sampled tile dominates."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="How the overlap band transitions. 'linear': straight cross-fade (default). 'smoothstep': eased fade, zero slope at both ends. 'overwrite': the overlap_mode side takes the whole band verbatim. 'midpoint': hard switch at the band's middle."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Accumulated latent with the tile stitched in. After the last tile this is the full result."),
            ],
        )

    @classmethod
    def execute(cls, segment_add, source, accumulated=None, index=0, tiles_info=None,
                overlap_mode="auto", overlap_blend="linear") -> io.NodeOutput:
        ncols = tiles_info["n_cols"]
        n = len(tiles_info["rows"]) * ncols
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise ValueError("index {} out of range for {} tiles".format(index, n))
        i = index // ncols
        j = index % ncols
        if overlap_mode == "auto":
            overlap_mode = tiles_info.get("overlap_mode", "earlier")
        if overlap_mode not in ("earlier", "later"):
            raise ValueError("overlap_mode must be 'earlier' or 'later' (got {!r})".format(overlap_mode))
        if overlap_blend not in ("linear", "smoothstep", "overwrite", "midpoint"):
            raise ValueError("overlap_blend must be one of linear/smoothstep/overwrite/midpoint (got {!r})".format(overlap_blend))

        add = segment_add["samples"]
        if not is_h3_av_latent(add):
            raise ValueError("MiniMaxH3SpatialAppend expects segment_add to be a MiniMax H3 AV latent")
        av = add.tensors[0]
        aa = add.tensors[1]
        if av.shape[0] != 1:
            raise ValueError("MiniMaxH3SpatialAppend expects batch-1 tiles")

        src = source["samples"]
        if not is_h3_av_latent(src):
            raise ValueError("MiniMaxH3SpatialAppend expects source to be a MiniMax H3 AV latent")
        src_v = src.tensors[0]
        if src_v.shape[0] != 1:
            raise ValueError("MiniMaxH3SpatialAppend expects a single-video source")

        base = accumulated["samples"] if accumulated is not None else src
        if not is_h3_av_latent(base):
            raise ValueError("MiniMaxH3SpatialAppend expects accumulated to be a MiniMax H3 AV latent")
        base_v = base.tensors[0]
        if base_v.shape[0] != 1:
            raise ValueError("MiniMaxH3SpatialAppend expects a single-video base latent")
        result_v = base_v.clone()
        result_a = base.tensors[1].clone()

        if tiles_info["orig_h"] != result_v.shape[3] or tiles_info["orig_w"] != result_v.shape[4]:
            raise ValueError(
                f"base latent is {result_v.shape[3]}x{result_v.shape[4]} but the spatial split was made "
                f"for {tiles_info['orig_h']}x{tiles_info['orig_w']}. 'source' must be the ORIGINAL full "
                f"latent you passed to the split (the upscaled one, here {tiles_info['orig_w'] * 16}x"
                f"{tiles_info['orig_h'] * 16}px) and 'accumulated' the previous output of this node - "
                f"not the original small latent, an extracted tile, or the split's 'tiles' batch."
            )

        r0 = tiles_info["rows"][i]
        c0 = tiles_info["cols"][j]
        tr = min(tiles_info["tile_rows"][i], av.shape[3])
        tc = min(tiles_info["tile_cols"][j], av.shape[4])
        ov = min(tiles_info["overlap"], tr, tc)
        if r0 + tr > result_v.shape[3] or c0 + tc > result_v.shape[4]:
            raise ValueError(
                f"tile {index} ({r0}x{c0} grid position, {tr}x{tc} true extent) does not fit inside the "
                f"accumulated latent ({result_v.shape[3]}x{result_v.shape[4]}). Check the wiring: 'source' "
                f"must be the ORIGINAL full latent (the split input), and 'accumulated' the previous output "
                f"of this node - full size, not a tile. Passing an extracted tile or the split's 'tiles' "
                f"batch here makes the base latent tile-sized and the later tiles no longer fit."
            )

        region = result_v[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
        tile_v = av[:, :, :, :tr, :tc]

        if j > 0 and ov > 0:
            t = torch.linspace(0.0, 1.0, ov, device=result_v.device, dtype=result_v.dtype)
            w = blend_weights(t, overlap_blend, overlap_mode)
            region[:, :, :, :, :ov] = region[:, :, :, :, :ov] * (1.0 - w[None, None, None, None, :]) + tile_v[:, :, :, :, :ov] * w[None, None, None, None, :]
        if i > 0 and ov > 0:
            t = torch.linspace(0.0, 1.0, ov, device=result_v.device, dtype=result_v.dtype)
            w = blend_weights(t, overlap_blend, overlap_mode)
            region[:, :, :, :ov, :] = region[:, :, :, :ov, :] * (1.0 - w[None, None, None, :, None]) + tile_v[:, :, :, :ov, :] * w[None, None, None, :, None]

        band = torch.zeros(1, 1, 1, tr, tc, device=result_v.device, dtype=torch.bool)
        if ov > 0:
            if j > 0:
                band[:, :, :, :, :ov] = True
            if i > 0:
                band[:, :, :, :ov, :] = True
        region = torch.where(band, region, tile_v)
        result_v[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

        out = {"samples": comfy.nested_tensor.NestedTensor((result_v, result_a))}
        return io.NodeOutput(out)
