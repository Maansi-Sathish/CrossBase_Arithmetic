"""Inference loop: generate on each prompt and capture / ablate internal activations via TransformerLens.

This file works on any decoder-only model out of the box (it talks to the model only through the
TransformerLens `TransformerBridge` from src/model.py). Two functions are the task-specific
override points you may want to customise (both have sensible defaults and `TODO` markers):
  - find_answer_span(generated_text):        which span of the generation is "the answer".
  - find_positions_of_interest(model, prompt): which prompt token positions to capture for geometry.
Everything else (the capture, ablation hooks, and the run loop) is task-agnostic.

HOW CAPTURE AND ABLATION WORK HERE (the TransformerLens idea):
    Instead of hand-writing PyTorch forward hooks per architecture, we name the activation we want
    with a TransformerLens "hook name" and let the library do the rest. Two calls cover everything:
      - model.run_with_cache(tokens) -> (logits, cache): runs the model once and hands back a
        `cache` you can index by hook name to read EVERY layer/position activation. We use:
            blocks.{l}.hook_resid_post   residual stream after block l   [batch, pos, d_model]
            blocks.{l}.mlp.hook_post     MLP intermediate "neurons"      [batch, pos, d_mlp]
            blocks.{l}.attn.hook_z       per-head attention output       [batch, pos, n_heads, d_head]
            hook_embed                   token embedding (pre-block 0)   [batch, pos, d_model]
            hook_pos_embed               positional embedding (pre-block 0) [batch, pos, d_model]
            blocks.0.hook_resid_pre      token_embed + pos_embed (block 0's input) [batch, pos, d_model]
      - model.run_with_hooks(tokens, fwd_hooks=[(name, fn), ...]): runs the model while calling
        each fn on the named activation, letting fn EDIT it (we set entries to 0 to ablate).

    Because the cache holds all positions from a single pass over the full output sequence, we can
    read the answer token's activations directly -- there is no "last token has no forward pass"
    edge case to work around.
"""

import re
from typing import Any

import torch
from tqdm.auto import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

from utils.dataset import PromptDataset


def find_answer_span(generated_text: str) -> tuple[str, int, int] | None:
    """Locate the model's answer inside the text it generated.

    WHY this exists: the model just produces a stream of tokens; we need to know which part of
    that stream is "the answer" so we can read activations at the right place. This function
    returns the character span of the answer; inference.py then maps it to tokens and captures
    activations at the answer's LAST token.

    Our model outputs answers in reversed-digit order (LSD first), terminated
    by a newline or end of string. The answer is the first unbroken run of
    valid hex characters [0-9A-F] after any leading whitespace.

    We match [0-9A-F]+ because:
      - Binary answers only use 0 and 1 — both in this set
      - Octal answers use 0-7 — all in this set
      - Decimal answers use 0-9 — all in this set
      - Hex answers use 0-9 and A-F — exactly this set
    So one pattern covers all four bases without needing to know the base.

    Args:
        generated_text: The decoded text the model generated after the prompt.

    Returns:
        (answer_text, start_char, end_char): the answer substring and its character offsets
        within generated_text, or None if no answer could be found (that prompt is skipped).
    """
    # Match the first contiguous block of valid characters in our alphabet.
    # ^ anchors to the start so we don't accidentally grab digits from later lines.
    # \s* skips any leading whitespace the model may have emitted before the answer.
    match = re.search(r"^\s*([0-9A-F]+)", generated_text)
    if not match:
        # Model generated something we can't parse — skip this prompt
        return None

    # Return (answer string, start char index, end char index)
    return match.group(1), match.start(1), match.end(1)


def find_positions_of_interest(model: TransformerBridge, prompt: str) -> dict[str, int | None]:
    """Return extra PROMPT token positions to capture activations at (for --capture-geometry).

    WHY this exists: by default we capture activations only at the answer token. But for
    circuit analysis we also want to know what the model is doing at specific INPUT positions
    — e.g. which head is looking at the carry digit, which head tracks the operator.

    For cross-base arithmetic we care about six positions in the TEST question
    (the last line of the prompt — not the few-shot examples):

      first_digit_a  — leftmost digit of operand a  (most significant)
      last_digit_a   — rightmost digit of operand a (carry originates here)
      operator       — the + sign
      first_digit_b  — leftmost digit of operand b
      last_digit_b   — rightmost digit of operand b (carry originates here)
      eq_sign        — the = sign (transition from reading to generating)

    Why these specifically:
      - If shared circuits exist across bases, heads that attend to the operator
        or eq_sign should fire identically regardless of which base is being used.
      - Carry-tracking heads should show high attention weights on last_digit_a
        and last_digit_b, since that's where carries originate.
      - first_digit_a / first_digit_b capture how the model encodes magnitude,
        useful for probing whether number representations are shared or base-specific.

    All positions reference the LAST line only (the actual test question),
    because the few-shot lines are just context — we don't analyze those.

    Args:
        model: The TransformerLens bridge. model.to_str_tokens(prompt, prepend_bos=False)
               returns the prompt as a list of per-token strings.
        prompt: The full prompt string.

    Returns:
        dict mapping position name -> token index within the prompt (or None if not found).
        IMPORTANT: indices are WITHOUT BOS prefix, matching how inference.py tokenizes.
    """
    # Split the prompt into individual character tokens.
    # e.g. "123+456=" -> ["1","2","3","+","4","5","6","="]
    # (plus the few-shot lines above, each separated by "\n")
    str_tokens = model.to_str_tokens(prompt, prepend_bos=False)

    positions: dict[str, int | None] = {}

    # --- Step 1: Find the last + and = in the entire token list ---
    # The few-shot lines also contain + and =, so we scan ALL tokens and keep
    # only the LAST occurrence of each — those belong to the test question.
    last_plus = None
    last_eq = None
    for idx, tok in enumerate(str_tokens):
        if tok == "+":
            last_plus = idx   # keep updating — we want the very last one
        elif tok == "=":
            last_eq = idx

    # Store operator and eq_sign positions
    # These are the most important positions for cross-base comparison:
    # if the model has a shared circuit, heads attending to + or = should
    # behave identically regardless of which base the prompt is in.
    positions["operator"] = last_plus
    positions["eq_sign"] = last_eq

    # --- Step 2: Locate operand a's digits (everything between last \n and +) ---
    if last_plus is not None:

        # last_digit_a is the token immediately before the + sign.
        # This is the LEAST significant digit of a — where carry arithmetic starts.
        # A carry-tracking head should show high attention weights here.
        positions["last_digit_a"] = last_plus - 1

        # Scan backwards from + to find where the last line begins.
        # The last line starts right after the last \n token (or at index 0
        # if there's no \n, which shouldn't happen in few-shot prompts).
        start_of_last_line = 0
        for idx in range(last_plus - 1, -1, -1):
            if str_tokens[idx] == "\n":
                start_of_last_line = idx + 1  # first token after the newline
                break

        # first_digit_a is the leftmost token of the last line.
        # This is the MOST significant digit of a.
        # Useful for probing how the model encodes number magnitude.
        positions["first_digit_a"] = start_of_last_line

    # --- Step 3: Locate operand b's digits (everything between + and =) ---
    if last_plus is not None and last_eq is not None:

        # first_digit_b is the token immediately after +
        # (most significant digit of b)
        positions["first_digit_b"] = last_plus + 1

        # last_digit_b is the token immediately before =
        # (least significant digit of b — carry originates here too)
        positions["last_digit_b"] = last_eq - 1

    return positions


def _map_char_to_token(model: TransformerBridge, token_ids: torch.Tensor, target_char_position: int) -> int | None:
    """Map a character position (in the decoded text) to the token index that covers it.

    We decode the tokens one more at a time; the first prefix whose decoded text is long enough to
    reach `target_char_position` is the token that contains that character.

    Args:
        model: The TransformerLens bridge (for decoding via model.to_string).
        token_ids: 1-D tensor of token ids that were decoded.
        target_char_position: A character index into the decoded text.

    Returns:
        The token index containing that character, or None if it can't be mapped.
    """
    for token_idx in range(len(token_ids)):
        text_so_far = model.to_string(token_ids[: token_idx + 1])
        if len(text_so_far) > target_char_position:
            return token_idx
    return None


def _locate_answer(
    model: TransformerBridge,
    prompt_length: int,
    output_ids: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[int | None, torch.Tensor | None, str | None, torch.Tensor | None]:
    """Find the answer span in the generated text and report which token to read activations from.

    Steps:
      1. Decode the generated tokens (everything after the prompt) to text.
      2. Ask `find_answer_span` which substring is "the answer" (you override that per task).
      3. Map that substring's first and last characters back to token indices.

    CONVENTION: we key everything off the answer span's LAST token (e.g. the '2' of "42", the
    '3' of "-63"). The caller then reads all activations at that one ABSOLUTE position, so every
    captured tensor describes the same token. We also return the full span's token ids / string so
    you still know the whole answer, and the logits the last answer token was predicted from.

    Args:
        model: The TransformerLens bridge (used for decoding).
        prompt_length: Number of prompt tokens (generation starts right after this index).
        output_ids: Full token sequence [prompt tokens + generated tokens], shape [seq_len].
        logits: Logits over the full output sequence, shape [1, seq_len, vocab_size].

    Returns:
        (answer_position, answer_token_ids, answer_token, answer_logits), or all None if no answer
        span could be located. answer_position is the ABSOLUTE index (into output_ids) of the
        answer span's last token.
    """
    generated_ids = output_ids[prompt_length:]
    if len(generated_ids) == 0:
        return None, None, None, None
    generated_text = model.to_string(generated_ids)

    span = find_answer_span(generated_text)
    if span is None:
        return None, None, None, None
    answer_text, answer_start_char, answer_end_char = span

    # Translate character positions in the generated text into token indices (relative to the
    # generated chunk).
    answer_start_token = _map_char_to_token(model, generated_ids, answer_start_char)
    answer_end_token = _map_char_to_token(model, generated_ids, answer_end_char - 1)
    if answer_start_token is None or answer_end_token is None:
        return None, None, None, None

    # The answer span may be several tokens (e.g. "42" -> ['4', '2']); keep the whole span for
    # reference but key the position off its LAST token (answer_end_token).
    answer_token_ids = generated_ids[answer_start_token : answer_end_token + 1]
    answer_token = model.to_string(answer_token_ids)
    if answer_token.strip() != answer_text:
        return None, None, None, None

    # Absolute position of the answer span's last token within the full output sequence.
    answer_position = prompt_length + answer_end_token

    # Logits the last answer token was predicted from: in an autoregressive model the prediction
    # for position p is computed at position p-1, so we read logits one step earlier.
    pred_position = answer_position - 1
    if 0 <= pred_position < logits.shape[1]:
        answer_logits = logits[0, pred_position].detach().cpu()
    else:
        answer_logits = None

    return answer_position, answer_token_ids, answer_token, answer_logits


def build_ablation_hooks(
    mlp_ablation: dict[int, list[int]] | None,
    head_ablation: dict[int, list[int]] | None,
) -> list[tuple[str, Any]]:
    """Build the TransformerLens forward-hook list that zeroes out the requested neurons/heads.

    Each entry is a (hook_name, fn) pair. TransformerLens calls fn(activation, hook) during the
    forward pass and uses fn's return value in place of the original activation, so setting some
    entries to 0 ABLATES (knocks out) those components -- the core causal test of mechinterp.

    Args:
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} of MLP neurons to zero, applied
            to blocks.{layer}.mlp.hook_post (shape [batch, pos, d_mlp]).
        head_ablation: Optional dict {layer_idx: [head indices]} of attention heads to zero,
            applied to blocks.{layer}.attn.hook_z (shape [batch, pos, n_heads, d_head]).

    Returns:
        A list of (hook_name, hook_fn) pairs suitable for model.run_with_hooks / model.hooks.
        Empty if no ablation was requested.
    """
    hooks: list[tuple[str, Any]] = []

    if mlp_ablation:
        for layer_idx, neuron_indices in mlp_ablation.items():
            def mlp_hook(act: torch.Tensor, hook: HookPoint, idxs: list[int] = list(neuron_indices)) -> torch.Tensor:
                act[..., idxs] = 0.0  # zero these neurons at every position
                return act

            hooks.append((f"blocks.{layer_idx}.mlp.hook_post", mlp_hook))

    if head_ablation:
        for layer_idx, head_indices in head_ablation.items():
            def head_hook(act: torch.Tensor, hook: HookPoint, idxs: list[int] = list(head_indices)) -> torch.Tensor:
                act[:, :, idxs, :] = 0.0  # zero these heads' outputs at every position
                return act

            hooks.append((f"blocks.{layer_idx}.attn.hook_z", head_hook))

    return hooks


def _read_activations_at(
    cache: dict[str, torch.Tensor],
    layer_indices: list[int],
    pos: int | None,
) -> dict[str, Any]:
    """Read MLP-neuron, per-head, residual, and embedding activations at ONE token position."""
    embed = cache["hook_embed"][0]
    in_range = pos is not None and 0 <= pos < embed.shape[0]

    return {
        "mlp_neurons": {
            i: cache[f"blocks.{i}.mlp.hook_post"][0, pos].detach().cpu() if in_range else None for i in layer_indices
        },
        "attn_heads": {
            i: cache[f"blocks.{i}.attn.hook_z"][0, pos].detach().cpu() if in_range else None for i in layer_indices
        },
        "residual": {
            i: cache[f"blocks.{i}.hook_resid_post"][0, pos].detach().cpu() if in_range else None for i in layer_indices
        },
        "token_embedding": embed[pos].detach().cpu() if in_range else None,
        "pos_embedding": cache["hook_pos_embed"][0, pos].detach().cpu() if in_range else None,
        "resid_pre": cache["blocks.0.hook_resid_pre"][0, pos].detach().cpu() if in_range else None,
    }


def _capture_geometry(
    cache: dict[str, torch.Tensor],
    layer_indices: list[int],
    positions: dict[str, int | None],
) -> dict[str, dict[int, dict[str, torch.Tensor | None]]]:
    """Slice residual / MLP / head / embedding activations at named prompt positions out of the cache."""
    geometry: dict[str, dict] = {}
    for name, pos in positions.items():
        act = _read_activations_at(cache, layer_indices, pos)
        geometry[name] = {
            -1: {
                "resid_pre": act["resid_pre"],
                "mlp": None,
                "heads": None,
                "token_embedding": act["token_embedding"],
                "pos_embedding": act["pos_embedding"],
            },
            **{
                layer_idx: {
                    "residual": act["residual"][layer_idx],
                    "mlp": act["mlp_neurons"][layer_idx],
                    "heads": act["attn_heads"][layer_idx],
                }
                for layer_idx in layer_indices
            },
        }

    return geometry


def _run_single_prompt(
    model: TransformerBridge,
    prompt: str,
    prompt_length: int,
    layer_indices: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> dict[str, Any]:
    """Run a single prompt through the model, capturing answer-token and geometry activations."""
    ablation_hooks = build_ablation_hooks(mlp_ablation, head_ablation)
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

    # Step 1: Generate the answer (with any ablation hooks active)
    with model.hooks(fwd_hooks=ablation_hooks):
        out_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="tokens",
            verbose=False,
        )
    output_ids = out_tokens[0].detach().cpu()
    generated_text = model.to_string(output_ids)
    completion = model.to_string(output_ids[prompt_length:]).strip()

    # Step 2: Re-run the full output sequence to get logits and activations
    if capture_geometry:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits, cache = model.run_with_cache(out_tokens)
    else:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits = model.run_with_hooks(out_tokens, return_type="logits")
        cache = None

    # Step 3: Find where the answer is in the generated text
    answer_position, answer_token_ids, answer_token, answer_logits = _locate_answer(
        model, prompt_length, output_ids, logits
    )

    # Step 4: Read activations at the answer token position and at named prompt positions
    if capture_geometry and cache is not None:
        answer_act = _read_activations_at(cache, layer_indices, answer_position)
        mlp_neurons = answer_act["mlp_neurons"]
        attn_heads = answer_act["attn_heads"]
        answer_residual = answer_act["residual"]
        answer_token_embedding = answer_act["token_embedding"]
        answer_pos_embedding = answer_act["pos_embedding"]
        answer_resid_pre = answer_act["resid_pre"]

        # Capture activations at the named prompt positions (operator, digits, etc.)
        positions = find_positions_of_interest(model, prompt)
        geometry = _capture_geometry(cache, layer_indices, positions)
    else:
        mlp_neurons = {i: None for i in layer_indices}
        attn_heads = {i: None for i in layer_indices}
        answer_residual = {i: None for i in layer_indices}
        answer_token_embedding = None
        answer_pos_embedding = None
        answer_resid_pre = None
        geometry = {}

    return {
        "text": generated_text,
        "completion": completion,
        "output_ids": output_ids if capture_geometry else None,
        "answer": {
            "position": answer_position,
            "token_id": answer_token_ids,
            "token": answer_token,
            "mlp_neurons": mlp_neurons,
            "attn_heads": attn_heads,
            "residual": answer_residual,
            "token_embedding": answer_token_embedding,
            "pos_embedding": answer_pos_embedding,
            "resid_pre": answer_resid_pre,
            "logits": answer_logits if answer_logits is not None else None,
        },
        "geometry": geometry,
    }


def run(
    model: TransformerBridge,
    dataset: PromptDataset,
    layers: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Run inference over a dataset, capturing MLP-neuron, per-head, residual, and geometry activations."""
    results: list[dict[str, Any]] = []
    for entry in tqdm(dataset.prompts, desc="Running neuron inference"):
        prompt = entry["prompt"]
        prompt_length = model.to_tokens(prompt, prepend_bos=False).shape[1]

        run_result = _run_single_prompt(
            model,
            prompt,
            prompt_length,
            layers,
            max_new_tokens,
            mlp_ablation=mlp_ablation,
            head_ablation=head_ablation,
            capture_geometry=capture_geometry,
        )

        results.append(
            {
                "prompt": prompt,
                "prompt_length": prompt_length,
                "metadata": entry.get("metadata", {}),
                "result": run_result,
            }
        )

    return results