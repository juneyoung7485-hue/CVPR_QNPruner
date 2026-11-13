"""Index bookkeeping for decoder-side pruning that really shortens the sequence.

FastV and SparseVLM drop visual tokens *inside* the decoder. Doing that for
real -- slicing ``hidden_states`` rather than masking columns -- means four
index spaces have to stay consistent as the sequence shrinks:

    sequence position   where a token sits in the decoder's current input
    original position   where it sat before any pruning (its RoPE position)
    visual index        which of the L_v tokens that entered the LLM it is
    rater position      which text tokens SparseVLM lets vote

``PrunedSequence`` owns all four. Every prune goes through :meth:`keep`, which
takes the visual tokens to retain and returns the sequence indices to slice
with, so the model-side code never recomputes an index by hand.

Two invariants the tests pin down, because breaking either is silent:

* **Text is never pruned.** Only visual positions are candidates; system,
  question and answer tokens always survive.
* **Order is preserved.** ``keep`` returns sorted sequence indices, so token
  order and therefore RoPE ordering never changes. Positions are *not*
  renumbered: a surviving token keeps its original position id, so the keys
  already in the KV cache stay consistent with queries that arrive later.
"""
import torch


class PrunedSequence:
    """The decoder's current token layout, as pruning shortens it.

    Args:
        seq_len: length of the prompt the decoder received.
        visual_start: first visual token's sequence position.
        visual_count: how many visual tokens entered.
        device: where the index tensors live.
    """

    def __init__(self, seq_len, visual_start, visual_count, device=None):
        if visual_start < 0 or visual_count < 0 or visual_start + visual_count > seq_len:
            raise ValueError(
                f'visual span [{visual_start}, {visual_start + visual_count}) '
                f'does not fit in a sequence of {seq_len}')
        self.device = device
        self.seq_len = int(seq_len)
        self.visual_start = int(visual_start)
        self.visual_count = int(visual_count)

        # Sequence position -> original position. Identity until the first prune.
        self.original_position = torch.arange(seq_len, device=device)
        # Which current positions are visual.
        self.is_visual = torch.zeros(seq_len, dtype=torch.bool, device=device)
        self.is_visual[visual_start:visual_start + visual_count] = True
        # Current visual tokens, as indices into the L_v that entered.
        self.alive_visual = torch.arange(visual_count, device=device)
        # Text tokens SparseVLM lets vote, as current sequence positions.
        self.rater_positions = None

    # -- views -------------------------------------------------------------
    @property
    def length(self):
        return int(self.original_position.numel())

    @property
    def visual_positions(self):
        """Current sequence positions of the visual tokens, ascending."""
        return torch.where(self.is_visual)[0]

    @property
    def num_visual(self):
        return int(self.is_visual.sum())

    def set_raters(self, positions):
        """Record which text positions vote. Visual positions are rejected."""
        positions = torch.as_tensor(positions, device=self.device).reshape(-1).long()
        if positions.numel() and bool(self.is_visual[positions].any()):
            raise ValueError('a rater must be a text token, not a visual one')
        self.rater_positions = positions

    # -- the prune ---------------------------------------------------------
    def keep(self, keep_visual):
        """Retain ``keep_visual`` of the live visual tokens; drop the rest.

        ``keep_visual`` indexes into ``visual_positions`` -- that is, into the
        tokens still alive, not into the original L_v -- which is what a topk
        over the current scores returns.

        Returns the sorted sequence indices to slice the decoder state with:
        every text token plus the retained visual ones.
        """
        visual = self.visual_positions
        keep_visual = torch.as_tensor(keep_visual, device=self.device).reshape(-1).long()
        if keep_visual.numel() and (keep_visual.min() < 0 or keep_visual.max() >= visual.numel()):
            raise ValueError('keep_visual indexes past the live visual tokens')

        keep_mask = torch.zeros(self.length, dtype=torch.bool, device=self.device)
        keep_mask[~self.is_visual] = True                 # text is never pruned
        keep_mask[visual[keep_visual]] = True
        keep_index = torch.where(keep_mask)[0]

        self.original_position = self.original_position[keep_index]
        self.is_visual = self.is_visual[keep_index]
        self.alive_visual = self.alive_visual[torch.sort(keep_visual).values]
        if self.rater_positions is not None:
            # Raters are text, so they all survive; their positions shift left.
            self.rater_positions = torch.searchsorted(keep_index, self.rater_positions)
        return keep_index

    # -- what the decoder needs after a prune -------------------------------
    def position_ids(self, full_position_ids):
        """``full_position_ids`` restricted to the surviving tokens.

        Sliced, never renumbered: a token keeps the position it had when its
        key was written to the cache, so RoPE stays consistent across the
        layers that were pruned and the ones that were not.
        """
        return full_position_ids[..., self.original_position]

    def attention_mask(self, full_mask):
        """A 4D mask restricted to the surviving rows and columns."""
        if full_mask is None:
            return None
        rows = self.original_position
        return full_mask[..., rows, :][..., rows]
