# Missing file

**Expected:** `012metadata_compositional-split-natural.t7`

**Status:** the drive backup copy of this cumulative-pairs metadata file
was truncated (BadZipFile, 262144 bytes — see also the earlier
power-of-two truncation described in the LFS setup commit). No intact
copy has been located yet.

**Expected schema** (from analogous intact files):
`torch.save(list[dict])` where each dict is
`{image: str, attr: str, obj: str, set: "train" | "val" | "test"}`,
containing all images whose (attr, obj) pair appears in the union of
`ut-zappos/session_{0,1,2}` train/val/test splits.

**To restore:** either locate an intact backup, or regenerate
deterministically from the per-session metadata using
`scripts/build_cumulative.py` (to be added in a follow-up commit).
