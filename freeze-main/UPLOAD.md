# Freeze package — main round

Upload these two files before **Saturday 9 PM Central**:

1. `submission.csv` — test forecasts (blend winner, validation **30.911 / 60**)
2. `entry.zip` — reproducible code (`entry-winner-blend.zip`)

## Model
Blend of:
- 25% `NeighborPooledRecent` (half_life=12, prior_strength=10, neighbor_weight=0.15, same_type_only)
- 75% tuned `pooled_recent` (half_life=16, prior_strength=10)

## Backup (if blend ZIP fails code check)
- `backup-pooled-recent-submission.csv`
- `backup-entry-pooled-recent.zip` (default starter `pooled_recent`, ~30.826 / 60)

## After upload
1. Wait for the receipt JSON
2. Confirm status becomes `passed` (or stays queued/running until checked)
3. Save the receipt in this folder as `receipt.json`

Team registration and the House Edge upload page / team token come from the participant guide — this freeze cannot upload for you without that token.
