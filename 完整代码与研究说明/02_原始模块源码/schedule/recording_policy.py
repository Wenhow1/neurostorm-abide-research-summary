"""Recording policy for NEW upstream experiments; no model or data access."""
POLICY = {
    'version': '2026-10-22-every5-v1',
    'validation_interval': 1,
    'checkpoint_interval': 5,
    'fixed_gradient_probe_interval': 1,
    'last_interval': 1,
    'best_metric': 'validation_mask50.MSE',
    'best_direction': 'min',
    'tie_break': 'earlier_epoch',
}
def is_record_epoch(epoch, total_epochs):
    return epoch % 5 == 0 or epoch == total_epochs

def best_row(history):
    candidates = [r for r in history if 'validation_mask50' in r]
    return min(candidates, key=lambda r: (r['validation_mask50']['MSE'], r['epoch'])) if candidates else None
