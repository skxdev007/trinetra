import json

# Load results
with open('benchmarking/videomme/long_video_coin/results/results_clip_20260305_195049.json') as f:
    data = json.load(f)

failures = [r for r in data['results'] if not r['correct']]
successes = [r for r in data['results'] if r['correct']]

print(f"Total questions: {len(data['results'])}")
print(f"Correct: {len(successes)} ({len(successes)/len(data['results'])*100:.1f}%)")
print(f"Failures: {len(failures)} ({len(failures)/len(data['results'])*100:.1f}%)")

# Analyze prediction bias
pred_a = sum(1 for r in data['results'] if r['predicted'] == 'A')
pred_b = sum(1 for r in data['results'] if r['predicted'] == 'B')
print(f"\nPrediction distribution:")
print(f"  Predicted A: {pred_a} ({pred_a/len(data['results'])*100:.1f}%)")
print(f"  Predicted B: {pred_b} ({pred_b/len(data['results'])*100:.1f}%)")

# Ground truth distribution
gt_a = sum(1 for r in data['results'] if r['ground_truth'] == 'A')
gt_b = sum(1 for r in data['results'] if r['ground_truth'] == 'B')
print(f"\nGround truth distribution:")
print(f"  GT A: {gt_a} ({gt_a/len(data['results'])*100:.1f}%)")
print(f"  GT B: {gt_b} ({gt_b/len(data['results'])*100:.1f}%)")

print("\n" + "="*80)
print("FAILURE ANALYSIS")
print("="*80)
print("\nSample failures (first 5):")
for i, f in enumerate(failures[:5], 1):
    print(f"\n{i}. Video: {f['video']}")
    print(f"   Predicted: {f['predicted']}, Ground Truth: {f['ground_truth']}")
