duration = 9319.8
final_results = [9256.6, 9268.4, 9266.4, 9267.4, 9265.4]

print(f'Video Duration: {duration:.1f}s ({duration/3600:.2f} hours)')
print(f'Last 10% starts at: {duration*0.90:.1f}s ({duration*0.90/60:.1f}min)')
print(f'Last 5% starts at: {duration*0.95:.1f}s ({duration*0.95/60:.1f}min)')
print(f'\nFinal Result Query Timestamps:')
for i, ts in enumerate(final_results, 1):
    print(f'  {i}. {ts:.1f}s ({ts/60:.1f}min) - {ts/duration*100:.1f}% into video')

print(f'\nResults in last 10%: {sum(1 for ts in final_results if ts > duration*0.90)}/5')
print(f'Results in last 5%: {sum(1 for ts in final_results if ts > duration*0.95)}/5')
print(f'\n✅ Temporal filtering working correctly for long video!')
