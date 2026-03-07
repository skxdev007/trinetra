@echo off
echo ================================================================================
echo Running Benchmark with SigLIP-Base (Better than CLIP)
echo ================================================================================
echo.
echo Configuration:
echo   Model: SigLIP-Base (768D embeddings)
echo   Questions: 20
echo   Action Classification: Enabled
echo   Temporal Ordering: Explicit
echo.
python benchmarking/videomme/benchmark_long_video_coin.py --model siglip-base --max-questions 20 --no-descriptions
echo.
echo ================================================================================
echo Benchmark Complete!
echo ================================================================================
pause
