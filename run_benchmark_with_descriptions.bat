@echo off
REM Run benchmark with frame descriptions enabled
REM This should improve accuracy from 55%% to 75-80%%

echo ==================================
echo SHARINGAN Benchmark with Frame Descriptions
echo ==================================
echo.
echo Testing with SigLIP + SmolVLM descriptions
echo Expected improvement: 55%% to 75-80%% accuracy
echo.

python benchmarking/videomme/benchmark_long_video_coin.py --model siglip --max-questions 20 --enable-descriptions --target-fps 5.0

echo.
echo ==================================
echo Benchmark Complete!
echo ==================================
echo.
echo Check the results in: benchmarking/videomme/long_video_coin/results/
echo.
echo To compare with baseline (no descriptions), run:
echo   python benchmarking/videomme/benchmark_long_video_coin.py --model siglip --max-questions 20 --no-descriptions
