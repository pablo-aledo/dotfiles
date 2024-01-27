python3 download_weights.py
zig build -DOptimize=ReleaseFast
./zig-out/bin/zig_gpt2 "Marcus Aurelius said"
