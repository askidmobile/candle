@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;%LIB%"
rem Общий target с master-репо — переиспользуем flash-attn (46мин) и candle-* кэш.
set "CARGO_TARGET_DIR=D:\Projects\yttri-build\candle-fork-qwen35-batch\target"
cd /d D:\Projects\yttri-build\candle-fork-vramfit
cargo build --release --features real-model,cuda --bin bench_candle_direct
