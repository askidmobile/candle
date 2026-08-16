@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
cargo run --release --features real-model --bin qwen35moe_compare -- %*
