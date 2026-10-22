@echo off
REM ============================================================
REM RWKE OpenAI 兼容服务器启动脚本（AIRI 落地入口）
REM 默认：RoleB × RWKV7-G1-1.5B，端口 8102
REM AIRI 配置：设置 → Providers → OpenAI Compatible
REM   Base URL : http://127.0.0.1:8102/v1
REM   API Key  : rwke（任意非空）
REM   Model    : rwkv7-g1-1.5b
REM ============================================================
chcp 65001 >nul

REM —— CUDA 算子编译工具链（首次已编好缓存，日常启动仅用于命中缓存校验）——
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
set "NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor"

cd /d "<workspace>/RWKE"

REM —— 默认：EmoCompanion（正式角色：日记 State + 角色 LoRA 自动挂载）+ 快速采样 ——
REM   （认证复现请去掉 --快速采样；--无适配器 可对比裸基座；RoleB为测试代号，--角色 RoleB --无适配器）
python 服务器.py --模型 RWKV7-G1-1.5B --角色 EmoCompanion --端口 8102 --快速采样

REM —— 可选：RoleB（测试代号，仅日记 State 无 LoRA，长聊更稳）——
REM python 服务器.py --模型 RWKV7-G1-1.5B --角色 RoleB --无适配器 --端口 8102 --快速采样

REM —— 可选：轻量快速（0.4B，弱机/低显存场景）——
REM python 服务器.py --模型 RWKV7-World-0.4B --角色 RoleB --端口 8102
