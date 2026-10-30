# nanyang-Chat 打包脚本（Windows）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
# 产物：$env:LOCALAPPDATA\nanyang-Chat-build\dist\nanyang-Chat\nanyang-Chat.exe
# 说明：torch/PySide6 体积大，onedir 模式（文件夹版）启动快、成功率高；分发时整体压缩为 zip。
# J: 工作区对进程只读，构建输出一律放 %LOCALAPPDATA%。

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OUT = Join-Path $env:LOCALAPPDATA "nanyang-Chat-build"

pyinstaller --noconfirm --clean --onedir --windowed --name nanyang-Chat `
  --distpath "$OUT\dist" --workpath "$OUT\build" --specpath "$OUT" `
  --add-data "$ROOT\configs;configs" `
  --paths "$ROOT\..\nanyang-Helios1.2" `
  --hidden-import helios.models `
  --hidden-import helios.diffusion.flow `
  --collect-all imageio `
  --collect-all torch `
  "$ROOT\run.py"

Write-Host "打包完成: $OUT\dist\nanyang-Chat\nanyang-Chat.exe"
