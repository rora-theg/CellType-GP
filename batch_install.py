"""
  批量收集并分批安装 /home/vs_theg/ST_program 中缺失的第三方包。
  """
  
import json
import subprocess
from pathlib import Path
import re
import sys
from typing import List, Set, Dict

PROJECT_ROOT = Path("/home/vs_theg/ST_program")
BATCH_SIZE = 5  # 每批安装的包数量，可自行调整
PYTHON_EXEC = Path("~/anaconda3/bin/python").expanduser()  # 修改为目标环境的 Python

  # 简单的内置库列表，用于过滤（可根据需要扩充）
STDLIB = {
      "os", "sys", "pathlib", "json", "re", "math", "itertools",
      "functools", "collections", "subprocess", "typing", "statistics",
      "datetime", "glob", "argparse", "logging", "random", "csv", "copy",
      "gzip", "pickle"
  }

IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")

from typing import Set

def extract_from_py(path: Path) -> Set[str]:

      packages = set()
      for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
          match = IMPORT_RE.match(line)
          if not match:
              continue
          module = match.group(1) or match.group(2)
          root = module.split(".")[0]
          if root and root not in STDLIB:
              packages.add(root)
      return packages

def extract_from_ipynb(path: Path) -> Set[str]:

      packages = set()
      try:
          data = json.loads(path.read_text(encoding="utf-8"))
      except Exception as exc:
          print(f"[跳过] 无法解析 {path}: {exc}")
          return packages
      for cell in data.get("cells", []):
          if cell.get("cell_type") != "code":
              continue
          for line in cell.get("source", []):
              match = IMPORT_RE.match(line)
              if not match:
                  continue
              module = match.group(1) or match.group(2)
              root = module.split(".")[0]
              if root and root not in STDLIB:
                  packages.add(root)
      return packages

def gather_packages(root: Path) -> List[str]:
      pkgs = set()
      for path in root.rglob("*"):
          if path.suffix == ".py":
              pkgs |= extract_from_py(path)
          elif path.suffix == ".ipynb":
              pkgs |= extract_from_ipynb(path)
      return sorted(pkgs)

def install_in_batches(packages: List[str], batch_size: int = 5):
      for i in range(0, len(packages), batch_size):
          batch = packages[i : i + batch_size]
          print(f"\n=== 开始安装批次 {i//batch_size + 1} / {((len(packages)-1)//batch_size)+1} ===")
          print(" ".join(batch))
          cmd = [str(PYTHON_EXEC), "-m", "pip", "install", *batch]
          try:
              subprocess.run(cmd, check=True)
          except subprocess.CalledProcessError as exc:
              print(f"[警告] 批次安装失败: {exc}. 请手动处理后继续。")

def main():
      if not PROJECT_ROOT.exists():
          print(f"[错误] 项目路径不存在: {PROJECT_ROOT}")
          sys.exit(1)
      packages = gather_packages(PROJECT_ROOT)
      print(f"共检测到 {len(packages)} 个第三方包：")
      print(packages)
      install_in_batches(packages, BATCH_SIZE)

if __name__ == "__main__":
      main()
