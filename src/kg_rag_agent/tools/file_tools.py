# -*- coding: utf-8 -*-
"""
file_tools.py

文件工具适配层。

职责：
    1. 提供安全、统一的文本 / JSON 读写能力。
    2. 提供目录扫描、批量文本加载、文件信息、复制、移动、删除等辅助能力。
    3. 为后续 Tool Calling、脚本调试、数据检查提供轻量工具入口。

注意：
    本文件属于 tools 层，只做文件工具适配。
    不实现 KG 查询、向量检索、实体链接、LangGraph 编排或 LLM 调用。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import ToolPathError, ToolValidationError


DEFAULT_ENCODING = "utf-8"


class FileTools:
    """安全文件工具类。"""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        *,
        encoding: str = DEFAULT_ENCODING,
        enforce_base_dir: bool = True,
        max_read_bytes: int = 10 * 1024 * 1024,
        max_write_bytes: int = 10 * 1024 * 1024,
        allow_hidden: bool = False,
        allow_delete: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else None
        self.encoding = str(encoding or DEFAULT_ENCODING)
        self.enforce_base_dir = bool(enforce_base_dir and self.base_dir is not None)
        self.max_read_bytes = int(max_read_bytes)
        self.max_write_bytes = int(max_write_bytes)
        self.allow_hidden = bool(allow_hidden)
        self.allow_delete = bool(allow_delete)

        if self.max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be > 0")
        if self.max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be > 0")

    # =====================================================
    # 1. 路径处理
    # =====================================================

    def resolve_path(self, path: str | Path) -> Path:
        """解析并校验路径；配置 base_dir 时禁止越过沙箱边界。"""

        raw_value = str(path or "").strip()
        if not raw_value:
            raise ToolValidationError("path must not be empty")

        raw_path = Path(raw_value).expanduser()
        if raw_path.is_absolute():
            resolved = raw_path.resolve(strict=False)
        elif self.base_dir is not None:
            resolved = (self.base_dir / raw_path).resolve(strict=False)
        else:
            resolved = raw_path.resolve(strict=False)

        if self.enforce_base_dir and self.base_dir is not None:
            try:
                resolved.relative_to(self.base_dir)
            except ValueError as exc:
                raise ToolPathError(
                    "Path is outside the configured base directory.",
                    details={
                        "path": resolved.as_posix(),
                        "base_dir": self.base_dir.as_posix(),
                    },
                ) from exc

        if not self.allow_hidden and self.base_dir is not None:
            try:
                relative = resolved.relative_to(self.base_dir)
            except ValueError:
                relative = resolved
            if any(part.startswith(".") and part not in {".", ".."} for part in relative.parts):
                raise ToolPathError(
                    "Hidden paths are not allowed.",
                    details={"path": resolved.as_posix()},
                )

        return resolved

    def relative_path(self, path: str | Path) -> str:
        """返回相对于 base_dir 的路径；无法相对化时返回文件名或绝对路径。"""

        resolved = self.resolve_path(path)

        if self.base_dir is None:
            return resolved.name

        try:
            return resolved.relative_to(self.base_dir).as_posix()
        except ValueError:
            return resolved.as_posix()

    def display_path(self, path: str | Path) -> str:
        """
        返回统一使用 / 的路径字符串。

        内部文件操作仍然使用 pathlib.Path，
        这里只影响返回给测试、日志、JSON 的字符串展示形式。
        """

        return self.resolve_path(path).as_posix()

    def ensure_dir(self, directory: str | Path) -> Dict[str, Any]:
        """确保目录存在，并返回统一结果。"""

        dir_path = self.resolve_path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        return {
            "ok": True,
            "path": self.display_path(dir_path),
            "relative_path": self.relative_path(dir_path),
            "is_dir": True,
        }

    # =====================================================
    # 2. 文本读写
    # =====================================================

    def read_text(
        self,
        file_path: str | Path,
        *,
        encoding: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """读取文本文件。"""

        path = self.resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        size_bytes = path.stat().st_size
        if size_bytes > self.max_read_bytes:
            raise ToolValidationError(
                "File exceeds max_read_bytes.",
                details={
                    "path": path.as_posix(),
                    "size_bytes": size_bytes,
                    "max_read_bytes": self.max_read_bytes,
                },
            )

        text = path.read_text(
            encoding=encoding or self.encoding,
            errors="ignore",
        )

        if max_chars is not None and max_chars >= 0:
            return text[:max_chars]

        return text

    def write_text(
        self,
        file_path: str | Path,
        content: str,
        *,
        encoding: Optional[str] = None,
        create_parent: bool = True,
        append: bool = False,
    ) -> Dict[str, Any]:
        """写入文本文件；append=True 时追加。"""

        path = self.resolve_path(file_path)

        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        text = str(content)
        mode = "a" if append else "w"
        used_encoding = encoding or self.encoding
        payload_bytes = len(text.encode(used_encoding, errors="ignore"))
        existing_bytes = path.stat().st_size if append and path.exists() else 0
        if existing_bytes + payload_bytes > self.max_write_bytes:
            raise ToolValidationError(
                "Write exceeds max_write_bytes.",
                details={
                    "path": path.as_posix(),
                    "payload_bytes": payload_bytes,
                    "existing_bytes": existing_bytes,
                    "max_write_bytes": self.max_write_bytes,
                },
            )

        with path.open(mode, encoding=used_encoding) as f:
            f.write(text)

        return {
            "ok": True,
            "path": self.display_path(path),
            "relative_path": self.relative_path(path),
            "bytes": payload_bytes,
            "num_chars": len(text),
            "append": append,
        }

    def append_text(
        self,
        file_path: str | Path,
        content: str,
        *,
        encoding: Optional[str] = None,
        create_parent: bool = True,
    ) -> Dict[str, Any]:
        """追加写入文本文件。"""

        return self.write_text(
            file_path,
            content,
            encoding=encoding,
            create_parent=create_parent,
            append=True,
        )

    def read_lines(
        self,
        file_path: str | Path,
        *,
        encoding: Optional[str] = None,
        strip: bool = True,
        skip_empty: bool = False,
    ) -> List[str]:
        """按行读取文本文件。"""

        text = self.read_text(file_path, encoding=encoding)
        lines = text.splitlines()

        if strip:
            lines = [line.strip() for line in lines]

        if skip_empty:
            lines = [line for line in lines if line]

        return lines

    def write_lines(
        self,
        file_path: str | Path,
        lines: Sequence[str],
        *,
        encoding: Optional[str] = None,
        create_parent: bool = True,
        append: bool = False,
    ) -> Dict[str, Any]:
        """按行写入文本文件。"""

        content = "\n".join(str(line) for line in lines)
        return self.write_text(
            file_path,
            content,
            encoding=encoding,
            create_parent=create_parent,
            append=append,
        )

    # =====================================================
    # 3. JSON 读写
    # =====================================================

    def read_json(
        self,
        file_path: str | Path,
        *,
        encoding: Optional[str] = None,
    ) -> Any:
        """读取 JSON 文件。"""

        text = self.read_text(file_path, encoding=encoding)
        return json.loads(text)

    def write_json(
        self,
        file_path: str | Path,
        data: Any,
        *,
        encoding: Optional[str] = None,
        ensure_ascii: bool = False,
        indent: int = 2,
        create_parent: bool = True,
    ) -> Dict[str, Any]:
        """写入 JSON 文件。"""

        content = json.dumps(
            data,
            ensure_ascii=ensure_ascii,
            indent=indent,
        )

        result = self.write_text(
            file_path,
            content,
            encoding=encoding,
            create_parent=create_parent,
        )
        result["data_type"] = type(data).__name__
        return result

    # =====================================================
    # 4. 文件信息
    # =====================================================

    def exists(self, path: str | Path) -> bool:
        """判断路径是否存在。"""

        return self.resolve_path(path).exists()

    def file_info(
        self,
        file_path: str | Path,
        *,
        include_hash: bool = False,
    ) -> Dict[str, Any]:
        """获取文件基础信息。"""

        path = self.resolve_path(file_path)

        info: Dict[str, Any] = {
            "path": self.display_path(path),
            "relative_path": self.relative_path(path),
            "name": path.name,
            "suffix": path.suffix,
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }

        if not path.exists():
            return info

        stat = path.stat()
        info.update(
            {
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "created_time": stat.st_ctime,
            }
        )

        if include_hash and path.is_file():
            info["sha256"] = self.sha256(path)

        return info

    def sha256(
        self,
        file_path: str | Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """计算文件 SHA256。"""

        path = self.resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        digest = hashlib.sha256()

        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)

        return digest.hexdigest()

    # =====================================================
    # 5. 目录扫描 / 批量文本加载
    # =====================================================

    def list_files(
        self,
        directory: str | Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        include_dirs: bool = False,
        max_files: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """扫描目录文件。limit 是 max_files 的兼容别名。"""

        dir_path = self.resolve_path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        effective_limit = limit if limit is not None else max_files
        iterator = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)

        results: List[Dict[str, Any]] = []

        for item in sorted(iterator, key=lambda p: p.as_posix()):
            if item.is_dir() and not include_dirs:
                continue

            results.append(self.file_info(item, include_hash=False))

            if effective_limit is not None and len(results) >= effective_limit:
                break

        return results

    def load_texts(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.txt",
        recursive: bool = False,
        limit: Optional[int] = None,
        max_files: Optional[int] = None,
        max_chars_per_file: Optional[int] = None,
        encoding: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按 pattern 批量加载文本文件。"""

        files = self.list_files(
            directory,
            pattern=pattern,
            recursive=recursive,
            limit=limit if limit is not None else max_files,
        )

        documents: List[Dict[str, Any]] = []

        for info in files:
            text = self.read_text(
                info["path"],
                encoding=encoding,
                max_chars=max_chars_per_file,
            )

            documents.append(
                {
                    "path": info["path"],
                    "relative_path": info.get("relative_path", ""),
                    "name": info.get("name", ""),
                    "suffix": info.get("suffix", ""),
                    "text": text,
                    "content": text,
                    "metadata": {
                        "size_bytes": info.get("size_bytes", 0),
                    },
                }
            )

        return documents

    def load_texts_from_directory(
        self,
        directory: str | Path,
        *,
        extensions: Sequence[str] = (".txt", ".md", ".json"),
        recursive: bool = False,
        max_files: Optional[int] = None,
        max_chars_per_file: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按扩展名批量读取文本文件，兼容旧接口。"""

        dir_path = self.resolve_path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        normalized_exts = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }

        iterator = dir_path.rglob("*") if recursive else dir_path.glob("*")
        documents: List[Dict[str, Any]] = []

        for item in sorted(iterator, key=lambda p: p.as_posix()):
            if not item.is_file():
                continue

            if item.suffix.lower() not in normalized_exts:
                continue

            text = self.read_text(item, max_chars=max_chars_per_file)

            documents.append(
                {
                    "path": self.display_path(item),
                    "relative_path": self.relative_path(item),
                    "name": item.name,
                    "suffix": item.suffix,
                    "text": text,
                    "content": text,
                    "metadata": {
                        "size_bytes": item.stat().st_size,
                    },
                }
            )

            if max_files is not None and len(documents) >= max_files:
                break

        return documents

    # =====================================================
    # 6. 文件复制、移动、删除
    # =====================================================

    def copy_file(
        self,
        source: str | Path,
        target: str | Path,
        *,
        create_parent: bool = True,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """复制文件。"""

        source_path = self.resolve_path(source)
        target_path = self.resolve_path(target)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if not source_path.is_file():
            raise IsADirectoryError(f"Source is not a file: {source_path}")

        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Target file already exists: {target_path}")

        if create_parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, target_path)

        return {
            "ok": True,
            "source": self.display_path(source_path),
            "target": self.display_path(target_path),
            "path": self.display_path(target_path),
            "relative_path": self.relative_path(target_path),
            "bytes": target_path.stat().st_size,
        }

    def move_file(
        self,
        source: str | Path,
        target: str | Path,
        *,
        create_parent: bool = True,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """移动文件。"""

        source_path = self.resolve_path(source)
        target_path = self.resolve_path(target)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if not source_path.is_file():
            raise IsADirectoryError(f"Source is not a file: {source_path}")

        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Target file already exists: {target_path}")

        if create_parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(source_path), str(target_path))

        return {
            "ok": True,
            "source": self.display_path(source_path),
            "target": self.display_path(target_path),
            "path": self.display_path(target_path),
            "relative_path": self.relative_path(target_path),
            "bytes": target_path.stat().st_size,
        }

    def delete_file(
        self,
        file_path: str | Path,
        *,
        missing_ok: bool = True,
    ) -> Dict[str, Any]:
        """删除文件；Registry 层仍需单独授予 DELETE 权限。"""

        path = self.resolve_path(file_path)

        if not self.allow_delete:
            raise ToolPathError(
                "Delete operation is disabled for this FileTools instance.",
                details={"path": path.as_posix()},
            )

        if not path.exists():
            if missing_ok:
                return {
                    "ok": False,
                    "deleted": False,
                    "path": self.display_path(path),
                    "relative_path": self.relative_path(path),
                    "reason": "missing",
                }
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        path.unlink()

        return {
            "ok": True,
            "deleted": True,
            "path": self.display_path(path),
            "relative_path": self.relative_path(path),
        }


# =========================================================
# 2. 默认工具实例
# =========================================================

_DEFAULT_FILE_TOOLS: Optional[FileTools] = None


def get_default_file_tools(
    *,
    base_dir: Optional[str | Path] = None,
    refresh: bool = False,
) -> FileTools:
    """获取默认 FileTools 实例。"""

    global _DEFAULT_FILE_TOOLS

    requested_base = Path(base_dir).expanduser().resolve() if base_dir else None
    if (
        refresh
        or _DEFAULT_FILE_TOOLS is None
        or _DEFAULT_FILE_TOOLS.base_dir != requested_base
    ):
        _DEFAULT_FILE_TOOLS = FileTools(base_dir=base_dir)

    return _DEFAULT_FILE_TOOLS


# =========================================================
# 3. 函数式工具入口
# =========================================================

def _new_tools(base_dir: Optional[str | Path] = None) -> FileTools:
    """函数式入口使用独立实例，避免测试之间的默认 base_dir 污染。"""

    return FileTools(base_dir=base_dir)


def read_text_tool(
    file_path: Optional[str | Path] = None,
    *,
    path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
    max_chars: Optional[int] = None,
) -> str:
    """读取文本工具函数。"""

    target = path if path is not None else file_path
    if target is None:
        raise ValueError("path is required")

    return _new_tools(base_dir).read_text(target, max_chars=max_chars)


def write_text_tool(
    file_path: Optional[str | Path] = None,
    content: Optional[str] = None,
    *,
    path: Optional[str | Path] = None,
    text: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    append: bool = False,
) -> Dict[str, Any]:
    """写入文本工具函数。"""

    target = path if path is not None else file_path
    if target is None:
        raise ValueError("path is required")

    body = text if text is not None else content
    if body is None:
        body = ""

    return _new_tools(base_dir).write_text(target, str(body), append=append)


def read_json_tool(
    file_path: Optional[str | Path] = None,
    *,
    path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
) -> Any:
    """读取 JSON 工具函数。"""

    target = path if path is not None else file_path
    if target is None:
        raise ValueError("path is required")

    return _new_tools(base_dir).read_json(target)


def write_json_tool(
    file_path: Optional[str | Path] = None,
    data: Any = None,
    *,
    path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """写入 JSON 工具函数。"""

    target = path if path is not None else file_path
    if target is None:
        raise ValueError("path is required")

    return _new_tools(base_dir).write_json(target, data)


def list_files_tool(
    directory: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    base_dir: Optional[str | Path] = None,
    max_files: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """目录扫描工具函数。"""

    return _new_tools(base_dir).list_files(
        directory,
        pattern=pattern,
        recursive=recursive,
        max_files=max_files,
        limit=limit,
    )


def file_info_tool(
    file_path: Optional[str | Path] = None,
    *,
    path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
    include_hash: bool = False,
) -> Dict[str, Any]:
    """文件信息工具函数。"""

    target = path if path is not None else file_path
    if target is None:
        raise ValueError("path is required")

    return _new_tools(base_dir).file_info(target, include_hash=include_hash)


def load_texts_tool(
    directory: str | Path,
    *,
    pattern: str = "*.txt",
    recursive: bool = False,
    base_dir: Optional[str | Path] = None,
    max_files: Optional[int] = None,
    limit: Optional[int] = None,
    max_chars_per_file: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """批量加载文本工具函数。"""

    return _new_tools(base_dir).load_texts(
        directory,
        pattern=pattern,
        recursive=recursive,
        max_files=max_files,
        limit=limit,
        max_chars_per_file=max_chars_per_file,
    )


__all__ = [
    "FileTools",
    "get_default_file_tools",
    "read_text_tool",
    "write_text_tool",
    "read_json_tool",
    "write_json_tool",
    "list_files_tool",
    "file_info_tool",
    "load_texts_tool",
]