# -*- coding: utf-8 -*-
"""
embedding.py

Embedding 统一封装模块。

作用：
    1. 统一加载 sentence-transformers embedding 模型。
    2. 提供单文本 / 多文本向量化接口。
    3. 支持向量归一化。
    4. 支持模型缓存，避免重复加载。
    5. 为 entity_vector_store.py、vector_retriever.py、hybrid_retriever.py 提供底层能力。

本文件属于 retrieval 层：
    retrieval/
        embedding.py

它不负责：
    1. Chroma 向量库读写。
    2. 图结构查询。
    3. 实体最终落地。
    4. LangGraph 节点调度。
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ._validation import positive_int, require_non_empty_text
from .errors import RetrievalDependencyError


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 32


# =========================================================
# 2. 全局缓存
# =========================================================

_MODEL_CACHE: Dict[str, Any] = {}
_EMBEDDER_CACHE: Dict[str, "EmbeddingClient"] = {}


# =========================================================
# 3. 项目路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/retrieval/embedding.py

    parents:
        0 -> retrieval/
        1 -> kg_rag_agent/
        2 -> src/
        3 -> 项目根目录
    """

    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """
    解析项目路径。

    支持：
        1. 绝对路径
        2. 相对于当前工作目录
        3. 相对于项目根目录
    """

    input_path = Path(path)

    if input_path.is_absolute():
        return input_path

    cwd_path = Path.cwd() / input_path

    if cwd_path.exists():
        return cwd_path

    return get_project_root() / input_path


# =========================================================
# 4. EmbeddingClient
# =========================================================

class EmbeddingClient:
    """
    Embedding 模型统一客户端。

    示例：
        embedder = EmbeddingClient()
        vector = embedder.embed_query("阿尔茨海默病")
        vectors = embedder.embed_documents(["A", "B", "C"])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_files_only: bool = True,
        lazy_load: bool = True,
        allow_hash_fallback: bool = False,
        hash_embedding_dim: int = 384,
    ) -> None:
        """
        Args:
            model_name:
                sentence-transformers 模型名称或本地路径。

            device:
                模型运行设备，例如 "cpu"、"cuda"。
                None 表示由 sentence-transformers 自动决定。

            normalize_embeddings:
                是否对向量做 L2 归一化。

            batch_size:
                批量编码大小。

            local_files_only:
                是否只使用本地模型。
                对服务器环境更安全，避免运行时联网下载。

            lazy_load:
                是否延迟加载模型。
                True 表示第一次真正 encode 时才加载。

            allow_hash_fallback:
                当 sentence-transformers 不可用时，是否允许使用 hash embedding 兜底。
                生产环境建议 False。
                单元测试或无模型环境可设为 True。

            hash_embedding_dim:
                hash fallback 的向量维度。
        """

        self.model_name = require_non_empty_text(model_name, field="model_name")
        self.device = str(device).strip() if device is not None else None
        self.normalize_embeddings = bool(normalize_embeddings)
        self.batch_size = positive_int(batch_size, field="batch_size")
        self.local_files_only = bool(local_files_only)
        self.lazy_load = bool(lazy_load)
        self.allow_hash_fallback = bool(allow_hash_fallback)
        self.hash_embedding_dim = positive_int(hash_embedding_dim, field="hash_embedding_dim")

        self.model: Optional[Any] = None
        self.backend: str = "sentence_transformer"

        if not self.lazy_load:
            self._ensure_model_loaded()

    # =====================================================
    # 4.1 对外主接口
    # =====================================================

    def embed_query(self, text: str) -> List[float]:
        """
        编码单条查询文本。
        """

        vectors = self.embed_documents([text])

        if not vectors:
            return []

        return vectors[0]

    def embed_document(self, text: str) -> List[float]:
        """
        编码单条文档文本。
        """

        return self.embed_query(text)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        编码多条文本。

        Args:
            texts:
                文本列表。

        Returns:
            List[List[float]]:
                向量列表。
        """

        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [
            normalize_text_for_embedding(text)
            for text in texts
        ]

        if not cleaned_texts:
            return []

        if self.allow_hash_fallback:
            try:
                return self._embed_with_sentence_transformer(cleaned_texts)
            except Exception:
                self.backend = "hash_fallback"
                return [
                    build_hash_embedding(
                        text,
                        dim=self.hash_embedding_dim,
                        normalize=self.normalize_embeddings,
                    )
                    for text in cleaned_texts
                ]

        return self._embed_with_sentence_transformer(cleaned_texts)

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        normalize_embeddings: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        兼容 sentence-transformers 风格的 encode 接口。

        Args:
            texts:
                单文本或文本列表。

            normalize_embeddings:
                是否归一化。
                如果为 None，则使用 self.normalize_embeddings。

            batch_size:
                批处理大小。
                如果为 None，则使用 self.batch_size。
        """

        old_normalize = self.normalize_embeddings
        old_batch_size = self.batch_size

        if normalize_embeddings is not None:
            self.normalize_embeddings = bool(normalize_embeddings)

        if batch_size is not None:
            self.batch_size = positive_int(batch_size, field="batch_size")

        try:
            if isinstance(texts, str):
                return self.embed_query(texts)

            return self.embed_documents(list(texts))

        finally:
            self.normalize_embeddings = old_normalize
            self.batch_size = old_batch_size

    # =====================================================
    # 4.2 模型加载
    # =====================================================

    def _ensure_model_loaded(self) -> None:
        """
        确保 sentence-transformers 模型已加载。
        """

        if self.model is not None:
            return

        cache_key = self._model_cache_key()

        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalDependencyError(
                "sentence-transformers is not installed. "
                "Please install sentence-transformers first."
            ) from exc

        model_kwargs: Dict[str, Any] = {}

        if self.device:
            model_kwargs["device"] = self.device

        try:
            model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
                **model_kwargs,
            )
        except TypeError:
            model = SentenceTransformer(
                self.model_name,
                **model_kwargs,
            )

        self.model = model
        _MODEL_CACHE[cache_key] = model

    def _model_cache_key(self) -> str:
        """
        构造模型缓存 key。
        """

        return f"{self.model_name}::{self.device or 'auto'}::{self.local_files_only}"

    # =====================================================
    # 4.3 sentence-transformers 编码
    # =====================================================

    def _embed_with_sentence_transformer(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        使用 sentence-transformers 编码。
        """

        self._ensure_model_loaded()

        if self.model is None:
            return []

        all_vectors: List[List[float]] = []

        for batch in batch_iter(texts, self.batch_size):
            vectors = self.model.encode(
                batch,
                batch_size=len(batch),
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )

            vectors = convert_vectors_to_list(vectors)
            all_vectors.extend(vectors)

        return all_vectors

    # =====================================================
    # 4.4 信息
    # =====================================================

    def info(self) -> Dict[str, Any]:
        """
        返回当前 embedder 信息。
        """

        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "local_files_only": self.local_files_only,
            "lazy_load": self.lazy_load,
            "allow_hash_fallback": self.allow_hash_fallback,
            "backend": self.backend,
            "model_loaded": self.model is not None,
        }


    def health_check(self) -> Dict[str, Any]:
        """返回不触发模型加载的健康状态。"""
        return {"ok": True, "backend": self.backend, "model_loaded": self.model is not None, "model_name": self.model_name}

    def close(self) -> None:
        """释放当前实例持有的模型引用。"""
        self.model = None

# =========================================================
# 5. 全局便捷函数
# =========================================================

def get_default_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
) -> EmbeddingClient:
    """
    获取默认 EmbeddingClient。

    避免重复加载模型。
    """

    cache_key = (
        f"{model_name}::"
        f"{device or 'auto'}::"
        f"{normalize_embeddings}::"
        f"{local_files_only}::"
        f"{allow_hash_fallback}"
    )

    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    embedder = EmbeddingClient(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
    )

    _EMBEDDER_CACHE[cache_key] = embedder

    return embedder


def embed_query(
    text: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
) -> List[float]:
    """
    函数式接口：编码单条查询文本。
    """

    embedder = get_default_embedder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
    )

    return embedder.embed_query(text)


def embed_documents(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
) -> List[List[float]]:
    """
    函数式接口：编码多条文本。
    """

    embedder = get_default_embedder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
    )

    return embedder.embed_documents(texts)


def clear_embedding_cache() -> None:
    """
    清空 embedding 模型缓存。
    """

    _MODEL_CACHE.clear()
    _EMBEDDER_CACHE.clear()


# =========================================================
# 6. Hash fallback
# =========================================================

def build_hash_embedding(
    text: str,
    *,
    dim: int = 384,
    normalize: bool = True,
) -> List[float]:
    """
    构造简单 hash embedding。

    注意：
        这是兜底方案，不建议生产环境使用。
        主要用于：
            1. 单元测试
            2. 没有本地 embedding 模型时让流程可跑通

    原理：
        对 token 做 hash，把值累加到固定维度向量中。
    """

    dim = max(int(dim), 8)
    vector = [0.0 for _ in range(dim)]

    tokens = simple_tokenize(text)

    if not tokens:
        tokens = [str(text)]

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    if normalize:
        vector = l2_normalize(vector)

    return vector


# =========================================================
# 7. 向量工具
# =========================================================

def convert_vectors_to_list(vectors: Any) -> List[List[float]]:
    """
    将 numpy / torch / list 向量统一转成 List[List[float]]。
    """

    if vectors is None:
        return []

    if hasattr(vectors, "detach"):
        try:
            vectors = vectors.detach().cpu().numpy()
        except Exception:
            pass

    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()

    if not isinstance(vectors, list):
        return []

    if not vectors:
        return []

    first = vectors[0]

    # 单条向量：[0.1, 0.2, ...]
    if isinstance(first, (int, float)):
        return [[float(x) for x in vectors]]

    # 多条向量：[[...], [...]]
    results: List[List[float]] = []

    for vector in vectors:
        if not isinstance(vector, list):
            continue

        results.append(
            [
                safe_float(value, default=0.0)
                for value in vector
            ]
        )

    return results


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """
    L2 归一化。
    """

    values = [
        safe_float(value, default=0.0)
        for value in vector
    ]

    norm = math.sqrt(sum(value * value for value in values))

    if norm <= 0:
        return values

    return [
        value / norm
        for value in values
    ]


def cosine_similarity(
    vec1: Sequence[float],
    vec2: Sequence[float],
) -> float:
    """
    计算余弦相似度。
    """

    if not vec1 or not vec2:
        return 0.0

    length = min(len(vec1), len(vec2))

    if length <= 0:
        return 0.0

    a = [
        safe_float(vec1[i], default=0.0)
        for i in range(length)
    ]
    b = [
        safe_float(vec2[i], default=0.0)
        for i in range(length)
    ]

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a <= 0 or norm_b <= 0:
        return 0.0

    return dot / (norm_a * norm_b)


def dot_product(
    vec1: Sequence[float],
    vec2: Sequence[float],
) -> float:
    """
    点积。
    """

    if not vec1 or not vec2:
        return 0.0

    length = min(len(vec1), len(vec2))

    return sum(
        safe_float(vec1[i], default=0.0) * safe_float(vec2[i], default=0.0)
        for i in range(length)
    )


# =========================================================
# 8. 文本工具
# =========================================================

def normalize_text_for_embedding(text: Any) -> str:
    """
    embedding 前文本清洗。

    不做过度清洗，避免破坏实体名。
    """

    if text is None:
        return ""

    value = str(text).strip()
    value = re.sub(r"\s+", " ", value)

    return value


def simple_tokenize(text: Any) -> List[str]:
    """
    简单 token 切分。

    用于 hash fallback。
    """

    text = normalize_text_for_embedding(text).lower()

    if not text:
        return []

    # 英文 / 数字 token
    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)

    # 中文字符按连续片段保留
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]+", text)

    tokens = english_tokens + chinese_tokens

    if tokens:
        return tokens

    return list(text)


# =========================================================
# 9. 批处理工具
# =========================================================

def batch_iter(
    items: Sequence[str],
    batch_size: int,
) -> Iterable[List[str]]:
    """
    按 batch_size 切分列表。
    """

    batch_size = max(int(batch_size), 1)

    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转 float。
    """

    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


# =========================================================
# 10. 快速测试入口
# =========================================================

if __name__ == "__main__":
    embedder = EmbeddingClient(
        allow_hash_fallback=True,
        local_files_only=True,
    )

    texts = [
        "阿尔茨海默病",
        "Alzheimer's disease",
        "Barack Obama",
    ]

    vectors = embedder.embed_documents(texts)

    print("Embedder info:")
    print(embedder.info())

    print("\nVector shapes:")
    for text, vector in zip(texts, vectors):
        print(text, len(vector), vector[:5])