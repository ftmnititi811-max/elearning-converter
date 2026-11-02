import os
import tempfile
from pathlib import Path


class OutputError(Exception):
    pass


def write_atomic(path: Path, content: str, overwrite: bool) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", suffix=".tmp", prefix=".elearning-converter-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # On Windows rename fails if destination exists, so creation stays atomic.
            os.rename(temporary, path)
    except OSError as exc:
        raise OutputError(f"出力を書き込めません: {path.name}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
