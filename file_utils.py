
from pathlib import Path

def get_latest_file(dir_path, *, excludes:list|None=None, extensions:list|None=None, limit=1):
    """
    extensions=[".rttm"], must start with '.'
    """
    p = Path(dir_path)

    if extensions:
        # assert all(ext.startswith('.') for ext in extensions)
        # Build a generator that finds all files matching the extensions
        files = (f for ext in extensions
                 for f in p.rglob(f'*{ext}*'.replace('**', '*'))
                 if f.is_file())
    else:
        files = (f for f in p.rglob('*') if f.is_file())

    excludes = excludes or []
    files = (
        f for f in files
        if not any(exclude in str(f) for exclude in excludes)
    )

    if limit == 1:
        return str(max(files, key=lambda f: f.stat().st_mtime))

    limit = limit if limit > 0 else 999999
    sorted_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
    return sorted_files[:limit]
    return sorted([str(f) for f in sorted_files[:limit]])

