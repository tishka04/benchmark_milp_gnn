"""Run the frozen campaign with retryable atomic JSON writes on Windows."""
import json
from pathlib import Path
import tempfile
import time

from . import data


def resilient_put(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(data.HOME):
        raise ValueError('Revision outputs must remain in their own directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
        stream.write(text)
    try:
        for attempt in range(21):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt == 20:
                    raise
                time.sleep(0.5)
    finally:
        temp.unlink(missing_ok=True)


def main():
    # Install before importing consumers, keeping the frozen source files intact.
    data.put = resilient_put
    from .complete_campaign import main as run_campaign
    run_campaign()


if __name__ == '__main__':
    main()
