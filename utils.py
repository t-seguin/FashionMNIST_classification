import sys
import inspect
import contextlib
import tqdm


def progress(loss, acc, prefix: str = None):
    print(f"{prefix}Loss : {loss:2.4f}, Acc : {acc:2.4f}")


@contextlib.contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.tqdm.write(*args, **kwargs)
        except Exception:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


if __name__ == "__main__":
    import random
    import time

    for i in range(10):
        progress(random.random(), random.random())
        time.sleep(0.5)
    sys.stdout.write("\n")
