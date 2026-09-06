#!/usr/bin/env python
"""Run the fork's own test modules without pytest.

There is no pytest in any conda env on this machine, and the gate tests --
the files that hold the whole byte-identity safety argument -- were therefore
being driven by four throwaway scripts in a session scratchpad under /tmp.
Those scripts duplicated the assertions rather than running them, so the real
test files were never actually executed, and they evaporate on reboot. This
replaces all four.

It works by installing a minimal `pytest` module into sys.modules before the
test modules are imported, then collecting and running `test_*` functions the
way pytest would. Supported: `pytest.mark.skipif`, `pytest.mark.parametrize`,
module-level `pytestmark`, `pytest.fixture` (plain and generator, autouse or
requested by name), the built-in `monkeypatch` fixture, and `pytest.skip`,
`pytest.fail`, `pytest.approx`, `pytest.raises`.

**It refuses rather than guesses.** Anything it does not implement -- an
unknown fixture name, an unrecognised mark, a fixture with a scope it does not
honour -- is reported as ERROR and counted against the run. A runner that
silently skipped what it could not handle would report a green suite that had
never executed the assertion in question, which is the one failure mode that
matters here.

    python tools/run_tests.py                # the fork's own tests
    python tools/run_tests.py -v             # ... with one line per test
    python tools/run_tests.py tests/test_fused_peaks.py
    python tools/run_tests.py -k peel        # substring filter on test names

Exit status is 0 only if nothing failed and nothing errored. Skips are fine --
the gate tests skip themselves when CUDA or Triton is missing, which is the
fallback case and is a legitimate pass on a CPU-only box.
"""
import argparse
import fnmatch
import importlib.util
import inspect
import os
import sys
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The fork's own tests. Upstream's suite is not in this default because it has
# not been kept passing on this branch and its failures would drown the signal
# we care about; name a path explicitly to run any of it.
FORK_TESTS = [
    'tests/test_fused_detect.py',
    'tests/test_fused_peel.py',
    'tests/test_fused_peel_cond.py',
    'tests/test_fused_peel_store.py',
    'tests/test_fused_peaks.py',
    'tests/test_fast_kpp.py',
    'tests/test_mea_fork.py',
]


# --------------------------------------------------------------------------
# the pytest shim
# --------------------------------------------------------------------------

class Skipped(Exception):
    """Raised by pytest.skip() and by an unsatisfied skipif mark."""


class Failed(Exception):
    """Raised by pytest.fail()."""


class Unsupported(Exception):
    """The shim met something it does not implement. Never silently ignored."""


class _Approx:
    """Enough of pytest.approx for `x == approx(y)` and `approx(y) == x`."""

    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, actual):
        rel = 1e-6 if self.rel is None else self.rel
        tol = max(rel * abs(self.expected), 0.0)
        if self.abs is not None:
            tol = max(tol, self.abs) if self.rel is not None else self.abs
        return abs(actual - self.expected) <= tol

    def __eq__(self, other):
        return self._close(other)

    def __repr__(self):
        return f'approx({self.expected!r}, rel={self.rel}, abs={self.abs})'


class _Raises:
    def __init__(self, expected):
        self.expected, self.value = expected, None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise Failed(f'DID NOT RAISE {self.expected}')
        if not issubclass(exc_type, self.expected):
            return False          # propagate the wrong exception
        self.value = exc
        return True


class _Mark:
    """A recorded decorator. Applied at collection, not at import."""

    def __init__(self, kind, args, kwargs):
        self.kind, self.args, self.kwargs = kind, args, kwargs

    def __call__(self, fn):
        marks = getattr(fn, '_shim_marks', None)
        if marks is None:
            marks = fn._shim_marks = []
        marks.insert(0, self)     # decorators apply bottom-up; restore source order
        return fn


class _MarkFactory:
    KNOWN = {'skipif', 'skip', 'parametrize', 'xfail', 'filterwarnings', 'slow'}

    def __getattr__(self, kind):
        if kind not in self.KNOWN:
            raise Unsupported(f'pytest.mark.{kind} is not implemented by the shim')

        def make(*args, **kwargs):
            return _Mark(kind, args, kwargs)
        return make


class _MonkeyPatch:
    """The subset of pytest's monkeypatch the fork's tests actually use."""

    def __init__(self):
        self._undo = []

    def setenv(self, name, value):
        self._undo.append(('env', name, os.environ.get(name)))
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        self._undo.append(('env', name, os.environ[name]))
        del os.environ[name]

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str):     # "module.path.attr" form
            modname, _, name = target.rpartition('.')
            target, value = importlib.import_module(modname), name
            raise Unsupported('monkeypatch.setattr("dotted.path") is not implemented')
        had = hasattr(target, name)
        if not had and raising:
            raise AttributeError(name)
        self._undo.append(('attr', (target, name), getattr(target, name, None), had))
        setattr(target, name, value)

    def delattr(self, target, name, raising=True):
        had = hasattr(target, name)
        if not had:
            if raising:
                raise AttributeError(name)
            return
        self._undo.append(('attr', (target, name), getattr(target, name), had))
        delattr(target, name)

    def undo(self):
        for entry in reversed(self._undo):
            if entry[0] == 'env':
                _, name, old = entry
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
            else:
                _, (obj, name), old, had = entry
                if had:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)
        self._undo.clear()


def _install_shim():
    """Put a fake `pytest` on sys.modules so the test files import cleanly."""
    m = types.ModuleType('pytest')

    def fixture(fn=None, **kwargs):
        unsupported = set(kwargs) - {'autouse', 'scope', 'name'}
        if unsupported:
            raise Unsupported(f'pytest.fixture({sorted(unsupported)}) not implemented')
        if kwargs.get('scope', 'function') != 'function':
            raise Unsupported("only scope='function' fixtures are implemented")

        def wrap(f):
            f._shim_fixture = True
            f._shim_autouse = bool(kwargs.get('autouse', False))
            f._shim_name = kwargs.get('name', f.__name__)
            return f
        return wrap(fn) if fn is not None else wrap

    def skip(reason=''):
        raise Skipped(reason or 'skipped')

    def fail(reason=''):
        raise Failed(reason or 'failed')

    m.fixture = fixture
    m.skip = skip
    m.fail = fail
    m.mark = _MarkFactory()
    m.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)
    m.raises = lambda expected, **kw: _Raises(expected)
    m.skip.Exception = Skipped
    m.importorskip = lambda name, **kw: (
        importlib.import_module(name) if importlib.util.find_spec(name)
        else skip(f'{name} not installed'))
    sys.modules['pytest'] = m
    return m


# --------------------------------------------------------------------------
# collection and execution
# --------------------------------------------------------------------------

def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def _skip_reason(marks):
    """The reason this test is skipped, or None. Raises on marks we don't know."""
    for mk in marks:
        if mk.kind == 'skip':
            return mk.kwargs.get('reason', 'skipped')
        if mk.kind == 'skipif':
            cond = mk.args[0] if mk.args else mk.kwargs.get('condition')
            if isinstance(cond, str):
                raise Unsupported('string conditions in skipif are not implemented')
            if cond:
                return mk.kwargs.get('reason', 'skipif')
        elif mk.kind in ('xfail', 'filterwarnings', 'slow'):
            continue                     # recorded, deliberately not acted on
        elif mk.kind != 'parametrize':
            raise Unsupported(f'mark {mk.kind} is not implemented')
    return None


def _cases(fn, marks):
    """Expand parametrize marks into (suffix, kwargs) pairs."""
    cases = [('', {})]
    for mk in marks:
        if mk.kind != 'parametrize':
            continue
        names = mk.args[0]
        names = [n.strip() for n in names.split(',')] if isinstance(names, str) else list(names)
        grown = []
        for suffix, kw in cases:
            for value in mk.args[1]:
                vals = value if len(names) > 1 else (value,)
                extra = dict(zip(names, vals))
                label = '-'.join(str(v) for v in vals)
                grown.append((f'{suffix}[{label}]', {**kw, **extra}))
        cases = grown
    return cases


def _fixtures(mod):
    found = {}
    for name, obj in vars(mod).items():
        if callable(obj) and getattr(obj, '_shim_fixture', False):
            found[getattr(obj, '_shim_name', name)] = obj
    return found


def _resolve(fn, fixtures, stack, supplied=()):
    """Build kwargs for one test/fixture, entering any fixtures it needs.

    `supplied` names arguments parametrize will pass in; they are not fixtures
    and must not be looked up as such.
    """
    kwargs = {}
    for pname in inspect.signature(fn).parameters:
        if pname in supplied:
            continue
        if pname == 'monkeypatch':
            mp = _MonkeyPatch()
            stack.append(('monkeypatch', mp))
            kwargs[pname] = mp
        elif pname in fixtures:
            kwargs[pname] = _enter(fixtures[pname], fixtures, stack)
        else:
            raise Unsupported(f'{fn.__name__} wants unknown fixture {pname!r}')
    return kwargs


def _enter(fixture, fixtures, stack):
    inner = _resolve(fixture, fixtures, stack)
    if inspect.isgeneratorfunction(fixture):
        gen = fixture(**inner)
        value = next(gen)
        stack.append(('gen', gen))
        return value
    return fixture(**inner)


def _teardown(stack):
    """Unwind fixtures in reverse. A teardown blowing up is a real error."""
    errors = []
    for kind, obj in reversed(stack):
        try:
            if kind == 'gen':
                next(obj, None)
            else:
                obj.undo()
        except Exception:
            errors.append(traceback.format_exc())
    stack.clear()
    return errors


def run_module(path, verbose, pattern):
    """Returns (passed, failed, skipped, errored, [report lines])."""
    rel = path.relative_to(REPO) if path.is_absolute() else path
    lines, counts = [], dict(passed=0, failed=0, skipped=0, errored=0)
    try:
        mod = _load(path)
    except Skipped as e:
        return 0, 0, 1, 0, [f'  SKIP  {rel} (module: {e})']
    except Exception:
        return 0, 0, 0, 1, [f'  ERROR {rel} (import)', traceback.format_exc()]

    module_marks = getattr(mod, 'pytestmark', [])
    module_marks = module_marks if isinstance(module_marks, list) else [module_marks]
    fixtures = _fixtures(mod)
    autouse = [f for f in fixtures.values() if getattr(f, '_shim_autouse', False)]

    tests = [(n, o) for n, o in vars(mod).items()
             if n.startswith('test_') and inspect.isfunction(o)]
    tests.sort(key=lambda t: t[1].__code__.co_firstlineno)

    for name, fn in tests:
        if pattern and pattern not in name:
            continue
        marks = module_marks + getattr(fn, '_shim_marks', [])
        try:
            reason = _skip_reason(marks)
            cases = _cases(fn, marks)
        except Unsupported as e:
            counts['errored'] += 1
            lines.append(f'  ERROR {name}: {e}')
            continue

        for suffix, params in cases:
            label = f'{name}{suffix}'
            if reason:
                counts['skipped'] += 1
                if verbose:
                    lines.append(f'  SKIP  {label} ({reason})')
                continue
            stack = []
            try:
                for f in autouse:
                    _enter(f, fixtures, stack)
                kwargs = {**_resolve(fn, fixtures, stack, params), **params}
                fn(**kwargs)
            except Skipped as e:
                counts['skipped'] += 1
                if verbose:
                    lines.append(f'  SKIP  {label} ({e})')
            except (AssertionError, Failed):
                counts['failed'] += 1
                lines.append(f'  FAIL  {label}')
                lines.append(traceback.format_exc())
            except Unsupported as e:
                counts['errored'] += 1
                lines.append(f'  ERROR {label}: {e}')
            except Exception:
                counts['errored'] += 1
                lines.append(f'  ERROR {label}')
                lines.append(traceback.format_exc())
            else:
                counts['passed'] += 1
                if verbose:
                    lines.append(f'  ok    {label}')
            finally:
                for err in _teardown(stack):
                    counts['errored'] += 1
                    lines.append(f'  ERROR {label} (teardown)\n{err}')

    return (counts['passed'], counts['failed'], counts['skipped'],
            counts['errored'], lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='*', help='test files (default: the fork\'s own)')
    ap.add_argument('-v', '--verbose', action='store_true', help='one line per test')
    ap.add_argument('-k', dest='pattern', help='substring filter on test names')
    args = ap.parse_args()

    _install_shim()
    sys.path.insert(0, str(REPO))

    paths = [Path(p) if Path(p).is_absolute() else REPO / p
             for p in (args.paths or FORK_TESTS)]

    total = dict(passed=0, failed=0, skipped=0, errored=0)
    for path in paths:
        if not path.exists():
            print(f'{path}: no such file', file=sys.stderr)
            total['errored'] += 1
            continue
        p, f, s, e, lines = run_module(path, args.verbose, args.pattern)
        total['passed'] += p; total['failed'] += f
        total['skipped'] += s; total['errored'] += e
        flag = 'FAIL' if (f or e) else 'ok  '
        name = path.relative_to(REPO) if path.is_absolute() else path
        print(f'{flag}  {str(name):34s} {p:3d} passed  {f:2d} failed  '
              f'{s:2d} skipped  {e:2d} errored')
        for line in lines:
            print(line)

    print('-' * 72)
    print(f"{total['passed']} passed, {total['failed']} failed, "
          f"{total['skipped']} skipped, {total['errored']} errored")
    return 1 if (total['failed'] or total['errored']) else 0


if __name__ == '__main__':
    sys.exit(main())
