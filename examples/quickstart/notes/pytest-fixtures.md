# pytest fixtures — patterns

## Scope ladder

`function` (default) → `class` → `module` → `package` → `session`. Pick
the **narrowest** scope that still gives you the perf you need.

## `tmp_path`

Always prefer `tmp_path` over hand-rolled `tempfile.mkdtemp`; pytest
cleans it up automatically.

```python
def test_writes_file(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("hello")
    assert target.read_text() == "hello"
```

## Parametrize

```python
@pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (10, 20, 30)])
def test_add(a, b, expected):
    assert a + b == expected
```

## Fixtures that yield

```python
@pytest.fixture
def db():
    conn = open_conn()
    yield conn
    conn.close()
```
