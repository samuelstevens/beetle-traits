lint: fmt
    uv run ruff check --fix .

fmt:
    uv run ruff format --preview .

clean:
    rm -rf .ruff_cache/

