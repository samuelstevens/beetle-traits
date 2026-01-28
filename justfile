docs: fmt
    rm -rf docs/api
    mkdir -p docs/api
    -yek src/btx README.md AGENTS.md > docs/api/llms.txt
    uvx pdoc3 --force --output-dir docs/api --config latex_math=True src/btx

lint: fmt
    uvx ruff check --fix .

fmt:
    uvx ruff format --preview .

clean:
    rm -rf .ruff_cache/

