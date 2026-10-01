# cuVS Fern documentation

The cuVS documentation lives in this Fern project. Pages are in `fern/pages`, and the sidebar navigation is configured in `fern/docs.yml`.

The C, C++, Python, Java, Rust, and Go API reference pages are generated from the source tree by `fern/scripts/generate_api_reference.py`. `fern/build_docs.sh` refreshes those pages before validation, preview, and publish runs.

Fern requires Node.js 22 or newer. If the docs fail with an error such as `SyntaxError: Unexpected token '.'`, check `node --version` and activate a newer Node.js runtime.

## Preview locally

Start the local preview server from the repository root:

```bash
fern/build_docs.sh dev
```

Fern serves the local preview at `http://localhost:3000` by default.

## Validate

Run Fern's checks before publishing changes:

```bash
fern/build_docs.sh check
```

## Publish

Create a Fern preview deployment:

```bash
fern/build_docs.sh preview
```

Publish to the instance configured in `fern/docs.yml` with:

```bash
fern/build_docs.sh publish
```
