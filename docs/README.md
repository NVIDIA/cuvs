# Building Documentation

The cuVS documentation is a Fern project in [../fern](../fern).

Fern requires Node.js 22 or newer. If the docs fail with an error such as `SyntaxError: Unexpected token '.'`, check `node --version` and activate a newer Node.js runtime.

## Preview locally

```bash
fern/build_docs.sh dev
```

Fern serves the preview at [http://localhost:3000](http://localhost:3000) by default.

## Validate

```bash
fern/build_docs.sh check
```

The Fern build refreshes the C, C++, Python, Java, Rust, and Go API reference pages from the source tree before validating.

## Publish

Create a Fern preview deployment:

```bash
fern/build_docs.sh preview
```

Publish the production docs site:

```bash
fern/build_docs.sh publish
```
