# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the declarations of a Java source file with tree-sitter.

``generate_api_reference.py`` uses this to find the types and members that the
Java and Lucene API pages document. Only the structure comes from here; the
text of each Javadoc comment is returned as written and interpreted by the
generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

TYPE_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "@interface",
}
MEMBER_KINDS = {
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "compact_constructor_declaration": "constructor",
    "field_declaration": "field",
    "constant_declaration": "field",
    "enum_constant": "enum_constant",
    "annotation_type_element_declaration": "method",
}
# The node that ends a declaration's signature: what follows is its body.
BODY_TYPES = {
    "block",
    "class_body",
    "constructor_body",
    "enum_body",
    "interface_body",
    "annotation_type_body",
}
COMMENT_TYPES = ("line_comment", "block_comment")


@dataclass
class JavaDeclaration:
    """A type or member declaration and the Javadoc comment before it."""

    kind: str
    """``class``, ``interface``, ``enum``, ``record``, ``@interface``,
    ``method``, ``constructor``, ``field`` or ``enum_constant``."""
    name: str
    signature: str
    """The declaration up to its body or initializer, on one line, with
    annotations and without comments."""
    comment: str
    """The ``/** ... */`` comment before the declaration, or ``""``."""
    line: int
    """The line of the declaration's name."""
    comment_end_line: int
    """The line the comment ends on, or 0 without one."""
    modifiers: frozenset[str]
    """Modifier keywords, such as ``public`` or ``abstract``."""
    visible: bool
    """Whether the declaration is public, protected or implicitly public. An
    enclosing type that is not visible hides it all the same."""
    supertypes: list[str] = field(default_factory=list)
    """For a type, the simple names of the types it extends and implements."""
    parameter_types: tuple[str, ...] = ()
    """For a method or constructor, the types of its parameters."""
    members: list[JavaDeclaration] = field(default_factory=list)
    """For a type, its members and nested types, in source order."""

    @property
    def is_type(self) -> bool:
        return self.kind in TYPE_KINDS.values()


@dataclass
class JavaSourceFile:
    package: str
    types: list[JavaDeclaration]
    """The top-level types, in source order."""


def check_tree_sitter() -> None:
    """Exits with instructions if tree-sitter is not installed."""
    _parser()


def parse_java_file(path: Path) -> JavaSourceFile:
    return parse_java_source(path.read_bytes(), str(path))


def parse_java_source(source: bytes, name: str = "<source>") -> JavaSourceFile:
    """Parses ``source``, which error messages call ``name``."""
    tree = _parser().parse(source)
    root = tree.root_node
    if root.has_error:
        error = _first_error(root)
        raise ValueError(
            f"{name}:{error.start_point[0] + 1}: cannot parse the Java source"
        )
    package = ""
    types = []
    for child in root.named_children:
        if child.type == "package_declaration":
            package = _text(child.named_children[-1])
        elif child.type in TYPE_KINDS:
            types.append(_type(child, source, enclosing_kind=None))
    return JavaSourceFile(package=package, types=types)


@cache
def _parser():
    try:
        import tree_sitter
        import tree_sitter_java
    except ImportError as error:
        raise SystemExit(
            "Generating the Java API pages needs the tree_sitter and "
            "tree-sitter-java packages, which the cuVS conda environment "
            "now includes. Update it, for example: conda env update "
            "--name cuvs --file "
            "conda/environments/all_cuda-<version>_arch-$(uname -m).yaml"
        ) from error
    return tree_sitter.Parser(
        tree_sitter.Language(tree_sitter_java.language())
    )


def _type(node, source: bytes, enclosing_kind: str | None):
    kind = TYPE_KINDS[node.type]
    decl = _declaration(node, kind, source, enclosing_kind)
    decl.supertypes = _supertypes(node)
    body = node.child_by_field_name("body")
    if body is None:
        return decl
    children = list(body.named_children)
    # An enum's fields, constructors and methods follow its constants.
    for child in body.named_children:
        if child.type == "enum_body_declarations":
            children.extend(child.named_children)
    for child in children:
        if child.type in TYPE_KINDS:
            decl.members.append(_type(child, source, kind))
        elif child.type in MEMBER_KINDS:
            decl.members.append(
                _declaration(child, MEMBER_KINDS[child.type], source, kind)
            )
    return decl


def _declaration(node, kind, source, enclosing_kind):
    modifiers = _modifiers(node)
    comment = _doc_comment(node)
    name_node = _name_node(node)
    return JavaDeclaration(
        kind=kind,
        name=_text(name_node),
        signature=_signature(node, source),
        comment=_text(comment) if comment is not None else "",
        line=name_node.start_point[0] + 1,
        comment_end_line=comment.end_point[0] + 1 if comment else 0,
        modifiers=modifiers,
        visible=_is_visible(kind, modifiers, enclosing_kind),
        parameter_types=_parameter_types(node),
    )


def _is_visible(
    kind: str, modifiers: frozenset[str], enclosing_kind: str | None
) -> bool:
    if kind == "enum_constant":
        return True
    if enclosing_kind in ("interface", "@interface"):
        # Interface members are implicitly public; only private methods are not.
        return "private" not in modifiers
    if enclosing_kind is None:
        return "public" in modifiers
    return bool(modifiers & {"public", "protected"})


def _modifiers(node) -> frozenset[str]:
    for child in node.children:
        if child.type == "modifiers":
            return frozenset(
                _text(token)
                for token in child.children
                if token.type not in ("marker_annotation", "annotation")
            )
    return frozenset()


def _doc_comment(node):
    """Returns the Javadoc comment that documents ``node``, if any.

    Other comments and the comma between enum constants may come in between.
    """
    previous = node.prev_sibling
    while previous is not None and (
        previous.type in ("line_comment", ",")
        or (
            previous.type == "block_comment"
            and not previous.text.startswith(b"/**")
        )
    ):
        previous = previous.prev_sibling
    if (
        previous is not None
        and previous.type == "block_comment"
        and previous.text.startswith(b"/**")
    ):
        return previous
    return None


def _name_node(node):
    name = node.child_by_field_name("name")
    if name is None:
        # A field: the name of its first declarator.
        name = node.child_by_field_name("declarator").child_by_field_name(
            "name"
        )
    return name


def _signature(node, source: bytes) -> str:
    """Returns ``node``'s source on one line and without comments.

    The source ends before the body, or for a field before its initializer.
    """
    end = node.end_byte
    for child in node.children:
        if child.type in BODY_TYPES or child.type == ";":
            end = child.start_byte
            break
        value = (
            child.child_by_field_name("value")
            if child.type == "variable_declarator"
            else None
        )
        if value is not None:
            end = value.start_byte
            break
    text = bytearray(source[node.start_byte : end])
    for comment in reversed(_comments_within(node, end)):
        start = comment.start_byte - node.start_byte
        text[start : comment.end_byte - node.start_byte] = b" "
    signature = " ".join(text.decode("utf-8").split())
    return signature.removesuffix("=").rstrip()


def _comments_within(node, end: int) -> list:
    comments = []
    stack = list(node.children)
    while stack:
        child = stack.pop()
        if child.start_byte >= end:
            continue
        if child.type in COMMENT_TYPES:
            comments.append(child)
        else:
            stack.extend(child.children)
    return sorted(comments, key=lambda comment: comment.start_byte)


def _supertypes(node) -> list[str]:
    names = []
    for field_name in ("superclass", "interfaces"):
        clause = node.child_by_field_name(field_name)
        if clause is not None:
            names.extend(_simple_type_names(clause))
    # An interface's supertypes are not in a field.
    for child in node.named_children:
        if child.type == "extends_interfaces":
            names.extend(_simple_type_names(child))
    return names


def _simple_type_names(clause) -> list[str]:
    """Returns the simple names of the types a supertype clause lists.

    Type arguments are left out.
    """
    names = []
    stack = list(reversed(clause.named_children))
    while stack:
        child = stack.pop()
        if child.type == "type_identifier":
            names.append(_text(child))
        elif child.type == "scoped_type_identifier":
            names.append(_text(child.named_children[-1]))
        elif child.type == "generic_type":
            stack.append(child.named_children[0])
        else:
            stack.extend(reversed(child.named_children))
    return names


def _parameter_types(node) -> tuple[str, ...]:
    parameters = node.child_by_field_name("parameters")
    if parameters is None or parameters.type != "formal_parameters":
        return ()
    types = []
    for parameter in parameters.named_children:
        if parameter.type == "formal_parameter":
            types.append(
                _type_text(parameter.child_by_field_name("type"))
                + "[]" * _dimensions(parameter)
            )
        elif parameter.type == "spread_parameter":
            type_node = next(
                child
                for child in parameter.named_children
                if child.type not in ("modifiers", "variable_declarator")
                and child.type not in COMMENT_TYPES
            )
            types.append(_type_text(type_node) + "...")
    return tuple(types)


def _dimensions(parameter) -> int:
    """Counts the ``[]`` after a parameter's name, as in ``int values[]``."""
    dimensions = parameter.child_by_field_name("dimensions")
    return _text(dimensions).count("[") if dimensions is not None else 0


def _type_text(node) -> str:
    """Returns a type without annotations, comments or whitespace."""
    parts = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in ("marker_annotation", "annotation", *COMMENT_TYPES):
            continue
        if current.child_count == 0:
            parts.append(_text(current))
        else:
            stack.extend(reversed(current.children))
    return "".join(parts)


def _first_error(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_error or current.is_missing:
            return current
        stack.extend(reversed(current.children))
    return node


def _text(node) -> str:
    return node.text.decode("utf-8")
