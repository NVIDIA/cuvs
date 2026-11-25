# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from java_source import parse_java_file, parse_java_source  # noqa: E402

SOURCE = b"""
package com.example.api;

import java.util.List;

/** The outer type. */
@Deprecated(since = "1.0", forRemoval = true)
public abstract class Outer<T extends Comparable<T>> extends Base<T>
    implements Api, java.io.Closeable {

  /** A constant. */
  public static final String NAME = "a { b";

  private int hidden;

  /** A method. */
  /* Not Javadoc. */
  @SuppressWarnings({"unchecked", "rawtypes"})
  public <R> List<R> map(final List</* any */ ? extends T> values, int[] counts, final /* c */ String... names)
      throws java.io.IOException {
    return null;
  }

  /** Not readable outside the package. */
  void packagePrivate() {}

  protected Outer(/* before */ int size) {}

  /** A builder. */
  public static class Builder {
    /** Builds it. */
    public Outer<?> build() { return null; }
    private void secret() {}
  }

  private static class Private {
    /** Public, but in a private class. */
    public void unreachable() {}
  }

  /** An enum. */
  public enum Mode {
    /** First. */
    FIRST(1),
    /** Second. */
    SECOND(2) {
      @Override public int code() { return 20; }
    },
    THIRD;

    private final int code;

    Mode() { this(0); }

    Mode(int code) { this.code = code; }

    /** The code. */
    public int code() { return code; }
  }

  /** An interface. */
  public interface Callback {
    /** Called. */
    void call(int value);

    default void twice() {}

    private void helper() {}
  }

  /** A record. */
  public record Pair(int left, int right) implements Api {
    /** Checks it. */
    public Pair {}
  }
}

class Sibling {
  public void notDocumented() {}
}
"""


@pytest.fixture(scope="module")
def outer():
    parsed = parse_java_source(SOURCE)
    assert parsed.package == "com.example.api"
    assert [decl.name for decl in parsed.types] == ["Outer", "Sibling"]
    assert [decl.visible for decl in parsed.types] == [True, False]
    return parsed.types[0]


def member(decl, name):
    return next(m for m in decl.members if m.name == name)


def test_type(outer):
    assert outer.kind == "class"
    assert outer.comment == "/** The outer type. */"
    assert outer.signature == (
        '@Deprecated(since = "1.0", forRemoval = true) public abstract class '
        "Outer<T extends Comparable<T>> extends Base<T> implements Api, "
        "java.io.Closeable"
    )
    assert outer.supertypes == ["Base", "Api", "Closeable"]
    assert outer.line == 8
    assert outer.comment_end_line == 6
    assert {"public", "abstract"} <= outer.modifiers


def test_members_in_source_order(outer):
    assert [m.name for m in outer.members] == [
        "NAME",
        "hidden",
        "map",
        "packagePrivate",
        "Outer",
        "Builder",
        "Private",
        "Mode",
        "Callback",
        "Pair",
    ]


def test_field_signature_stops_at_initializer(outer):
    name = member(outer, "NAME")
    assert name.kind == "field"
    assert name.signature == "public static final String NAME"
    assert name.visible
    assert not member(outer, "hidden").visible


def test_method(outer):
    method = member(outer, "map")
    assert method.kind == "method"
    assert method.signature == (
        '@SuppressWarnings({"unchecked", "rawtypes"}) public <R> List<R> '
        "map(final List< ? extends T> values, int[] counts, final String... names) "
        "throws java.io.IOException"
    )
    assert method.parameter_types == ("List<?extendsT>", "int[]", "String...")
    assert method.comment == "/** A method. */"
    assert not member(outer, "packagePrivate").visible


def test_comments_are_dropped_from_signatures(outer):
    constructor = member(outer, "Outer")
    assert constructor.kind == "constructor"
    assert constructor.signature == "protected Outer( int size)"
    assert constructor.comment == ""
    assert constructor.visible


def test_nested_class(outer):
    builder = member(outer, "Builder")
    assert builder.is_type and builder.visible
    assert [(m.name, m.visible) for m in builder.members] == [
        ("build", True),
        ("secret", False),
    ]
    assert not member(outer, "Private").visible


def test_enum(outer):
    mode = member(outer, "Mode")
    assert mode.kind == "enum"
    constants = [m for m in mode.members if m.kind == "enum_constant"]
    assert [(c.signature, c.comment) for c in constants] == [
        ("FIRST(1)", "/** First. */"),
        ("SECOND(2)", "/** Second. */"),
        ("THIRD", ""),
    ]
    assert all(c.visible for c in constants)
    # Enum constructors are implicitly private.
    constructors = [m for m in mode.members if m.kind == "constructor"]
    assert len(constructors) == 2
    assert not any(c.visible for c in constructors)
    assert [
        (m.kind, m.name, m.visible)
        for m in mode.members
        if m.kind in ("field", "method")
    ] == [("field", "code", False), ("method", "code", True)]


def test_interface_members_are_implicitly_public(outer):
    callback = member(outer, "Callback")
    assert [(m.name, m.visible) for m in callback.members] == [
        ("call", True),
        ("twice", True),
        ("helper", False),
    ]
    assert member(callback, "call").signature == "void call(int value)"


def test_record(outer):
    pair = member(outer, "Pair")
    assert pair.kind == "record"
    assert (
        pair.signature
        == "public record Pair(int left, int right) implements Api"
    )
    assert pair.supertypes == ["Api"]
    constructor = member(pair, "Pair")
    assert constructor.kind == "constructor"
    assert constructor.signature == "public Pair"
    assert constructor.comment == "/** Checks it. */"


BROKEN = b"class A {\n  void f( {}\n}\n"


def test_syntax_error():
    with pytest.raises(ValueError, match="^<source>:2: cannot parse"):
        parse_java_source(BROKEN)


def test_syntax_error_names_the_file(tmp_path):
    path = tmp_path / "A.java"
    path.write_bytes(BROKEN)
    with pytest.raises(ValueError) as error:
        parse_java_file(path)
    assert str(error.value) == f"{path}:2: cannot parse the Java source"
