#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import msgspec
from itertools import zip_longest
import pathlib
from typing import Generator, Optional, Self
import os


import clang.cindex

try:
    from termcolor import colored
except ImportError:

    def colored(text, *args, **kwargs):
        return text


def _is_unnamed_struct(cursor: clang.cindex.Cursor) -> bool:
    return (
        cursor.kind == clang.cindex.CursorKind.STRUCT_DECL
        and cursor.spelling.startswith("struct ")
        and "unnamed " in cursor.spelling
    )


class SymbolLocation(msgspec.Struct):
    filename: str
    line: int
    column: int

    @classmethod
    def from_cursor(
        cls,
        cursor: clang.cindex.Cursor,
        root_path: Optional[str | os.PathLike] = None,
    ) -> Self:
        filename = cursor.location.file.name
        if root_path:
            filename = str(pathlib.Path(filename).relative_to(root_path))
        return cls(
            filename=filename,
            line=cursor.location.line,
            column=cursor.location.column,
        )


class FunctionDefinition(msgspec.Struct):
    name: str
    return_type: str
    parameters: list[tuple[str, str]]
    location: SymbolLocation

    @classmethod
    def from_cursor(
        cls,
        cursor: clang.cindex.Cursor,
        root_path: Optional[str | os.PathLike] = None,
    ) -> Self:
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            raise ValueError(
                f"FunctionDefinition.from_cursor called with cursor of kind={cursor.kind}"
            )

        return cls(
            name=cursor.spelling,
            return_type=cursor.result_type.spelling,
            parameters=[
                (child.type.spelling, child.spelling)
                for child in cursor.get_children()
                if child.kind == clang.cindex.CursorKind.PARM_DECL
            ],
            location=SymbolLocation.from_cursor(cursor, root_path),
        )


class StructDefinition(msgspec.Struct):
    name: str
    members: list[tuple[str, str]]
    location: SymbolLocation

    @classmethod
    def from_cursor(
        cls,
        cursor: clang.cindex.Cursor,
        root_path: Optional[str | os.PathLike] = None,
    ) -> Self:
        if cursor.kind != clang.cindex.CursorKind.STRUCT_DECL:
            raise ValueError(
                f"StructDefinition.from_cursor called with cursor of kind={cursor.kind}"
            )

        return cls(
            name=cursor.spelling,
            members=[
                (child.type.spelling, child.spelling)
                for child in cursor.get_children()
                if child.kind == clang.cindex.CursorKind.FIELD_DECL
            ],
            location=SymbolLocation.from_cursor(cursor, root_path),
        )


class EnumDefinition(msgspec.Struct):
    name: str
    values: list[tuple[str, int]]
    location: SymbolLocation

    @classmethod
    def from_cursor(
        cls,
        cursor: clang.cindex.Cursor,
        root_path: Optional[str | os.PathLike] = None,
    ) -> Self:
        if cursor.kind != clang.cindex.CursorKind.ENUM_DECL:
            raise ValueError(
                f"EnumDefinition.from_cursor called with cursor of kind={cursor.kind}"
            )

        return cls(
            name=cursor.spelling,
            values=[
                (child.spelling, child.enum_value)
                for child in cursor.get_children()
                if child.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL
            ],
            location=SymbolLocation.from_cursor(cursor, root_path),
        )


class Abi(msgspec.Struct):
    functions: list[FunctionDefinition]
    structs: list[StructDefinition]
    enums: list[EnumDefinition]

    @classmethod
    def from_include_path(
        cls,
        root: str | os.PathLike,
        header: str | os.PathLike,
        extra_clang_args: Optional[list[str]] = None,
    ) -> Self:
        """Loads the Abi from a root path ('/source/cuvs/c/include') and a header file
        ("cuvs/include/all.h")
        """
        path = pathlib.Path(root).resolve()
        all_header = path / header

        if not all_header.is_file():
            raise ValueError(f"header file '{all_header}' not found")

        index = clang.cindex.Index.create()

        args = [f"-I{str(path)}"]
        if extra_clang_args:
            args.extend(extra_clang_args)

        tu = index.parse(all_header, args=args)

        functions, structs, enums = [], [], []

        # note: we could use tu.cursor.walk_preorder() here instead to recurse through the AST
        # but it is slightly slower to do so (extra 100ms or so) and for the cuvs C-ABI everything
        # is at the top level
        for child in tu.cursor.get_children():
            # ignore things like cuda headers and other files not installed in
            # in the cuvs C path
            if not (
                child.location.file
                and pathlib.Path(child.location.file.name).is_relative_to(path)
            ):
                continue

            # Store definitions for each function, struct and enum
            if child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                functions.append(
                    FunctionDefinition.from_cursor(child, root_path=path)
                )
            elif child.kind == clang.cindex.CursorKind.STRUCT_DECL:
                # ignore unnamed structs (will get picked up via the typedef)
                if _is_unnamed_struct(child):
                    continue
                structs.append(
                    StructDefinition.from_cursor(child, root_path=path)
                )
            elif child.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                # check if this is a typedef to an unnamed struct, if so use the
                # typedef as the symbolname for the struct
                grandchildren = list(child.get_children())
                if len(grandchildren) == 1 and _is_unnamed_struct(
                    grandchildren[0]
                ):
                    struct = StructDefinition.from_cursor(
                        grandchildren[0], root_path=path
                    )
                    struct.name = child.spelling
                    structs.append(struct)
            elif child.kind == clang.cindex.CursorKind.ENUM_DECL:
                enums.append(EnumDefinition.from_cursor(child, root_path=path))

        return cls(functions, structs, enums)


class AbiError(msgspec.Struct):
    """Holds information about an ABI breaking error"""

    error: str
    symbol: Optional[str] = None
    location: Optional[SymbolLocation] = None


def _analyze_function_abi(
    old_abi: Abi, new_abi: Abi
) -> Generator[AbiError, None, None]:
    """This iterates over every function in the existing abi, and make sure that no functions
    have been removed or had function parameters removed, parameters added or the type of any
    parameter changed. Note: adding new functions to the new abi is allowed
    """
    old_functions = {f.name: f for f in old_abi.functions}
    new_functions = {f.name: f for f in new_abi.functions}

    for name, old_function in old_functions.items():
        try:
            new_function = new_functions[name]
        except KeyError:
            yield AbiError(
                "Function has been removed",
                symbol=old_function.name,
                location=old_function.location,
            )
            continue

        if old_function.return_type != new_function.return_type:
            yield AbiError(
                f"Function has return type changed from '{old_function.return_type}' to '{new_function.return_type}'",
                symbol=new_function.name,
                location=new_function.location,
            )

        for (old_type, old_name), (new_type, new_name) in zip_longest(
            old_function.parameters,
            new_function.parameters,
            fillvalue=(None, None),
        ):
            if old_type is None:
                yield AbiError(
                    f"Function has a new parameter '{new_type} {new_name}'",
                    symbol=new_function.name,
                    location=new_function.location,
                )

            elif new_type is None:
                yield AbiError(
                    f"Function has a deleted parameter '{old_type} {old_name}'",
                    symbol=old_function.name,
                    location=old_function.location,
                )

            elif new_type != old_type:
                yield AbiError(
                    f"Function has changed parameter type '{old_type}' to '{new_type}' for '{old_name}'",
                    symbol=new_function.name,
                    location=new_function.location,
                )


def _analyze_struct_abi(
    old_abi: Abi, new_abi: Abi
) -> Generator[AbiError, None, None]:
    """Checks to see if any existing structures have had items removed, reordered, renamed, or types
    changed (adding new members is considered to be ok, as long as functions are initialized via
    a create factory function)
    """
    old_structs = {f.name: f for f in old_abi.structs}
    new_structs = {f.name: f for f in new_abi.structs}

    for name, old_struct in old_structs.items():
        try:
            new_struct = new_structs[name]
        except KeyError:
            yield AbiError(
                "Struct has been removed",
                symbol=name,
                location=old_struct.location,
            )
            continue

        for (old_type, old_name), (new_type, new_name) in zip_longest(
            old_struct.members,
            new_struct.members,
            fillvalue=(None, None),
        ):
            if new_type is None:
                yield AbiError(
                    f"Struct has a deleted member '{old_type} {old_name}'",
                    symbol=name,
                    location=new_struct.location,
                )
            elif old_type is None:
                # adding an item to the end of the struct is allowed here
                pass

            elif new_type != old_type:
                yield AbiError(
                    f"Struct member has changed type '{old_type}' to '{new_type}' for member '{old_name}'",
                    symbol=name,
                    location=new_struct.location,
                )


def _analyze_enum_abi(
    old_abi: Abi, new_abi: Abi
) -> Generator[AbiError, None, None]:
    # flatten enum values: since values inside an enum in C are in the global scope
    old_enum_values = {
        k: (v, enum) for enum in old_abi.enums for k, v in enum.values
    }
    new_enum_values = {
        k: (v, enum) for enum in new_abi.enums for k, v in enum.values
    }

    # check to see if enum values have been removed, or had their numeric values changed
    for name, (old_value, old_enum) in old_enum_values.items():
        try:
            new_value, new_enum = new_enum_values[name]
        except KeyError:
            yield AbiError(
                f"Enum value {name} has been removed",
                symbol=old_enum.name,
                location=old_enum.location,
            )
            continue

        if new_value != old_value:
            yield AbiError(
                f"Enum value {name} has been changed from {old_value} to {new_value}",
                symbol=old_enum.name,
                location=new_enum.location,
            )


def analyze_c_abi(old_abi: Abi, new_abi: Abi) -> list[AbiError]:
    """Compares two Abi objects and returns a list of errors for any ABI breaking
    changes between them
    """
    errors = []
    errors.extend(_analyze_function_abi(old_abi, new_abi))
    errors.extend(_analyze_struct_abi(old_abi, new_abi))
    errors.extend(_analyze_enum_abi(old_abi, new_abi))
    return errors
