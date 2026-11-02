#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile


from check_c_abi.abi import Abi, analyze_c_abi


def abi_from_str(header_src: str) -> Abi:
    with tempfile.NamedTemporaryFile(
        mode="w+t", delete=True, suffix=".h"
    ) as tmp:
        tmp.write(header_src)
        tmp.flush()
        filename = pathlib.Path(tmp.name)
        return Abi.from_include_path(filename.parent, filename.name)


def test_function():
    old_abi = abi_from_str("int func(int x, char * y); ")
    errors = analyze_c_abi(old_abi, old_abi)
    assert not errors

    # changing the return type should return an error
    new_abi = abi_from_str("char func(int x, char * y); ")
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "func"
    assert errors[0].error.startswith("Function has return type changed")

    # adding parameters should return an error
    new_abi = abi_from_str("int func(int x, char * y, float z); ")
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "func"
    assert errors[0].error.startswith("Function has a new parameter")

    # removing parameters should return an error
    new_abi = abi_from_str("int func(int x); ")
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "func"
    assert errors[0].error.startswith("Function has a deleted parameter")

    # adding new functions is allowed
    new_abi = abi_from_str("""
        int func(int x, char*y);
        int func2(float x);
    """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert not errors


def test_struct():
    old_abi = abi_from_str("""
        struct Foo {
            int a;
            float b;
        }; """)

    errors = analyze_c_abi(old_abi, old_abi)
    assert not errors

    # removing a struct should return an error
    new_abi = abi_from_str("")
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "Foo"
    assert errors[0].error == "Struct has been removed"

    # removing a field should return an error
    new_abi = abi_from_str("""
        struct Foo {
            int a;
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "Foo"
    assert errors[0].error.startswith("Struct has a deleted member")

    # adding a new field should not return an error
    new_abi = abi_from_str("""
        struct Foo {
            int a;
            float b;
            bool c;
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert not errors

    # Changing the type of a field should return an error
    new_abi = abi_from_str("""
        struct Foo {
            int a;
            int b;
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "Foo"
    assert errors[0].error.startswith("Struct member has changed type")

    # adding new structs is allowed
    new_abi = abi_from_str("""
        struct Foo {
            int a;
            float b;
        };
        struct Foo2 {
            double c;
        };
        """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert not errors


def test_enum():
    old_abi = abi_from_str("""
        enum Foo {
            ZERO = 0,
            TWO =  2,
            HUNDRED = 100
        }; """)

    errors = analyze_c_abi(old_abi, old_abi)
    assert not errors

    # test removing values from an enum
    new_abi = abi_from_str("""
        enum Foo {
            ZERO = 0,
            HUNDRED = 100
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "Foo"
    assert errors[0].error == "Enum value TWO has been removed"

    # adding new values to an enum is allowed
    new_abi = abi_from_str("""
        enum Foo {
            ZERO = 0,
            TWO =  2,
            THREE = 3,
            HUNDRED = 100
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert not errors

    # changing enum values isn't allowed
    new_abi = abi_from_str("""
        enum Foo {
            ZERO = 0,
            TWO =  200,
            HUNDRED = 100
        }; """)
    errors = analyze_c_abi(old_abi, new_abi)
    assert len(errors) == 1
    assert errors[0].symbol == "Foo"
    assert errors[0].error.startswith("Enum value TWO has been changed")
