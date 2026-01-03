## Test Helper Module
##
## Provides utilities for reducing test redundancy:
## - Table-driven tests
## - Parameterized test cases
## - Common assertion patterns

import std/[strformat, strutils, tables, macros]
import unittest

# ============================================================================
# Table-Driven Test Support
# ============================================================================

template tableTest*[T](suiteName: string, cases: openArray[T],
                        body: untyped) =
  ## Run table-driven tests
  ## Usage:
  ##   tableTest "MySuite", [("case1", 1), ("case2", 2)]:
  ##     test tc.name:
  ##       check tc.value > 0
  suite suiteName:
    for tc {.inject.} in cases:
      body

# ============================================================================
# Parameterized Test Helpers
# ============================================================================

type
  TestParam*[I, E] = object
    name*: string
    input*: I
    expected*: E

proc param*[I, E](name: string, input: I, expected: E): TestParam[I, E] =
  TestParam[I, E](name: name, input: input, expected: expected)

template runParams*[I, E](suiteName: string, params: openArray[TestParam[I, E]],
                          testFn: proc(input: I): E) =
  ## Run parameterized tests
  suite suiteName:
    for p in params:
      test p.name:
        let result = testFn(p.input)
        check result == p.expected

# ============================================================================
# Validation Test Helpers
# ============================================================================

type
  ValidationCase*[T] = object
    name*: string
    setup*: proc(): T
    valid*: bool

proc validCase*[T](name: string, setup: proc(): T, valid: bool): ValidationCase[T] =
  ValidationCase[T](name: name, setup: setup, valid: valid)

template runValidation*[T](suiteName: string, cases: openArray[ValidationCase[T]],
                            validateFn: proc(obj: T): bool) =
  ## Run validation tests
  suite suiteName:
    for vc in cases:
      test vc.name:
        let obj = vc.setup()
        check validateFn(obj) == vc.valid

# ============================================================================
# Constructor Test Helpers
# ============================================================================

type
  ConstructorCase*[T] = object
    name*: string
    create*: proc(): T
    checks*: seq[tuple[desc: string, check: proc(obj: T): bool]]

template runConstructor*[T](suiteName: string, cases: openArray[ConstructorCase[T]]) =
  ## Run constructor tests with multiple checks per case
  suite suiteName:
    for cc in cases:
      test cc.name:
        let obj = cc.create()
        for chk in cc.checks:
          check chk.check(obj)

# ============================================================================
# Batch Assertion Helpers
# ============================================================================

template checkAll*(checks: varargs[bool]) =
  ## Check multiple conditions at once
  for c in checks:
    check c

template checkFields*[T](obj: T, fields: varargs[tuple[name: string, check: bool]]) =
  ## Check multiple fields of an object
  for f in fields:
    check f.check

# ============================================================================
# Error Test Helpers
# ============================================================================

template expectError*(errorType: typedesc, body: untyped): bool =
  ## Check that code raises expected error
  var caught = false
  try:
    body
  except errorType:
    caught = true
  except CatchableError:
    discard
  caught

template shouldRaise*(body: untyped): bool =
  ## Check that code raises any error
  var caught = false
  try:
    body
  except CatchableError:
    caught = true
  caught

# ============================================================================
# Summary Helpers
# ============================================================================

proc containsAll*(s: string, substrings: varargs[string]): bool =
  ## Check if string contains all substrings
  for sub in substrings:
    if sub notin s:
      return false
  true

proc containsAny*(s: string, substrings: varargs[string]): bool =
  ## Check if string contains any of the substrings
  for sub in substrings:
    if sub in s:
      return true
  false
