"""The command table must map each name to the handler it claims.

Black-box CLI runs cannot catch every mis-wiring: ``hook-check`` and
``hook-guard`` both exit 0 with no output, so swapping them is invisible from
outside.  This test is the only thing that would notice.

It also enforces the one-owner rule: a command that moved into the table must
no longer have an ``elif cmd == "<name>"`` branch left behind in ``cli.py``.
Two owners for one name is a silent behaviour change waiting to happen.
"""
from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

from graphify.commands import TABLE

CMDS_SRC = Path(__file__).parent.parent / "graphify" / "commands" / "__init__.py"

# Grows as branches migrate. A rename that is not reflected here fails loudly,
# which is the point -- the table is the dispatch contract.
EXPECTED_HANDLERS = {
    "prs": "_cmd_prs",
    "benchmark": "_cmd_benchmark",
    "hook": "_cmd_hook",
    "hook-check": "_cmd_hook_check",
    "hook-guard": "_cmd_hook_guard",
    "check-update": "_cmd_check_update",
    "merge-driver": "_cmd_merge_driver",
    "merge-graphs": "_cmd_merge_graphs",
    "merge-chunks": "_cmd_merge_chunks",
    "merge-semantic": "_cmd_merge_semantic",
    "affected": "_cmd_affected",
    "god-nodes": "_cmd_god_nodes",
    "god_nodes": "_cmd_god_nodes",
    "save-result": "_cmd_save_result",
    "reflect": "_cmd_reflect",
    "diagnose": "_cmd_diagnose",
    "add": "_cmd_add",
    "watch": "_cmd_watch",
    "update": "_cmd_update",
    "tree": "_cmd_tree",
    "clone": "_cmd_clone",
    "global": "_cmd_global",
    "cache-check": "_cmd_cache_check",
    "query": "_cmd_query",
    "path": "_cmd_path",
    "explain": "_cmd_explain",
    "cluster-only": "_cmd_cluster_only",
    "label": "_cmd_cluster_only",
    "export": "_cmd_export",
    "provider": "_cmd_provider",
    "extract": "_cmd_extract",
}

CLI_SRC = Path(__file__).parent.parent / "graphify" / "cli.py"


def test_table_maps_each_name_to_its_expected_handler():
    assert {name: fn.__name__ for name, fn in TABLE.items()} == EXPECTED_HANDLERS


def test_handlers_accept_the_command_name():
    # Aliases sharing one branch (cluster-only/label) tell themselves apart by
    # `cmd`, so every handler takes it -- uniformly, including the ones that
    # ignore it.
    import inspect

    for name, fn in TABLE.items():
        params = list(inspect.signature(fn).parameters)
        assert params == ["cmd"], f"{name} -> {fn.__name__}{tuple(params)}"


def test_moved_commands_have_no_branch_left_in_cli():
    src = CLI_SRC.read_text(encoding="utf-8")
    for name in TABLE:
        # Match the dispatch arm only: `if cmd == "x"` / `elif cmd in ("x", ...)`.
        arm = re.search(rf'^    (?:el)?if cmd (?:== "{re.escape(name)}"'
                        rf'|in \([^)]*"{re.escape(name)}")', src, re.M)
        assert arm is None, (
            f"'{name}' is in TABLE but still has a branch in cli.py:"
            f" {arm.group(0) if arm else ''} -- two owners, one command"
        )


def test_every_handler_resolves_all_the_names_it_uses():
    """A moved branch that lost one of its imports is invisible to every other gate.

    The bodies are moved verbatim, so the equivalence check passes; the file
    parses, so the syntax check passes; and most commands exit on an argument
    check long before they touch the helper, so the black-box characterization
    run passes too.  What is left is a ``NameError`` on the first real call.

    That was not hypothetical: ``merge-graphs`` uses ``_GRAPHIFY_OUT`` on line 4
    of 97, and the mover dropped it (2026-08-08).  This is the gate that does not
    depend on a command being reachable with the arguments a test happens to pass.
    """
    tree = ast.parse(CMDS_SRC.read_text(encoding="utf-8"))
    module_level = {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    } | {
        t.id for n in tree.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    } | {
        a.asname or a.name.split(".")[0]
        for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    safe = module_level | set(dir(builtins))

    def params(args: ast.arguments) -> set[str]:
        # Lambdas count: `sorted(..., key=lambda kv: kv[0])` binds `kv`, and
        # missing that reports a correct handler as broken (seen on `explain`).
        named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        named += [a for a in (args.vararg, args.kwarg) if a]
        return {a.arg for a in named}

    unresolved = {}
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        bound = params(fn.args)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
                bound.add(node.name)
                bound |= params(node.args)
            elif isinstance(node, ast.Lambda):
                bound |= params(node.args)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)}
        missing = sorted(used - bound - safe)
        if missing:
            unresolved[fn.name] = missing

    assert not unresolved, (
        f"handlers reference names that are neither local, imported, nor module-level: "
        f"{unresolved} -- a lazy `from graphify.cli import ...` is missing"
    )


def test_table_is_consulted_first():
    # dispatch_command must look the table up before anything else decides where
    # a command goes; otherwise that other thing wins and the table is dead code.
    #
    # This used to compare against `    if cmd == `, the head of the if/elif
    # chain.  Moving `extract` removed the last arm, so that string is gone and
    # the assertion raised ValueError instead of passing -- the test was written
    # for a shape the refactor was designed to destroy.  What is left to protect
    # is the path fallback (`graphify <path>` -> extract).
    src = CLI_SRC.read_text(encoding="utf-8")
    body = src[src.index("def dispatch_command("):]
    lookup = body.index("TABLE.get(cmd)")
    others = [body.index(m) for m in ("    if cmd == ", "Path(cmd).exists()") if m in body]
    assert others, "nothing dispatches after the table -- this test now proves nothing"
    assert lookup < min(others), "TABLE lookup must come before any other dispatch decision"
