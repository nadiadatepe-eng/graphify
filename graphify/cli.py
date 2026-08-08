"""graphify command dispatch — every non-install subcommand.

Extracted verbatim from __main__.main(); __main__ now calls dispatch_command(cmd)
after the install/platform dispatch. Kept out of __main__ to shrink the CLI entry
module. The path-redirect (`graphify <path>` -> extract) re-enters via a lazy
import of main to avoid a cli<->__main__ import cycle.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from graphify.commands import TABLE
from graphify.paths import GRAPHIFY_OUT as _GRAPHIFY_OUT
from pathlib import Path


_SEARCH_NUDGE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'MANDATORY: graphify-out/graph.json exists. You MUST run '
            '`graphify query "<question>"` before grepping raw files. Only grep '
            'after graphify has oriented you, or to modify/debug specific lines.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_READ_NUDGE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'MANDATORY: graphify-out/graph.json exists. You MUST run graphify '
            'before reading source files. Use: `graphify query "<question>"` '
            '(scoped subgraph), `graphify explain "<concept>"`, or '
            '`graphify path "<A>" "<B>"`. Only read raw files after graphify has '
            'oriented you, or to modify/debug specific lines. This rule applies to '
            'subagents too — include it in every subagent prompt involving code '
            'exploration.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_READ_NUDGE_STALE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'graphify-out/graph.json exists but may be STALE for this file (the file '
            'changed after the last build). Prefer `graphify query "<question>"` for '
            'orientation, and run `graphify update` to refresh the graph. Reading the '
            'file directly is fine.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
# Strict-mode block (opt-in). Claude Code PreToolUse honors
# hookSpecificOutput.permissionDecision == "deny" and shows permissionDecisionReason
# to the model. Fires at most once per session (see _mark_session_denied) so it can
# never strand an agent: the very next read proceeds with the soft nudge.
_READ_DENY = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            'graphify strict mode: this project has a fresh knowledge graph that covers '
            'this file. Run `graphify query "<your question>"` (or `graphify explain` / '
            '`graphify path`) FIRST to orient yourself, then re-issue this Read — it '
            'will be allowed. This block fires at most once per session; reading raw '
            'files to modify or debug specific lines is fine after one query. Apply the '
            'same rule in any subagent prompt that explores code.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_HOOK_SOURCE_EXTS = (
    '.py', '.js', '.cjs', '.ts', '.tsx', '.jsx', '.astro', '.vue', '.svelte', '.go',
    '.rs', '.java', '.rb', '.c', '.h', '.cpp', '.hpp', '.cc', '.cs', '.kt',
    '.swift', '.php', '.scala', '.lua', '.sh', '.md', '.rst', '.txt', '.mdx',
)
_GEMINI_NUDGE_TEXT = (
    'graphify: knowledge graph at graphify-out/. For focused questions, run '
    '`graphify query "<question>"` (scoped subgraph, usually much smaller than '
    'GRAPH_REPORT.md) instead of grepping raw files. Read GRAPH_REPORT.md only '
    'for broad architecture context.'
)


def _default_graph_path() -> str:
    return str(Path(_GRAPHIFY_OUT) / "graph.json")


def _stamped_manifest_files(
    files_by_type: dict[str, list[str]],
    sem_result: dict,
    root: Path,
    partial_source_files: "set[str] | None" = None,
) -> dict[str, list[str]]:
    """Manifest-safe files dict: only stamp semantic files that actually
    produced output (cache hit or fresh extraction). Files whose chunk failed
    have no source_file entry in sem_result — leaving their semantic_hash
    empty so detect_incremental re-queues them (#933).

    A file in ``partial_source_files`` DID produce output this run, but only a
    truncated fragment of it, so it is excluded from stamping too — otherwise
    detect_incremental would see it "done" and never re-dispatch it, leaving the
    incomplete node set live forever on the warm-incremental path. Same #933
    mechanism: leave it unstamped and it is re-queued next run.

    Both sides of the membership test are resolved against the scan ``root``
    before comparing (#1897): node/edge/hyperedge ``source_file`` values are
    root-relative on a fresh extraction while ``files_by_type`` entries are
    absolute (from detect()), so a raw string comparison never matched and
    every freshly-extracted semantic doc was dropped from the manifest.
    Mirrors the #1890 path normalization in graphify.llm.

    Hyperedges are counted as output (#1920): a chunk whose only result for a
    document is a hyperedge (3+ nodes sharing a concept) is valid output that
    the semantic cache persists per-``source_file`` — omitting it here left the
    doc unstamped, so detect_incremental re-queued it on every run. The stamping
    condition mirrors the cache-write keying (a hyperedge carries its own
    ``source_file``); do not derive it from member nodes.
    """
    root = Path(root)

    def _resolve(value: str) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        try:
            return p.resolve()
        except (OSError, RuntimeError):
            return p

    sem_extracted: set[Path] = set()
    for coll in ("nodes", "edges", "hyperedges"):
        for item in sem_result.get(coll, []):
            sf = item.get("source_file", "")
            if sf:
                sem_extracted.add(_resolve(sf))
    partial_resolved = {_resolve(p) for p in (partial_source_files or set())}
    sem_types = {"document", "paper", "image"}
    return {
        ftype: [
            f for f in flist
            if ftype not in sem_types
            or (_resolve(f) in sem_extracted and _resolve(f) not in partial_resolved)
        ]
        for ftype, flist in files_by_type.items()
    }


def _stale_graph_sources(
    graph_path: Path,
    scan_root: Path,
    seen_files: set[str],
    detection: dict | None = None,
) -> list[str]:
    """Source files graph.json still references but the current scan no longer
    contains (#1909).

    Incremental extract's prune set was historically derived from the manifest
    alone (``manifest - corpus``), so a file that became EXCLUDED
    (.graphifyignore/.gitignore/--exclude changed) without being listed in the
    manifest kept its stale nodes in graph.json forever. Derive prune
    candidates from the graph's own node ``source_file``s instead: anything
    the graph references that the post-exclude detect corpus no longer
    contains is stale, whether the file was deleted or newly excluded.

    Only IN-ROOT paths are candidates: out-of-root/absolute entries
    (--include sources, symlinked external corpora) are never walked by
    detect, so their absence from the corpus is not staleness evidence.
    Relative entries are re-anchored against both the scan root and the
    graph's own output root; only anchors that land inside the scan root
    count. Since #1941 extracts always store source_file relative to the SCAN
    root, so the scan-root anchor is the live one; the out-root anchor stays
    for graphs written by <=0.9.16, which stored them relative to the OUT root
    (e.g. ``../project/x.py``, #555/#1899).
    ``seen_files`` must be the FULL detect output including unclassified
    files, so nodes from walked-but-unsupported sources (e.g. introspected
    Cargo.toml manifests) are not misread as stale.

    Paths are compared NFC-normalized on both sides: macOS reports NFD
    filenames while graph ``source_file`` entries are typically NFC, and a
    raw-string membership test misread every accented live file as stale
    (#2210; same class as the manifest-layer #2221/#2224).

    Fail-closed liveness guard (#2210, mirrors watch.py's excluded-vs-deleted
    distinction): a source missing from the scan corpus is only pruned when
    the file is gone from disk, or when its exclusion is PROVABLE from the
    same scan that produced ``seen_files`` — ``detection``'s ``ignored`` /
    ``pruned_noise_dirs`` / ``skipped_sensitive`` output, or detect's
    sensitivity predicate. An alive file that merely failed the membership
    test (path-spelling drift the normalization didn't cover, walk errors,
    …) is KEPT and reported, never mass-evicted.
    """
    from graphify.paths import nfc
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    try:
        root_res = scan_root.resolve()
    except (OSError, RuntimeError):
        root_res = scan_root
    # <out>/graphify-out/graph.json — relative source_files may be anchored here.
    out_base = graph_path.parent.parent
    try:
        out_base = out_base.resolve()
    except (OSError, RuntimeError):
        pass

    def _within_root(p: Path) -> bool:
        try:
            p.relative_to(root_res)
            return True
        except ValueError:
            pass
        try:
            p.resolve().relative_to(root_res)
            return True
        except (ValueError, OSError, RuntimeError):
            return False

    seen_nfc = {nfc(s) for s in seen_files}
    seen_basenames = {nfc(os.path.basename(s)) for s in seen_files}

    def _in_seen(p: Path) -> bool:
        if nfc(str(p)) in seen_nfc:
            return True
        try:
            return nfc(str(p.resolve())) in seen_nfc
        except (OSError, RuntimeError):
            return False

    # Provable-exclusion evidence from the scan that produced seen_files:
    # individually ignored files are exact entries; ignored/noise-pruned
    # directories are recorded once with a trailing separator and cover
    # their whole subtree. skipped_sensitive entries may carry a
    # " [reason]" suffix.
    excluded_exact: set[str] = set()
    excluded_prefixes: list[str] = []
    if detection:
        for entry in list(detection.get("ignored", [])) + list(
            detection.get("pruned_noise_dirs", [])
        ):
            e = nfc(str(entry))
            if e.endswith(os.sep) or e.endswith("/"):
                excluded_prefixes.append(e)
            else:
                excluded_exact.add(e)
        for entry in detection.get("skipped_sensitive", []):
            excluded_exact.add(nfc(str(entry).split(" [", 1)[0]))

    def _provably_excluded(c: Path) -> bool:
        spellings = [nfc(str(c))]
        try:
            spellings.append(nfc(str(c.resolve())))
        except (OSError, RuntimeError):
            pass
        for s in spellings:
            if s in excluded_exact:
                return True
            if any(s.startswith(pref) for pref in excluded_prefixes):
                return True
        try:
            from graphify.detect import _is_sensitive as _det_sensitive
            if _det_sensitive(c):
                return True
        except Exception:
            pass
        return False

    stale: list[str] = []
    kept_alive: list[str] = []
    checked: set[str] = set()
    for n in data.get("nodes", []):
        if not isinstance(n, dict):
            continue
        sf = n.get("source_file")
        if not sf or not isinstance(sf, str) or sf in checked:
            continue
        checked.add(sf)
        if "://" in sf:
            continue  # remote/virtual source (e.g. Google Workspace), not a scanned path
        p = Path(sf)
        if p.is_absolute():
            candidates = [p]
        else:
            rel = sf.replace("\\", "/")
            bases = [root_res]
            if out_base != root_res:
                bases.append(out_base)
            candidates = [
                Path(os.path.normpath(str(base / rel))) for base in bases
            ]
        in_root = [c for c in candidates if _within_root(c)]
        if not in_root:
            continue  # out-of-root under every anchor: never prune
        if any(_in_seen(c) for c in in_root):
            continue  # still part of the scan corpus
        # Fail-closed liveness guard (#2210): absence from the corpus is
        # only deletion evidence when the file is actually gone from disk.
        alive = []
        for c in in_root:
            try:
                if c.exists():
                    alive.append(c)
            except OSError:
                pass
        if alive:
            if all(_provably_excluded(c) for c in alive):
                stale.append(sf)  # alive but excluded under current rules (#1909)
            else:
                kept_alive.append(sf)
            continue
        # No anchored candidate exists, but a legacy bare-basename spelling
        # can't be anchored reliably — a live corpus file with the same name
        # means deletion is unproven; keep.
        rel_sf = sf.replace("\\", "/")
        if "/" not in rel_sf and nfc(rel_sf) in seen_basenames:
            kept_alive.append(sf)
            continue
        stale.append(sf)
    if kept_alive:
        print(
            f"[graphify] fail-closed: kept node(s) from {len(kept_alive)} "
            "source file(s) that left the scan corpus but still exist on disk "
            "(ignore rules or filters changed?). Run a full re-extraction to "
            "purge them if the exclusion is intentional.",
            file=sys.stderr,
        )
    return stale


def _prune_graph_json_sources(graph_path: Path, stale_sources: list[str]) -> int:
    """Drop nodes/edges/hyperedges owned by ``stale_sources`` from graph.json
    in place. Returns the number of nodes removed.

    Used by the ``--no-cluster`` incremental early-exit: that path never runs
    ``build_merge`` (it would raw-dump only the new chunks), so an
    exclusion-only change must prune the existing raw graph directly or the
    newly-excluded file's nodes survive forever (#1909).
    ``stale_sources`` comes from :func:`_stale_graph_sources`, i.e. the
    graph's own ``source_file`` spellings, so exact string matching is enough.
    """
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    stale = set(stale_sources)
    links_key = "links" if "links" in data else "edges"
    nodes = [n for n in data.get("nodes", []) if isinstance(n, dict)]
    kept_nodes = [n for n in nodes if n.get("source_file") not in stale]
    removed_ids = {
        n.get("id") for n in nodes if n.get("source_file") in stale
    }
    n_removed = len(nodes) - len(kept_nodes)
    kept_edges = [
        e for e in data.get(links_key, [])
        if isinstance(e, dict)
        and e.get("source_file") not in stale
        and e.get("source") not in removed_ids
        and e.get("target") not in removed_ids
    ]
    kept_hyper = [
        h for h in data.get("hyperedges", [])
        if isinstance(h, dict) and h.get("source_file") not in stale
    ]
    if n_removed == 0 and len(kept_edges) == len(data.get(links_key, [])) and (
        len(kept_hyper) == len(data.get("hyperedges", []))
    ):
        return 0
    data["nodes"] = kept_nodes
    data[links_key] = kept_edges
    if "hyperedges" in data:
        data["hyperedges"] = kept_hyper
    from graphify.export import backup_if_protected as _backup
    _backup(graph_path.parent)
    from graphify.paths import write_json_atomic
    write_json_atomic(graph_path, data, indent=2)
    return n_removed


class _StageTimer:
    """Print per-stage wall-clock timings to stderr when --timing is set (#1490).

    Monotonic (perf_counter), diagnostic-only: emits ``[graphify timing] <stage>:
    N.Ns`` after each stage and a final total. Off by default, so normal output is
    byte-identical and machine-read stdout is untouched.
    """

    def __init__(self, enabled: bool) -> None:
        import time as _time
        self._now = _time.perf_counter
        self.enabled = enabled
        self.start = self._now()
        self._last = self.start

    def mark(self, stage: str) -> None:
        now = self._now()
        if self.enabled:
            print(f"[graphify timing] {stage}: {now - self._last:.1f}s", file=sys.stderr)
        self._last = now

    def total(self) -> None:
        if self.enabled:
            print(f"[graphify timing] total: {self._now() - self.start:.1f}s", file=sys.stderr)
def _enforce_graph_size_cap_or_exit(gp: Path) -> None:
    """Reject oversized graph files before parsing (CLI exit-on-fail flavor).

    Delegates to ``graphify.security.check_graph_file_size_cap`` and turns the
    raised ``ValueError`` into a CLI-style ``error: ...`` message + exit 1.
    Use this from ``__main__.py`` subcommands that already use the ``print +
    sys.exit(1)`` idiom. Library/MCP/loader callers (``serve._load_graph``,
    ``build``, ``benchmark``, ``tree_html``, ``callflow_html``, ``prs``,
    ``global_graph``, ``watch``, ``export``) call the security helper directly
    and let the ``ValueError`` propagate.
    """
    from graphify.security import check_graph_file_size_cap
    try:
        check_graph_file_size_cap(gp)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
def _hook_strict_enabled(flag: bool) -> bool:
    """Resolve strict mode: GRAPHIFY_HOOK_STRICT env overrides the baked-in flag
    (truthy forces on without a reinstall, falsy is the kill switch); unset defers
    to the flag the installed hook command carried."""
    v = os.environ.get("GRAPHIFY_HOOK_STRICT", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return flag


def _touch_query_stamp(graph_path: "Path") -> None:
    """Record that graphify oriented the agent recently, next to the queried graph.
    The strict guard suppresses its block while this stamp is fresh. Fail-silent."""
    try:
        from graphify.paths import write_text_atomic
        stamp = Path(graph_path).parent / "cache" / "last_query_stamp"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(stamp, str(time.time()))
    except Exception:
        pass


def _query_stamp_fresh() -> bool:
    """True if a query/explain/path ran within GRAPHIFY_HOOK_STRICT_TTL (default
    1800s) — recent orientation, so strict mode does not block this read."""
    from graphify.paths import out_path
    try:
        ttl = float(os.environ.get("GRAPHIFY_HOOK_STRICT_TTL", "1800"))
        return (time.time() - out_path("cache", "last_query_stamp").stat().st_mtime) < ttl
    except Exception:
        return False


def _mark_session_denied(session_id: str) -> bool:
    """Atomically claim a one-time strict block for this session. Returns True only
    on the FIRST call for a given session id (O_EXCL create wins once); every later
    call — or any error — returns False, so a session is blocked at most once and an
    agent can never be stranded. Best-effort GC of markers older than 24h."""
    from graphify.paths import out_path
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:64]
    if not sid:
        return False
    try:
        d = out_path("cache", "hook_sessions")
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(d / f"{sid}.denied"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            cutoff = time.time() - 86400
            for entry in os.scandir(d):
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                except OSError:
                    pass
        except OSError:
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _run_hook_guard(kind: str, strict: bool = False) -> None:
    """Shell-agnostic PreToolUse guard (#522).

    Reads the tool-call JSON from stdin and, when a fresh in-project knowledge graph
    exists, nudges the agent to use graphify instead of grepping/reading raw files.
    Replaces the old inline bash hooks that failed to parse on Windows.

    Fails open everywhere: any error, or a non-matching tool call, prints nothing
    and the caller exits 0, so a legitimate tool call is never blocked by a bug.

    In strict mode (opt-in, Claude Code Read only) the FIRST raw read of indexed,
    in-project, fresh code per session is DENIED with a redirect to `graphify query`
    (permissionDecision), then downgrades to the soft nudge — it fires at most once
    per session and can never strand the agent. Search (Bash) and Glob stay
    nudge-only: a compound shell command has no single parseable target and blocking
    file listing would strand navigation. #1840: reads of out-of-project files are
    ignored, and a graph that is stale for the target file softens to a non-mandatory
    nudge instead of blocking or demanding.
    """
    from graphify.paths import out_path, GRAPHIFY_OUT_NAME
    # Gemini's BeforeTool hook takes no stdin and must ALWAYS return a decision so
    # the tool is never blocked; the graph nudge is appended only when a graph
    # exists. Handled before the stdin read below (which the search/read guards need).
    if kind == "gemini":
        payload = {"decision": "allow"}
        try:
            if out_path("graph.json").is_file():
                payload["additionalContext"] = _GEMINI_NUDGE_TEXT
        except Exception:
            pass
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return
    try:
        d = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except Exception:
        return
    if not isinstance(d, dict):
        return
    t = d.get("tool_input", d)
    if not isinstance(t, dict):
        return
    try:
        if kind == "search":
            cmd_str = str(t.get("command", "") or "")
            # Two input shapes reach this guard (matcher "Bash|Grep", #1986):
            # the Bash tool carries `command`, while Claude Code's dedicated
            # Grep tool carries `pattern` (plus optional path/glob) and no
            # command — a Grep call IS a content search by definition, so it
            # nudges whenever a graph exists. For Bash, keep matching the same
            # set the old `case` matched: *grep*, *ripgrep*, and rg/find/fd/
            # ack/ag as a token (name followed by a space). Nudge-only, even in
            # strict mode — see the docstring.
            is_grep_tool = not cmd_str and bool(t.get("pattern"))
            is_bash_search = any(tok in cmd_str for tok in (
                "grep", "ripgrep", "rg ", "find ", "fd ", "ack ", "ag "))
            if (is_grep_tool or is_bash_search) and out_path("graph.json").is_file():
                sys.stdout.write(_SEARCH_NUDGE)
        elif kind == "read":
            vals = [str(t.get("file_path") or ""), str(t.get("pattern") or ""), str(t.get("path") or "")]
            j = " ".join(vals).lower().replace("\\", "/")
            tails = [
                "." + seg.rsplit(".", 1)[-1]
                for v in vals if v
                for seg in [v.lower().replace("\\", "/").rsplit("/", 1)[-1]]
                if "." in seg
            ]
            under_out = "graphify-out/" in j or (GRAPHIFY_OUT_NAME.lower() + "/") in j
            if under_out or not any(tl in _HOOK_SOURCE_EXTS for tl in tails):
                return
            # #1840 (a): skip files outside the graph's project. cwd (or
            # CLAUDE_PROJECT_DIR, which Claude Code sets) is the project root, since
            # the guard only triggers when graph.json exists relative to cwd. A path
            # candidate that resolves outside that root is out-of-project.
            root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
            try:
                root = root.resolve()
            except (OSError, RuntimeError):
                pass
            path_vals = [str(t.get("file_path") or ""), str(t.get("path") or "")]
            explicit = [v for v in path_vals if v]
            if explicit:
                in_project = False
                for v in explicit:
                    p = Path(v)
                    if not p.is_absolute():
                        in_project = True  # relative -> anchored at cwd == in project
                        break
                    try:
                        p.resolve().relative_to(root)
                        in_project = True
                        break
                    except (ValueError, OSError, RuntimeError):
                        continue
                if not in_project:
                    return
            # One stat for existence + mtime of the graph.
            try:
                gmtime = os.stat(str(out_path("graph.json"))).st_mtime
            except OSError:
                return
            # #1840 (b): stale-for-target -> soften, never block. The target file
            # changed after the last build, or watch flagged the tree.
            stale = False
            fp = str(t.get("file_path") or "")
            if fp:
                try:
                    stale = os.stat(fp).st_mtime > gmtime
                except OSError:
                    stale = False
            try:
                if out_path("needs_update").exists():
                    stale = True
            except Exception:
                pass
            if stale:
                sys.stdout.write(_READ_NUDGE_STALE)
                return
            # Strict block: Read tool only, first time per session, not recently
            # oriented, and the file is demonstrably indexed.
            tool_name = d.get("tool_name")
            if _hook_strict_enabled(strict) and tool_name in (None, "Read") \
                    and not _query_stamp_fresh() \
                    and _target_is_indexed(fp, root) \
                    and _mark_session_denied(str(d.get("session_id") or "")):
                sys.stdout.write(_READ_DENY)
                return
            sys.stdout.write(_READ_NUDGE)
    except Exception:
        pass


def _target_is_indexed(file_path: str, root: "Path") -> bool:
    """Guard the strict deny: only block a read of a file the graph actually indexes.
    Reads manifest.json (cheap, capped); on any doubt (missing/corrupt/oversized
    manifest, unresolvable path) returns True so the once-per-session deny still
    applies — that block is self-limiting, so erring toward it is safe."""
    from graphify.paths import out_path
    if not file_path:
        return True
    try:
        mp = out_path("manifest.json")
        st = mp.stat()
        if st.st_size > 2_000_000:
            return True
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not manifest:
            return True
        p = Path(file_path)
        rels = set()
        try:
            rels.add(p.resolve().relative_to(root).as_posix())
        except (ValueError, OSError, RuntimeError):
            pass
        rels.add(p.name)
        keys = {str(k).replace("\\", "/") for k in manifest}
        abskey = str(p).replace("\\", "/")
        return abskey in keys or any(r and (r in keys or any(k.endswith("/" + r) or k == r for k in keys)) for r in rels)
    except Exception:
        return True
def _clone_repo(
    url: str, branch: str | None = None, out_dir: Path | None = None
) -> Path:
    """Clone a GitHub repo to a local cache dir and return the path.

    Clones into ~/.graphify/repos/<owner>/<repo> by default so repeated
    runs on the same URL reuse the existing clone (git pull instead of clone).
    """
    import subprocess as _sp
    import re as _re

    # Normalise URL — strip trailing .git if present
    url = url.rstrip("/")
    if not url.endswith(".git"):
        git_url = url + ".git"
    else:
        git_url = url
        url = url[:-4]

    # Extract owner/repo from URL
    m = _re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
    if not m:
        print(f"error: not a recognised GitHub URL: {url}", file=sys.stderr)
        sys.exit(1)
    owner, repo = m.group(1), m.group(2)

    if out_dir:
        dest = out_dir
    else:
        dest = Path.home() / ".graphify" / "repos" / owner / repo

    if branch and branch.startswith("-"):
        print(f"error: invalid branch name: {branch!r}", file=sys.stderr)
        sys.exit(1)

    if dest.exists():
        print(f"Repo already cloned at {dest} - pulling latest...", flush=True)
        cmd = ["git", "-C", str(dest), "pull"]
        if branch:
            cmd += ["origin", "--", branch]
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"warning: git pull failed:\n{result.stderr}", file=sys.stderr)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url} -> {dest} ...", flush=True)
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += ["--", git_url, str(dest)]
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"error: git clone failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    print(f"Ready at: {dest}", flush=True)
    return dest


def _reenter_main() -> None:
    from graphify.__main__ import main
    main()


def dispatch_command(cmd: str) -> None:
    handler = TABLE.get(cmd)
    if handler is not None:
        handler(cmd)
        return
    if Path(cmd).exists() or cmd in (".", "..") or cmd.startswith(("./", "../", "/", "~")):
        # User ran `graphify <path>` directly — treat as `graphify extract <path>`.
        # Common when following the PowerShell note in README (`graphify .`) or
        # copy-pasting skill invocations without the leading slash.
        sys.argv.insert(2, sys.argv[1])
        sys.argv[1] = "extract"
        _reenter_main()
    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        print("Run 'graphify --help' for usage.", file=sys.stderr)
        sys.exit(1)
