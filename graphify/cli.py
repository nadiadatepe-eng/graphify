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
    if cmd == "extract":
        # Headless full-pipeline extraction for CI / scripts (#698).
        # Runs detect -> AST extraction on code -> semantic LLM extraction on
        # docs/papers/images -> merge -> build -> cluster -> write outputs.
        # Unlike the skill.md path (which runs through Claude Code subagents),
        # this calls extract_corpus_parallel directly using whichever backend
        # has an API key set.
        if len(sys.argv) < 3:
            print(
                "Usage: graphify extract <path> [--backend gemini|kimi|claude|openai|deepseek|ollama] "
                "[--model M] [--mode deep] [--out DIR|--output DIR] [--google-workspace] [--no-cluster] "
                "[--no-gitignore] [--code-only] "
                "[--max-workers N] [--token-budget N] [--max-concurrency N] "
                "[--api-timeout S] [--postgres DSN] [--cargo] [--allow-partial] [--timing]",
                file=sys.stderr,
            )
            sys.exit(1)

        has_path = True
        if sys.argv[2].startswith("-"):
            has_path = False
            target = Path(".").resolve()
        else:
            target = Path(sys.argv[2]).resolve()
            if not target.exists():
                print(f"error: path not found: {target}", file=sys.stderr)
                sys.exit(1)

        backend: str | None = None
        model: str | None = None
        extract_mode: str | None = None
        out_dir: Path | None = None
        cli_postgres_dsn: str | None = None
        cli_cargo: bool = False
        cli_allow_partial: bool = False
        no_cluster = False
        dedup_llm = False
        google_workspace = False
        global_merge = False
        code_only = False
        no_gitignore = False
        global_repo_tag: str | None = None
        # Performance/tuning knobs (issue #792). None means "use library default".
        cli_max_workers: int | None = None
        cli_token_budget: int | None = None
        cli_max_concurrency: int | None = None
        cli_api_timeout: float | None = None
        # Clustering tuning knobs
        cli_resolution: float = 1.0
        cli_exclude_hubs: float | None = None
        cli_excludes: list[str] = []
        cli_timing: bool = False
        # --force parity with `graphify update`: the flag or GRAPHIFY_FORCE=1
        # disables the incremental gate and skips semantic-cache reads (#1894).
        force = os.environ.get("GRAPHIFY_FORCE", "").lower() in ("1", "true", "yes")

        def _parse_int(name: str, raw: str) -> int:
            try:
                v = int(raw)
            except ValueError:
                print(f"error: {name} must be a positive integer (got {raw!r})", file=sys.stderr)
                sys.exit(2)
            if v <= 0:
                print(f"error: {name} must be > 0 (got {v})", file=sys.stderr)
                sys.exit(2)
            return v

        def _parse_float(name: str, raw: str) -> float:
            try:
                v = float(raw)
            except ValueError:
                print(f"error: {name} must be a positive number (got {raw!r})", file=sys.stderr)
                sys.exit(2)
            if v <= 0:
                print(f"error: {name} must be > 0 (got {v})", file=sys.stderr)
                sys.exit(2)
            return v

        args = sys.argv[3:] if has_path else sys.argv[2:]
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--backend" and i + 1 < len(args):
                backend = args[i + 1]; i += 2
            elif a.startswith("--backend="):
                backend = a.split("=", 1)[1]; i += 1
            elif a == "--model" and i + 1 < len(args):
                model = args[i + 1]; i += 2
            elif a.startswith("--model="):
                model = a.split("=", 1)[1]; i += 1
            elif a == "--mode" and i + 1 < len(args):
                extract_mode = args[i + 1]; i += 2
            elif a.startswith("--mode="):
                extract_mode = a.split("=", 1)[1]; i += 1
            elif a in ("--out", "--output") and i + 1 < len(args):
                # --output is an alias of --out (#2004): it was silently dropped
                # before, and `graphify tree` already documents --output, so the
                # mistake is natural. (--output= does not startswith --out=.)
                out_dir = Path(args[i + 1]); i += 2
            elif a.startswith(("--out=", "--output=")):
                out_dir = Path(a.split("=", 1)[1]); i += 1
            elif a == "--no-cluster":
                no_cluster = True; i += 1
            elif a == "--dedup-llm":
                dedup_llm = True; i += 1
            elif a == "--code-only":
                code_only = True; i += 1
            elif a == "--google-workspace":
                google_workspace = True; i += 1
            elif a == "--no-gitignore":
                no_gitignore = True; i += 1
            elif a == "--global":
                global_merge = True; i += 1
            elif a == "--as" and i + 1 < len(args):
                global_repo_tag = args[i + 1]; i += 2
            elif a == "--max-workers" and i + 1 < len(args):
                cli_max_workers = _parse_int("--max-workers", args[i + 1]); i += 2
            elif a.startswith("--max-workers="):
                cli_max_workers = _parse_int("--max-workers", a.split("=", 1)[1]); i += 1
            elif a == "--token-budget" and i + 1 < len(args):
                cli_token_budget = _parse_int("--token-budget", args[i + 1]); i += 2
            elif a.startswith("--token-budget="):
                cli_token_budget = _parse_int("--token-budget", a.split("=", 1)[1]); i += 1
            elif a == "--max-concurrency" and i + 1 < len(args):
                cli_max_concurrency = _parse_int("--max-concurrency", args[i + 1]); i += 2
            elif a.startswith("--max-concurrency="):
                cli_max_concurrency = _parse_int("--max-concurrency", a.split("=", 1)[1]); i += 1
            elif a == "--api-timeout" and i + 1 < len(args):
                cli_api_timeout = _parse_float("--api-timeout", args[i + 1]); i += 2
            elif a.startswith("--api-timeout="):
                cli_api_timeout = _parse_float("--api-timeout", a.split("=", 1)[1]); i += 1
            elif a == "--resolution" and i + 1 < len(args):
                cli_resolution = _parse_float("--resolution", args[i + 1]); i += 2
            elif a.startswith("--resolution="):
                cli_resolution = _parse_float("--resolution", a.split("=", 1)[1]); i += 1
            elif a == "--exclude-hubs" and i + 1 < len(args):
                cli_exclude_hubs = float(args[i + 1]); i += 2
            elif a.startswith("--exclude-hubs="):
                cli_exclude_hubs = float(a.split("=", 1)[1]); i += 1
            elif a == "--exclude" and i + 1 < len(args):
                cli_excludes.append(args[i + 1]); i += 2
            elif a.startswith("--exclude="):
                cli_excludes.append(a.split("=", 1)[1]); i += 1
            elif a == "--postgres" and i + 1 < len(args):
                cli_postgres_dsn = args[i + 1]; i += 2
            elif a.startswith("--postgres="):
                cli_postgres_dsn = a.split("=", 1)[1]; i += 1
            elif a == "--cargo":
                cli_cargo = True
                i += 1
            elif a == "--force":
                force = True; i += 1
            elif a == "--allow-partial":
                cli_allow_partial = True; i += 1
            elif a == "--timing":
                cli_timing = True; i += 1
            else:
                i += 1

        if not has_path and cli_postgres_dsn is None:
            print("error: must specify a path to scan or a --postgres DSN", file=sys.stderr)
            sys.exit(1)

        _VALID_MODES = {"deep"}
        if extract_mode is not None and extract_mode not in _VALID_MODES:
            print(
                f"error: unknown --mode '{extract_mode}'. "
                f"Available: {', '.join(sorted(_VALID_MODES))}",
                file=sys.stderr,
            )
            sys.exit(2)
        deep_mode = extract_mode == "deep"
        if deep_mode:
            print("[graphify extract] deep mode enabled: richer semantic extraction")

        # CLI flag wins over env var. Setting GRAPHIFY_API_TIMEOUT here so
        # _call_openai_compat picks it up without needing a new kwarg path.
        if cli_api_timeout is not None:
            os.environ["GRAPHIFY_API_TIMEOUT"] = str(cli_api_timeout)
        if cli_max_workers is not None:
            os.environ["GRAPHIFY_MAX_WORKERS"] = str(cli_max_workers)

        # Resolve output dir. The user-facing contract is "<out>/graphify-out/"
        # so a fresh checkout writes graphify-out/ at the project root, matching
        # the skill.md pipeline.
        out_root = (out_dir.resolve() if out_dir else target)
        graphify_out = out_root / _GRAPHIFY_OUT
        graphify_out.mkdir(parents=True, exist_ok=True)
        # Persist corpus-shaping options so later update/watch/hook rebuilds
        # use the same file set as the initial extraction (#1886).
        from graphify.watch import (
            _write_build_config as _write_build_cfg,
            _read_build_excludes as _read_build_ex,
            _read_build_gitignore as _read_build_gi,
        )
        # #1971 persistence: an explicit --no-gitignore persists False; a later
        # flag-less `graphify extract` must NOT clobber it back to True, which
        # would make the git-ignored code silently disappear again (the exact
        # complaint #1971 is about). Honor the persisted value for THIS run when
        # the flag is absent (read before the write below), and write False only
        # when the flag is set — None leaves the setting as-is, mirroring how
        # #1886 persists --exclude.
        _effective_gitignore = False if no_gitignore else _read_build_gi(graphify_out)
        # An explicit list replaces the persisted one; omission reuses it.
        _effective_excludes = cli_excludes or _read_build_ex(graphify_out)
        _write_build_cfg(
            graphify_out,
            excludes=cli_excludes or None,
            gitignore=False if no_gitignore else None,
        )

        stages = _StageTimer(cli_timing)

        from graphify.detect import (
            detect as _detect,
            detect_incremental as _detect_incremental,
            save_manifest as _save_manifest,
        )
        manifest_path = graphify_out / "manifest.json"
        existing_graph_path = graphify_out / "graph.json"
        # #1925: a missing manifest.json must not degrade to a full scan that
        # discards the existing graph's semantic layer. An existing graph.json
        # is a sufficient incremental baseline: detect_incremental treats an
        # absent manifest as "everything is new" (re-extract all, nothing
        # deleted), and build_merge + _stale_graph_sources reconcile replaced
        # and genuinely-deleted sources against the current corpus, so doc/
        # paper/image nodes survive a --code-only rebuild instead of being
        # dropped with the rest of the committed graph.
        incremental_mode = existing_graph_path.exists() if has_path else False
        # --force: full scan, not the manifest-gated incremental diff — a warm
        # unchanged tree would otherwise dispatch zero files (#1894).
        incremental_mode = incremental_mode and not force
        if force:
            print("[graphify extract] --force: full re-scan, semantic cache reads skipped")
        elif incremental_mode and not manifest_path.exists():
            print(
                "[graphify extract] manifest.json missing; using existing "
                "graph.json as the incremental baseline (all files re-checked; "
                "nodes for files outside this run's scope are preserved)"
            )

        if not has_path:
            detection = {}
            code_files = []
            doc_files = []
            paper_files = []
            image_files = []
            deleted_files = []
            excluded_files = []
            graph_stale_sources = []
            unchanged_total = 0
            files_by_type = {}
        elif incremental_mode:
            print(f"[graphify extract] incremental scan of {target}")
            detection = _detect_incremental(
                target,
                manifest_path=str(manifest_path),
                google_workspace=google_workspace or None,
                extra_excludes=_effective_excludes or None,
                gitignore=_effective_gitignore,
            )
            files_by_type = detection.get("files", {})
            new_by_type = detection.get("new_files", {})
            code_files = [Path(p) for p in new_by_type.get("code", [])]
            doc_files = [Path(p) for p in new_by_type.get("document", [])]
            paper_files = [Path(p) for p in new_by_type.get("paper", [])]
            image_files = [Path(p) for p in new_by_type.get("image", [])]
            deleted_files = list(detection.get("deleted_files", []))
            excluded_files = list(detection.get("excluded_files", []))
            unchanged_total = sum(len(v) for v in detection.get("unchanged_files", {}).values())
            # #1909: derive the prune set from the existing graph itself, not
            # just the manifest. A file that became excluded without ever
            # being manifest-listed (every pre-#1897 graph is in this state)
            # still has stale nodes carried forward by build_merge unless the
            # graph's own sources are reconciled against the current corpus.
            _seen_files = {f for _fl in files_by_type.values() for f in _fl}
            _seen_files.update(detection.get("unclassified", []))
            graph_stale_sources = _stale_graph_sources(
                existing_graph_path, target, _seen_files, detection=detection
            )
        else:
            print(f"[graphify extract] scanning {target}")
            detection = _detect(
                target,
                google_workspace=google_workspace or None,
                extra_excludes=_effective_excludes or None,
                cache_root=out_root,
                gitignore=_effective_gitignore,
            )
            files_by_type = detection.get("files", {})
            code_files = [Path(p) for p in files_by_type.get("code", [])]
            doc_files = [Path(p) for p in files_by_type.get("document", [])]
            paper_files = [Path(p) for p in files_by_type.get("paper", [])]
            image_files = [Path(p) for p in files_by_type.get("image", [])]
            deleted_files = []
            excluded_files = []
            graph_stale_sources = []
            unchanged_total = 0

        semantic_files = doc_files + paper_files + image_files
        # --code-only: index code (pure local AST, no key) and skip the semantic
        # (doc/paper/image) pass entirely, so a mixed repo doesn't hard-fail when no
        # LLM backend is configured (#1734). Report what was skipped rather than
        # silently dropping it.
        if code_only and semantic_files:
            print(
                f"[graphify extract] --code-only: skipping {len(semantic_files)} "
                f"non-code file(s) ({len(doc_files)} docs, {len(paper_files)} papers, "
                f"{len(image_files)} images) — no LLM extraction"
            )
            semantic_files = []
            doc_files = []
            paper_files = []
            image_files = []
        if deep_mode and incremental_mode and not code_only:
            # Deep mode reads/writes its own cache namespace
            # (cache/semantic-deep/), so the manifest's changed-file gate is
            # not a valid proxy for deep coverage: over a warm unchanged tree
            # it dispatches zero files and `--mode deep` silently no-ops
            # (#1894). Widen the semantic pass to the FULL live
            # doc/paper/image set (``files_by_type`` from detect_incremental,
            # which already excludes excluded files) and let the
            # mode-namespaced cache decide hits/misses — the first deep run
            # re-dispatches everything (deep namespace cold), later deep runs
            # hit the deep cache.
            _deep_all = [
                Path(p)
                for _ftype in ("document", "paper", "image")
                for p in files_by_type.get(_ftype, [])
            ]
            if len(_deep_all) != len(semantic_files):
                print(
                    f"[graphify extract] deep mode: widening semantic pass from "
                    f"{len(semantic_files)} changed to {len(_deep_all)} live "
                    f"doc/paper/image file(s); the deep semantic cache decides "
                    f"what is re-extracted"
                )
            semantic_files = _deep_all
        if incremental_mode:
            # Excluded-but-alive files are reported separately from deletions
            # (#1908): they still exist on disk, the scan just stopped
            # covering them (ignore rules / --exclude changed).
            _excl_note = f"; {len(excluded_files)} excluded" if excluded_files else ""
            print(
                f"[graphify extract] {len(code_files)} code, {len(doc_files)} docs, "
                f"{len(paper_files)} papers, {len(image_files)} images changed; "
                f"{unchanged_total} unchanged; {len(deleted_files)} deleted"
                f"{_excl_note}"
            )
        else:
            print(
                f"[graphify extract] found {len(code_files)} code, "
                f"{len(doc_files)} docs, {len(paper_files)} papers, "
                f"{len(image_files)} images"
            )
        # Surface files that were seen but not classified (extensionless non-shebang
        # project files like Dockerfile/Makefile, or unsupported extensions), so they
        # are no longer invisible in graphify's own output (#1692).
        _unclassified = detection.get("unclassified", []) if isinstance(detection, dict) else []
        if _unclassified:
            _names = ", ".join(sorted({Path(p).name for p in _unclassified})[:6])
            _more = f" (+{len(_unclassified) - 6} more)" if len(_unclassified) > 6 else ""
            print(
                f"[graphify extract] {len(_unclassified)} file(s) not classified "
                f"(no supported extension or shebang), skipped: {_names}{_more}"
            )
        # Name the files dropped by the sensitive-file filter so a wrongly-flagged
        # source/doc is visible, not just a count (#2106). Operational skips
        # (symlink/office/Workspace) carry a " [reason]" suffix; exclude those here
        # so this line reports only the security-heuristic drops.
        _sensitive = detection.get("skipped_sensitive", []) if isinstance(detection, dict) else []
        _sec = [s for s in _sensitive if " [" not in s]
        if _sec:
            _snames = ", ".join(sorted({Path(p).name for p in _sec})[:6])
            _smore = f" (+{len(_sec) - 6} more)" if len(_sec) > 6 else ""
            print(
                f"[graphify extract] {len(_sec)} file(s) skipped as potentially sensitive "
                f"(rename or move if wrongly flagged): {_snames}{_smore}"
            )
        stages.mark("detect")

        # Resolve the LLM backend only now that we know whether the corpus
        # needs one. A code-only corpus is pure local AST and must not require
        # an API key; the key is enforced below only when there's LLM work.
        from graphify.llm import (
            BACKENDS as _BACKENDS,
            detect_backend as _detect_backend,
            estimate_cost as _estimate_cost,
            extract_corpus_parallel as _extract_corpus_parallel,
            _format_backend_env_keys,
            _get_backend_api_key,
        )
        needs_llm = bool(semantic_files) or dedup_llm
        if backend is None and needs_llm:
            backend = _detect_backend()
        if backend is not None and backend not in _BACKENDS:
            print(
                f"error: unknown backend '{backend}'. "
                f"Available: {', '.join(sorted(_BACKENDS))}",
                file=sys.stderr,
            )
            sys.exit(1)
        if needs_llm:
            if backend is None:
                reasons = []
                if semantic_files:
                    reasons.append(
                        f"{len(semantic_files)} doc/paper/image file(s) need semantic extraction"
                    )
                if dedup_llm:
                    reasons.append("--dedup-llm was passed")
                hint = ""
                if semantic_files:
                    hint = (" Or pass --code-only to index just the code "
                            "(local AST, no key) and skip the non-code files.")
                print(
                    "error: no LLM API key found (" + "; ".join(reasons) + "). "
                    "Set GEMINI_API_KEY or GOOGLE_API_KEY (gemini), MOONSHOT_API_KEY "
                    "(kimi), ANTHROPIC_API_KEY (claude), OPENAI_API_KEY (openai), "
                    "DEEPSEEK_API_KEY (deepseek), or pass --backend. A code-only "
                    "corpus needs no key." + hint,
                    file=sys.stderr,
                )
                sys.exit(1)
            if backend == "ollama":
                from graphify.llm import _validate_ollama_base_url
                _oll_url = os.environ.get("OLLAMA_BASE_URL", _BACKENDS["ollama"].get("base_url", ""))
                try:
                    _validate_ollama_base_url(_oll_url, warn=False)
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    sys.exit(2)
            if not _get_backend_api_key(backend):
                allow_no_key = False
                if backend == "ollama":
                    from urllib.parse import urlparse
                    ollama_url = os.environ.get(
                        "OLLAMA_BASE_URL",
                        _BACKENDS["ollama"].get("base_url", ""),
                    )
                    try:
                        host = (urlparse(ollama_url).hostname or "").lower()
                    except Exception:
                        host = ""
                    allow_no_key = (
                        host in ("localhost", "127.0.0.1", "::1")
                        or host.startswith("127.")
                    )
                elif backend == "bedrock":
                    allow_no_key = bool(
                        os.environ.get("AWS_PROFILE")
                        or os.environ.get("AWS_REGION")
                        or os.environ.get("AWS_DEFAULT_REGION")
                        or os.environ.get("AWS_ACCESS_KEY_ID")
                    )
                elif backend == "claude-cli":
                    import shutil as _shutil
                    allow_no_key = _shutil.which("claude") is not None
                    if not allow_no_key:
                        print(
                            "error: backend 'claude-cli' requires the `claude` CLI on $PATH "
                            "(install Claude Code and run `claude` once to authenticate).",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                if not allow_no_key:
                    print(
                        f"error: backend '{backend}' requires {_format_backend_env_keys(backend)} to be set.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        # Track whether this run's extraction was incomplete (a whole extractor
        # pass crashed, or some semantic chunks failed). A partial result must not
        # be force-written over a good complete graph — the final write falls back
        # to the #479 shrink guard unless --allow-partial is set.
        _extraction_incomplete = False
        # A walk that couldn't fully enumerate the corpus (permission-denied
        # subtree, I/O error) yields a legitimately smaller graph that must not
        # be force-written over a complete one — same failure class as a crashed
        # pass. detect()/detect_incremental() already record these; consume them.
        if detection.get("walk_errors"):
            _extraction_incomplete = True

        # AST extraction on code files. Empty code list (docs-only corpus) is
        # the issue #698 case — skip cleanly instead of crashing inside extract().
        ast_result: dict = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}
        if code_files:
            from graphify.extract import extract as _ast_extract
            # Anchor the cache at the output root, not the scanned project:
            # with --out, a <target>/graphify-out/cache/ would leak a
            # graphify-out/ dir into a project that asked for external output.
            # `root` stays the scanned project so source_file/ids relativize
            # against it; conflating the two basenamed every node (#1941).
            ast_kwargs: dict = {"cache_root": out_root, "root": target}
            if cli_max_workers is not None:
                ast_kwargs["max_workers"] = cli_max_workers
            # #2437/#2438 (the `graphify update` twin of watch's #2406 fix): an
            # incremental re-scan extracts only the changed code files, so the
            # cross-file resolvers cannot see a callee living in an unchanged
            # file and every changed->unchanged call edge silently vanished on
            # merge. Hand extract() read-only resolution context from the
            # persisted graph: its AST-tier nodes (with their `_callable`/
            # `_callable_class` markers, #2438) plus the contains/method edges
            # the member-call resolvers walk (#2437), scoped to the UNCHANGED
            # live corpus — never a re-extracted, deleted, or excluded file, so
            # stale symbols cannot resurrect. Fails open (changed-batch-only
            # resolution, the pre-fix behavior) on an unreadable graph.
            if incremental_mode and existing_graph_path.exists():
                _ctx_nodes: list[dict] = []
                _ctx_edges: list[dict] = []
                try:
                    from graphify.build import _is_ast_tier as _ctx_is_ast_tier
                    from graphify.security import (
                        check_graph_file_size_cap as _ctx_size_cap,
                    )
                    _ctx_size_cap(existing_graph_path)
                    _ctx_graph = json.loads(
                        existing_graph_path.read_text(encoding="utf-8")
                    )
                    _ctx_root = Path(os.path.abspath(target))

                    def _ctx_identity(source_file) -> str | None:
                        # graph.json source_file values are relative to the
                        # scanned root (`root=target` above); detect's
                        # unchanged_files keep their scan-time form. Compare
                        # both as absolute posix paths.
                        if not source_file:
                            return None
                        _p = Path(str(source_file))
                        if not _p.is_absolute():
                            _p = _ctx_root / _p
                        return Path(os.path.abspath(_p)).as_posix()

                    _ctx_live = {
                        _ctx_identity(f)
                        for _flist in detection.get("unchanged_files", {}).values()
                        for f in _flist
                    }
                    _ctx_live.discard(None)
                    for _node in _ctx_graph.get("nodes", []):
                        if not _node.get("id") or not _ctx_is_ast_tier(_node):
                            continue
                        _sf = _node.get("source_file")
                        if not _sf or _ctx_identity(_sf) not in _ctx_live:
                            continue
                        _ctx_node = {
                            "id": _node["id"],
                            "label": _node.get("label"),
                            "source_file": _sf,
                            "file_type": _node.get("file_type"),
                            "type": _node.get("type"),
                        }
                        for _marker in ("_callable", "_callable_class"):
                            if _node.get(_marker):
                                _ctx_node[_marker] = _node[_marker]
                        _ctx_nodes.append(_ctx_node)
                    for _edge in _ctx_graph.get(
                        "links", _ctx_graph.get("edges", [])
                    ):
                        if _edge.get("relation") not in ("contains", "method"):
                            continue
                        if not _ctx_is_ast_tier(_edge):
                            continue
                        _sf = _edge.get("source_file")
                        if not _sf or _ctx_identity(_sf) not in _ctx_live:
                            continue
                        _ctx_edges.append({
                            "source": _edge.get("source"),
                            "target": _edge.get("target"),
                            "relation": _edge.get("relation"),
                            "source_file": _sf,
                        })
                except Exception:
                    _ctx_nodes, _ctx_edges = [], []
                if _ctx_nodes:
                    ast_kwargs["resolution_context_nodes"] = _ctx_nodes
                if _ctx_edges:
                    ast_kwargs["resolution_context_edges"] = _ctx_edges
            print(f"[graphify extract] AST extraction on {len(code_files)} code files...")
            try:
                ast_result = _ast_extract(code_files, **ast_kwargs)
            except Exception as exc:
                print(f"[graphify extract] AST extraction failed: {exc}", file=sys.stderr)
                # #2445: losing the whole AST pass is fatal by default. The
                # empty stand-in only reaches the shrink guard when an existing
                # graph is larger — on a fresh build it used to be written as a
                # 0-node graph with exit 0, indistinguishable from success.
                # --allow-partial opts back into the best-effort continuation.
                if not cli_allow_partial:
                    sys.exit(1)
                ast_result = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}
                _extraction_incomplete = True  # the whole AST pass was lost
        stages.mark("AST extract")

        # Semantic extraction on docs/papers/images. Check cache first.
        from graphify.cache import (
            check_semantic_cache as _check_semantic_cache,
            prune_semantic_cache as _prune_semantic_cache,
            save_semantic_cache as _save_semantic_cache,
        )
        sem_result: dict = {
            "nodes": [], "edges": [], "hyperedges": [],
            "input_tokens": 0, "output_tokens": 0,
        }
        # Semantic files whose extraction truncated this run. They are left
        # unstamped in the manifest so detect_incremental re-queues them next run
        # (mirrors the #933 failed-chunk handling); captured below before the
        # _partial markers are stripped from the corpus.
        _partial_semantic_files: set[str] = set()
        sem_cache_hits = 0
        sem_cache_misses = 0
        # Deep mode uses its own namespace (cache/semantic-deep/) so deep and
        # standard results for the same content never shadow each other (#1894).
        sem_cache_mode = "deep" if deep_mode else None
        # Entries are attributed to the extraction prompt that produced them, so
        # a release that changes the prompt re-extracts rather than replaying the
        # older vintage alongside the new one (#1939). Read and write must pass
        # the same prompt, or the write lands where the next read won't look.
        from graphify.llm import _extraction_system as _sem_prompt_for
        sem_prompt = _sem_prompt_for(deep=deep_mode)
        if semantic_files:
            sem_paths_str = [str(p) for p in semantic_files]
            if force:
                # --force: skip the cache READ so every semantic file is
                # re-dispatched; the save below still runs so the fresh
                # results replace the stale entries.
                cached_nodes, cached_edges, cached_hyperedges = [], [], []
                uncached_paths = list(sem_paths_str)
            else:
                cached_nodes, cached_edges, cached_hyperedges, uncached_paths = (
                    _check_semantic_cache(sem_paths_str, root=target, cache_root=out_root,
                                          mode=sem_cache_mode, prompt=sem_prompt)
                )
            sem_cache_hits = len(semantic_files) - len(uncached_paths)
            sem_cache_misses = len(uncached_paths)
            sem_result["nodes"].extend(cached_nodes)
            sem_result["edges"].extend(cached_edges)
            sem_result["hyperedges"].extend(cached_hyperedges)
            if sem_cache_hits:
                print(f"[graphify extract] semantic cache: {sem_cache_hits} hit / {sem_cache_misses} miss")

            if uncached_paths:
                print(f"[graphify extract] semantic extraction on {len(uncached_paths)} files via {backend}...")
                corpus_kwargs: dict = {
                    "backend": backend,
                    "model": model,
                    "root": target,
                    "cache_root": out_root,
                }
                if deep_mode:
                    corpus_kwargs["deep_mode"] = True
                if cli_token_budget is not None:
                    corpus_kwargs["token_budget"] = cli_token_budget
                if cli_max_concurrency is not None:
                    corpus_kwargs["max_concurrency"] = cli_max_concurrency

                # Minimal progress callback so the CLI is no longer silent
                # during long local-inference runs (issue #792 addendum).
                # Also track per-chunk success so we can fail loudly when
                # every chunk errors (e.g. missing backend SDK package).
                _chunk_stats = {"total": 0, "succeeded": 0}
                def _progress(idx: int, total: int, _result: dict) -> None:
                    _chunk_stats["total"] = total
                    _chunk_stats["succeeded"] += 1
                    print(
                        f"[graphify extract] chunk {idx + 1}/{total} done",
                        flush=True,
                    )
                corpus_kwargs["on_chunk_done"] = _progress

                try:
                    fresh = _extract_corpus_parallel(
                        [Path(p) for p in uncached_paths],
                        **corpus_kwargs,
                    )
                except ImportError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    sys.exit(1)
                except Exception as exc:
                    print(
                        f"[graphify extract] semantic extraction failed: {exc}",
                        file=sys.stderr,
                    )
                    fresh = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
                    _extraction_incomplete = True  # the semantic pass crashed

                # on_chunk_done only fires after a chunk succeeds. If fresh
                # semantic extraction was requested and no chunks completed,
                # fail instead of writing an AST-only graph with exit 0.
                if uncached_paths and _chunk_stats["succeeded"] == 0:
                    print(
                        f"[graphify extract] error: all semantic chunks failed "
                        f"for backend '{backend}' ({len(uncached_paths)} uncached files) - "
                        f"see per-chunk errors above. If you see 'requires the X package', "
                        f"run `pip install X` and retry.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                # Some (but not all) chunks failed — the graph is missing nodes
                # from the failed chunks, so it must not clobber a larger complete
                # graph without an explicit --allow-partial override.
                if _chunk_stats["total"] and _chunk_stats["succeeded"] < _chunk_stats["total"]:
                    _extraction_incomplete = True
                # Which files truncated this run (item markers + the empty-parse
                # _partial_files set). Computed BEFORE the save so it can be passed
                # as partial_source_files: without it, a file whose only truncated
                # chunk parsed empty (so it has no item markers here) would be
                # written as a complete cache entry, re-promoting it (#1950).
                from graphify.llm import (
                    _partial_source_files as _partial_sf,
                    _strip_partial_markers as _strip_partial,
                )
                _partial_semantic_files = set(_partial_sf(fresh))
                try:
                    _save_semantic_cache(
                        fresh.get("nodes", []),
                        fresh.get("edges", []),
                        fresh.get("hyperedges", []),
                        root=target,
                        cache_root=out_root,
                        allowed_source_files=uncached_paths,
                        mode=sem_cache_mode,
                        prompt=sem_prompt,
                        partial_source_files=_partial_semantic_files or None,
                    )
                except Exception as exc:
                    print(f"[graphify extract] warning: could not write semantic cache: {exc}", file=sys.stderr)
                # Strip the markers before the corpus feeds the graph so the
                # internal flag never leaks into graph.json.
                _strip_partial(fresh)
                sem_result["nodes"].extend(fresh.get("nodes", []))
                sem_result["edges"].extend(fresh.get("edges", []))
                sem_result["hyperedges"].extend(fresh.get("hyperedges", []))
                sem_result["input_tokens"] += fresh.get("input_tokens", 0)
                sem_result["output_tokens"] += fresh.get("output_tokens", 0)

        # Prune orphaned semantic cache entries. The semantic cache is
        # content-hash-keyed and unversioned, so it is never swept by the AST
        # version-cleanup: every content change or file deletion leaves a
        # permanent orphan that accumulates unbounded (#1527). Sweep it against
        # the FULL live document set (``files_by_type`` — present in both the
        # incremental and full branches), NOT the incremental ``semantic_files``
        # changed-subset, which would delete every unchanged doc's valid entry.
        # Best-effort: a prune failure must never break extraction.
        # Hash keys are anchored to the corpus (``target``) — the same anchor
        # the cache read/write above use — while the stat-index artifact
        # follows the cache location (``out_root``). Anchoring these hashes to
        # ``out_root`` instead would mismatch every key under ``--out`` and
        # sweep the entire fresh cache as orphaned (#1990/#1991).
        try:
            from graphify.cache import file_hash as _file_hash
            _live_hashes: set[str] = set()
            for _kind in ("document", "paper", "image"):
                for _fp in files_by_type.get(_kind, []):
                    _abs = Path(_fp)
                    if not _abs.is_absolute():
                        _abs = Path(target) / _abs
                    if not _abs.is_file():
                        continue  # deleted/missing — leave out so its entry is pruned
                    try:
                        _live_hashes.add(_file_hash(_abs, target, cache_root=out_root))
                    except OSError:
                        pass
            # A pathless database extraction has no filesystem corpus to sweep.
            if has_path:
                _prune_semantic_cache(out_root, _live_hashes)
        except Exception as exc:
            print(f"[graphify extract] warning: could not prune semantic cache: {exc}", file=sys.stderr)
        stages.mark("semantic extract")

        pg_result: dict = {"nodes": [], "edges": []}
        if cli_postgres_dsn is not None:
            from graphify.pg_introspect import introspect_postgres
            print(f"[graphify extract] introspecting PostgreSQL schema...")
            try:
                pg_result = introspect_postgres(cli_postgres_dsn)
            except (ConnectionError, ImportError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"[graphify extract] PostgreSQL: {len(pg_result['nodes'])} nodes, "
                  f"{len(pg_result['edges'])} edges")

        cargo_result: dict = {"nodes": [], "edges": []}
        if cli_cargo:
            from graphify.cargo_introspect import introspect_cargo
            print("[graphify extract] introspecting Cargo workspace...")
            try:
                cargo_result = introspect_cargo(target)
            except (ConnectionError, ImportError, OSError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"[graphify extract] Cargo: {len(cargo_result['nodes'])} nodes, "
                  f"{len(cargo_result['edges'])} edges")

        # Merge AST + semantic + pg_result + cargo_result. Order matters for deduplication: passing AST
        # first means semantic node attributes win on collision (richer labels
        # for symbols also referenced in docs). Hyperedges only come from the
        # semantic side.
        merged: dict = {
            "nodes": list(ast_result.get("nodes", [])) + list(sem_result.get("nodes", [])) + list(pg_result.get("nodes", [])) + list(cargo_result.get("nodes", [])),
            "edges": list(ast_result.get("edges", [])) + list(sem_result.get("edges", [])) + list(pg_result.get("edges", [])) + list(cargo_result.get("edges", [])),
            "hyperedges": list(sem_result.get("hyperedges", [])),
            "input_tokens": ast_result.get("input_tokens", 0) + sem_result.get("input_tokens", 0),
            "output_tokens": ast_result.get("output_tokens", 0) + sem_result.get("output_tokens", 0),
        }

        graph_json_path = graphify_out / "graph.json"
        analysis_path = graphify_out / ".graphify_analysis.json"

        # Build a manifest-safe files dict: only stamp semantic_hash for files
        # that actually produced output (cache hit or fresh extraction). Files
        # whose chunk failed have no source_file entry in sem_result — leaving
        # their semantic_hash empty so detect_incremental re-queues them (#933).
        # Path normalization against the scan root happens inside the helper
        # (#1897) so fresh root-relative source_files match detect()'s
        # absolute file lists.
        _manifest_files = _stamped_manifest_files(files_by_type, sem_result, target,
                                                   partial_source_files=_partial_semantic_files)

        # Files dispatched this run but dropped by _stamped_manifest_files
        # above (failed chunk, LLM omission, or any future exclusion) still
        # carry a stale semantic_hash from a prior successful run in the
        # on-disk manifest; save_manifest's seed loop would otherwise copy it
        # verbatim and mask the omission (#1948). Derived from semantic_files
        # — what was actually SENT to the backend this run (narrowed by the
        # incremental gate and --code-only, widened by deep mode) — NOT from
        # files_by_type: the full live corpus includes untouched files that
        # were never dispatched, and clearing those would blank the whole
        # manifest on every partial incremental run, forcing a full-corpus
        # re-extraction on the next one.
        _stamped_semantic = {
            f for _flist in _manifest_files.values() for f in _flist
        }
        _cleared_semantic = {str(p) for p in semantic_files} - _stamped_semantic

        # Full-scan manifest saves prune rows for in-root files that left the
        # scan corpus but still exist on disk (#1908). The corpus must be the
        # RAW detect output (files_by_type), NOT the #933-stamp-filtered
        # _manifest_files above — pruning to the filtered set would erase
        # failed-chunk/omitted-doc rows and every doc row on --code-only runs.
        _scan_corpus = (
            {f for _fl in files_by_type.values() for f in _fl}
            if has_path else None
        )

        def _invalidate_file_manifest_for_db_graph() -> None:
            if has_path:
                return
            try:
                manifest_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"error: could not invalidate file manifest: {exc}", file=sys.stderr)
                sys.exit(1)

        if no_cluster:
            # --no-cluster: dump the raw merged extraction as graph.json.
            # No NetworkX, no community detection, no analysis sidecar.
            # Dedupe nodes (by id) and parallel edges so the raw output matches the
            # clustered path (whose DiGraph collapses both) and stays deterministic
            # across modes (#1317; node dedup also collapses shared Swift module
            # anchors emitted per importing file, #1327).
            from graphify.build import dedupe_edges as _dedupe_edges, dedupe_nodes as _dedupe_nodes
            from graphify.export import (
                backup_if_protected as _backup,
                existing_graph_node_count as _existing_graph_node_count,
            )
            if (
                incremental_mode
                and not code_files
                and not semantic_files
                and not deleted_files
                and not pg_result.get("nodes")
                and not pg_result.get("edges")
                and not cargo_result.get("nodes")
                and not cargo_result.get("edges")
            ):
                # An exclusion-only change reaches this gate (excluded files
                # are deliberately NOT in deleted_files, #1908) but must still
                # scrub the newly-excluded sources from the raw graph (#1909).
                # This path never runs build_merge, so prune in place.
                if graph_stale_sources:
                    _n_pruned = _prune_graph_json_sources(
                        existing_graph_path, graph_stale_sources
                    )
                    if _n_pruned:
                        print(
                            f"[graphify extract] pruned {_n_pruned} node(s) from "
                            f"{len(graph_stale_sources)} source file(s) no longer "
                            "in the scan (deleted or excluded)."
                        )
                print(
                    "[graphify extract] no incremental changes detected "
                    "(--no-cluster); outputs left untouched."
                )
                try:
                    _save_manifest(_manifest_files, manifest_path=str(manifest_path), kind="both", root=target, scan_corpus=_scan_corpus, clear_semantic=_cleared_semantic)
                except Exception as exc:
                    print(f"[graphify extract] warning: could not write manifest: {exc}", file=sys.stderr)
                stages.total()
                sys.exit(0)

            if incremental_mode:
                # #2169: this raw path used to write ONLY this run's extraction
                # over graph.json — on an incremental run that is just the
                # changed files, silently dropping every node/edge owned by an
                # unchanged file. Merge the existing graph forward first, with
                # the same replace/prune semantics as the clustered path's
                # build_merge: re-extracted sources replaced, deleted +
                # excluded + graph-stale sources pruned, everything else
                # carried. Survivors are prepended, so the dedupe below keeps
                # this run's fresh attributes for re-extracted nodes.
                from graphify.build import merge_raw_extraction as _merge_raw_extraction
                _raw_prune_sources: list[str] = list(deleted_files)
                for _src in list(excluded_files) + graph_stale_sources:
                    if _src not in _raw_prune_sources:
                        _raw_prune_sources.append(_src)
                try:
                    merged = _merge_raw_extraction(
                        merged,
                        graph_path=existing_graph_path,
                        prune_sources=_raw_prune_sources or None,
                        root=target,
                    )
                except RuntimeError as exc:
                    # Existing graph present but unparseable: refuse to
                    # raw-dump this run's partial extraction over it.
                    print(f"error: {exc}", file=sys.stderr)
                    sys.exit(1)
            merged["nodes"] = _dedupe_nodes(merged["nodes"])
            merged["edges"] = _dedupe_edges(merged["edges"])
            # Disambiguate colliding-basename file-node labels (#2032). This raw
            # --no-cluster path bypasses build_from_json (where the clustered path
            # gets this), so apply it directly on the merged node list.
            from graphify.build import disambiguate_file_labels_in_nodes as _disamb_labels
            _disamb_labels(merged["nodes"])
            # Backfill source_file from endpoint nodes — this raw path bypasses
            # build_from_json's backfill, and semantic edges sometimes omit it (#1279).
            _node_sf = {n.get("id"): n.get("source_file") for n in merged["nodes"]}
            for _e in merged["edges"]:
                if not _e.get("source_file"):
                    _e["source_file"] = (
                        _node_sf.get(_e.get("source")) or _node_sf.get(_e.get("target")) or ""
                    )
            # RT-parity for the raw path: an incomplete build must not force a
            # partial graph over a larger complete one here either. The clustered
            # path gets this from to_json's #479 guard; this path never calls
            # to_json, so replicate the shrink check against the existing file and
            # exit before the write/manifest unless --allow-partial is set.
            if _extraction_incomplete and not cli_allow_partial:
                from graphify.export import MALFORMED_GRAPH as _MALFORMED_GRAPH
                _existing_n = _existing_graph_node_count(graph_json_path)
                _malformed = _existing_n is _MALFORMED_GRAPH
                _shrinks = isinstance(_existing_n, int) and len(merged["nodes"]) < _existing_n
                if _malformed or _shrinks:
                    _detail = (
                        f"the existing {graph_json_path} is present but unparseable "
                        "(corrupt or a mid-write), so a shrink cannot be ruled out"
                        if _malformed
                        else f"smaller than the existing {graph_json_path} "
                        f"({len(merged['nodes'])} < {_existing_n} nodes)"
                    )
                    print(
                        "[graphify extract] error: extraction was incomplete (an AST/"
                        f"semantic pass failed) and the resulting --no-cluster graph is {_detail}. "
                        "Refusing to overwrite a complete graph with a partial one. Re-run after "
                        "fixing the failures, or pass --allow-partial to overwrite anyway.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            _backup(graphify_out)
            _invalidate_file_manifest_for_db_graph()
            from graphify.paths import write_json_atomic as _write_json_atomic
            _write_json_atomic(graph_json_path, merged, indent=2)
            try:
                # Record the scan root so a later build_merge / update runbook can
                # relativize deleted-file paths correctly even for a custom --out
                # (its grandparent-of-graph.json fallback points at the wrong dir
                # otherwise, and deleted files never prune — #2012/#1571).
                (graphify_out / ".graphify_root").write_text(
                    str(Path(target).resolve()), encoding="utf-8"
                )
            except OSError:
                pass
            stages.mark("write")
            cost = _estimate_cost(
                backend, merged["input_tokens"], merged["output_tokens"]
            )
            print(
                f"[graphify extract] wrote {graph_json_path} — "
                f"{len(merged['nodes'])} nodes, {len(merged['edges'])} edges "
                f"(no clustering)"
            )
            if merged["input_tokens"] or merged["output_tokens"]:
                print(
                    f"[graphify extract] tokens: "
                    f"{merged['input_tokens']:,} in / "
                    f"{merged['output_tokens']:,} out, "
                    f"est. cost: ${cost:.4f}"
                )
            try:
                if has_path:
                    _save_manifest(_manifest_files, manifest_path=str(manifest_path), kind="both", root=target, scan_corpus=_scan_corpus, clear_semantic=_cleared_semantic)
            except Exception as exc:
                print(f"[graphify extract] warning: could not write manifest: {exc}", file=sys.stderr)
            if global_merge:
                from graphify.global_graph import global_add as _global_add
                _tag = global_repo_tag or target.name
                try:
                    result = _global_add(graphify_out / "graph.json", _tag)
                    if result["skipped"]:
                        print(f"[graphify global] '{_tag}' unchanged since last add - skipped.")
                    else:
                        print(f"[graphify global] '{_tag}' merged into global graph "
                              f"(+{result['nodes_added']} nodes, -{result['nodes_removed']} pruned).")
                except Exception as exc:
                    print(f"[graphify global] warning: failed to merge into global graph: {exc}", file=sys.stderr)
            stages.total()
            sys.exit(0)

        # Build graph + cluster + score + write.
        from graphify.build import (
            build as _build,
            build_from_json as _build_from_json,
            build_merge as _build_merge,
        )
        from graphify.cluster import cluster as _cluster, score_all as _score_all
        from graphify.export import to_json as _to_json
        from graphify.analyze import god_nodes as _god_nodes, surprising_connections as _surprising
        dedup_backend = backend if dedup_llm else None
        if incremental_mode:
            # Prune everything the current scan no longer covers: genuinely
            # deleted manifest rows, excluded-but-alive manifest rows (#1908),
            # and the graph's own stale sources — which catches files that
            # became excluded without ever being manifest-listed (#1909).
            _prune_sources: list[str] = list(deleted_files)
            for _src in list(excluded_files) + graph_stale_sources:
                if _src not in _prune_sources:
                    _prune_sources.append(_src)
            G = _build_merge(
                [merged],
                graph_path=existing_graph_path,
                prune_sources=_prune_sources or None,
                dedup=True,
                dedup_llm_backend=dedup_backend,
                root=target,
            )
        else:
            G = _build([merged], dedup=True, dedup_llm_backend=dedup_backend, root=target)
        stages.mark("build")
        if G.number_of_nodes() == 0:
            print(
                "[graphify extract] graph is empty — extraction produced no nodes. "
                "Possible causes: all files skipped, binary-only corpus, or LLM "
                "returned no edges.",
                file=sys.stderr,
            )
            sys.exit(1)

        communities = _cluster(G, resolution=cli_resolution, exclude_hubs_percentile=cli_exclude_hubs)
        stages.mark("cluster")
        cohesion = _score_all(G, communities)
        try:
            gods = _god_nodes(G)
        except Exception:
            gods = []
        try:
            surprises = _surprising(G, communities)
        except Exception:
            surprises = []
        stages.mark("analyze")

        from graphify.export import backup_if_protected as _backup
        _backup(graphify_out)
        _invalidate_file_manifest_for_db_graph()
        # force=True bypasses the #479 shrink guard entirely. A full build
        # legitimately shrinks (fuzzy dedup collapse, deleted code) so it keeps
        # force=True — EXCEPT when this run's extraction was incomplete (an
        # extractor pass crashed or some semantic chunks failed). Then a partial
        # graph could silently overwrite a good complete one, so fall back to the
        # shrink guard (force=False) unless the user opts in with --allow-partial.
        #
        # Both write paths are guarded: the clustered path here via to_json's
        # #479 check, and the `--no-cluster` raw-dump path above via the same
        # shrink check against the existing file (existing_graph_node_count).
        #
        # Trade-off: this reuses to_json's coarse node-count guard, not the
        # source-aware _check_shrink that watch/update use. On an incremental run
        # a legitimate deletion that coincides with an unrelated transient chunk
        # failure can therefore be refused here — recoverable by re-running or
        # passing --allow-partial (the good graph is preserved and the manifest
        # is not stamped, so the retry re-extracts).
        _force_write = cli_allow_partial or not _extraction_incomplete
        _wrote = _to_json(G, communities, str(graph_json_path), force=_force_write)
        if not _wrote:
            # The shrink guard refused: this partial build is smaller than the
            # existing graph. Exit before writing the manifest/marker below, which
            # would otherwise stamp these files as done and make the next
            # incremental run skip re-extracting them (poisoning the manifest
            # against the graph we declined to write). Exit non-zero so a retry
            # re-attempts.
            print(
                "[graphify extract] error: extraction was incomplete (an AST/semantic "
                f"pass failed) and the resulting graph is smaller than the existing "
                f"{graph_json_path}. Refusing to overwrite a complete graph with a "
                "partial one. Re-run after fixing the failures, or pass --allow-partial "
                "to overwrite anyway.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            # See the --no-cluster path above: persist the scan root so build_merge
            # can relativize deleted-file paths under a custom --out (#2012/#1571).
            (graphify_out / ".graphify_root").write_text(
                str(Path(target).resolve()), encoding="utf-8"
            )
        except OSError:
            pass
        stages.mark("export")
        if merged.get("output_tokens", 0) > 0:
            (graphify_out / ".graphify_semantic_marker").write_text(
                json.dumps({"output_tokens": merged["output_tokens"]}), encoding="utf-8"
            )
        if global_merge:
            from graphify.global_graph import global_add as _global_add
            _tag = global_repo_tag or target.name
            try:
                result = _global_add(graphify_out / "graph.json", _tag)
                if result["skipped"]:
                    print(f"[graphify global] '{_tag}' unchanged since last add - skipped.")
                else:
                    print(f"[graphify global] '{_tag}' merged into global graph "
                          f"(+{result['nodes_added']} nodes, -{result['nodes_removed']} pruned).")
            except Exception as exc:
                print(f"[graphify global] warning: failed to merge into global graph: {exc}", file=sys.stderr)
        analysis = {
            "communities": {str(k): v for k, v in communities.items()},
            "cohesion": {str(k): v for k, v in cohesion.items()},
            "gods": gods,
            "surprises": surprises,
            "tokens": {
                "input": merged["input_tokens"],
                "output": merged["output_tokens"],
            },
        }
        from graphify.paths import write_json_atomic as _wja
        _wja(analysis_path, analysis, indent=2)
        try:
            if has_path:
                _save_manifest(_manifest_files, manifest_path=str(manifest_path), kind="both", root=target, scan_corpus=_scan_corpus, clear_semantic=_cleared_semantic)
        except Exception as exc:
            print(f"[graphify extract] warning: could not write manifest: {exc}", file=sys.stderr)

        cost = _estimate_cost(backend, merged["input_tokens"], merged["output_tokens"])
        print(
            f"[graphify extract] wrote {graph_json_path}: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{len(communities)} communities"
        )
        print(f"[graphify extract] wrote {analysis_path}")
        if incremental_mode:
            _excl_note = f", {len(excluded_files)} excluded" if excluded_files else ""
            print(
                f"[graphify extract] incremental summary: "
                f"{sem_cache_hits + unchanged_total} files cached/unchanged, "
                f"{len(code_files) + sem_cache_misses} re-extracted, "
                f"{len(deleted_files)} deleted{_excl_note}"
            )
        elif sem_cache_hits:
            print(f"[graphify extract] semantic cache: {sem_cache_hits} cached, {sem_cache_misses} re-extracted")
        if merged["input_tokens"] or merged["output_tokens"]:
            print(
                f"[graphify extract] tokens: "
                f"{merged['input_tokens']:,} in / "
                f"{merged['output_tokens']:,} out, "
                f"est. cost (~{backend}): ${cost:.4f}"
            )
        # extract intentionally stops at graph.json + analysis; the report and
        # community labels are produced by `cluster-only` (or an agent's Step 5).
        # Point standalone users at it so communities get named (#1097).
        print(
            "[graphify extract] next: run "
            f"`graphify cluster-only {graphify_out.parent}` "
            "to generate GRAPH_REPORT.md and name communities"
        )
        stages.total()

    elif Path(cmd).exists() or cmd in (".", "..") or cmd.startswith(("./", "../", "/", "~")):
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
