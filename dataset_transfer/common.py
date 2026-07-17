"""Shared hierarchy and JSONL helpers for cross-domain dataset conversion."""

import json
from collections import defaultdict, deque
from pathlib import Path


def load_taxonomy(path):
    children = defaultdict(set)
    nodes = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if not fields[0]:
                raise ValueError(f"Empty taxonomy parent at {path}:{line_number}")
            parent, descendants = fields[0], fields[1:]
            if any(not child for child in descendants):
                raise ValueError(f"Empty taxonomy child at {path}:{line_number}")
            nodes.add(parent)
            nodes.update(descendants)
            children[parent].update(descendants)

    if not nodes:
        raise ValueError(f"Taxonomy is empty: {path}")
    _reject_cycles(children, nodes)
    return {parent: set(values) for parent, values in children.items()}


def _reject_cycles(children, nodes):
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            raise ValueError(f"Taxonomy contains a cycle involving {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for child in children.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in nodes:
        visit(node)


def parent_map(children):
    parents = defaultdict(set)
    for parent, descendants in children.items():
        for child in descendants:
            parents[child].add(parent)
    return {child: set(values) for child, values in parents.items()}


def leaf_only(labels, children):
    """Remove every selected label that is an ancestor of another selection."""
    selected = set(labels)

    def has_selected_descendant(label):
        queue = list(children.get(label, ()))
        seen = set()
        while queue:
            child = queue.pop()
            if child in seen:
                continue
            seen.add(child)
            if child in selected:
                return True
            queue.extend(children.get(child, ()))
        return False

    return [label for label in labels if not has_selected_descendant(label)]


def expand_ancestors(labels, children, include_root=False):
    parents = parent_map(children)
    expanded, seen = [], set()
    queue = deque(labels)
    while queue:
        label = queue.popleft()
        if label in seen:
            continue
        seen.add(label)
        if include_root or label != "Root":
            expanded.append(label)
        queue.extend(sorted(parents.get(label, ()), key=str.casefold))
    return expanded


def contract_taxonomy(children, retained, root="Root"):
    """Connect retained nodes through the nearest retained descendants."""
    retained = set(retained) | {root}
    reachable = set()
    queue = [root]
    while queue:
        node = queue.pop()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(children.get(node, ()))
    missing = retained - reachable
    if missing:
        raise ValueError(f"Labels are not reachable from {root!r}: {sorted(missing)}")

    contracted = {}
    for parent in retained:
        nearest = set()
        queue = list(children.get(parent, ()))
        seen = set()
        while queue:
            child = queue.pop()
            if child in seen:
                continue
            seen.add(child)
            if child in retained:
                nearest.add(child)
            else:
                queue.extend(children.get(child, ()))
        if nearest:
            contracted[parent] = nearest
    return contracted


def write_taxonomy(children, path, root="Root"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    visited = set()
    queue = deque([root])
    with path.open("w", encoding="utf-8") as stream:
        while queue:
            parent = queue.popleft()
            if parent in visited:
                continue
            visited.add(parent)
            descendants = sorted(children.get(parent, ()), key=str.casefold)
            if descendants:
                stream.write("\t".join([parent, *descendants]) + "\n")
                queue.extend(descendants)


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record.get("token"), str):
                raise ValueError(f"Missing string token at {path}:{line_number}")
            labels = record.get("label")
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"Missing non-empty label list at {path}:{line_number}")
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError(f"Labels must be non-empty strings at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset split is empty: {path}")
    return records


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")
