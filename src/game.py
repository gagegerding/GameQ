#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GameQ CLI: practice basic quantum gates on a single qubit.

Two modes:
  1) Single-qubit Gate Effect (SQ_GATE_EFFECT): predict the effect of one gate on an initial state
  2) Single-qubit Target Build (SQ_BUILD_TARGET): enter a sequence of gates that prepares a target state

This script expects a JSON file produced by build_content.py (questions.json)
at the repository root. It also displays pre-rendered circuit SVGs' paths when available.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------
# Utility: pretty printing
# -----------------------------

def hr(char: str = "─", n: int = 60) -> str:
    return char * n

def fmt_vec(v: np.ndarray, ndigits: int = 4) -> str:
    a, b = v
    def cstr(z: complex) -> str:
        r = round(z.real, ndigits)
        i = round(z.imag, ndigits)
        if abs(i) < 10**(-ndigits): i = 0.0
        if abs(r) < 10**(-ndigits): r = 0.0
        if i == 0.0:
            return f"{r}"
        sign = "+" if i >= 0 else "-"
        return f"{r}{sign}{abs(i)}j"
    return f"[{cstr(a)}, {cstr(b)}]"

def fmt_bloch(r: np.ndarray, ndigits: int = 4) -> str:
    x, y, z = [round(float(t), ndigits) for t in r]
    return f"({x}, {y}, {z})"


# -----------------------------
# Canonical states and helpers
# -----------------------------

KET = {
    "ket0": np.array([1+0j, 0+0j]),
    "ket1": np.array([0+0j, 1+0j]),
    "plus": (1/np.sqrt(2))*np.array([1, 1], dtype=complex),
    "minus": (1/np.sqrt(2))*np.array([1, -1], dtype=complex),
    "plus_i": (1/np.sqrt(2))*np.array([1, 1j], dtype=complex),
    "minus_i": (1/np.sqrt(2))*np.array([1, -1j], dtype=complex),
}

NAME_BY_VEC = {
    "ket0": KET["ket0"],
    "ket1": KET["ket1"],
    "|0>": KET["ket0"],
    "|1>": KET["ket1"],
    "0": KET["ket0"],
    "1": KET["ket1"],
    "plus": KET["plus"],
    "minus": KET["minus"],
    "plus_i": KET["plus_i"],
    "minus_i": KET["minus_i"],
    "|+>": KET["plus"],
    "|->": KET["minus"],
}

def norm_phase_strip(v: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Normalize a state and strip global phase so max-magnitude component is real & >= 0."""
    v = v.astype(complex)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    v = v / n
    idx = int(np.argmax(np.abs(v)))
    if abs(v[idx]) > tol:
        v = v * np.exp(-1j * np.angle(v[idx]))
    # zero-out tiny numerical noise
    v.real[np.abs(v.real) < tol] = 0
    v.imag[np.abs(v.imag) < tol] = 0
    return v

def bloch_from_state(v: np.ndarray) -> np.ndarray:
    a, b = v
    x = 2*np.real(a*np.conj(b))
    y = 2*np.imag(a*np.conj(b))
    z = (abs(a)**2 - abs(b)**2).real
    return np.array([float(x), float(y), float(z)], dtype=float)

def axis_angle_from_gate(token: str) -> Tuple[np.ndarray, float]:
    """Map tokens like H, X, S, Td, Rx(pi/3) to (axis, angle)."""
    token = token.strip()
    if token.startswith("Rx(") and token.endswith(")"):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return np.array([1.0, 0.0, 0.0]), float(th)
    if token.startswith("Ry(") and token.endswith(")"):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return np.array([0.0, 1.0, 0.0]), float(th)
    if token.startswith("Rz(") and token.endswith(")"):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return np.array([0.0, 0.0, 1.0]), float(th)

    mapping = {
        "X": (np.array([1.0, 0.0, 0.0]), math.pi),
        "Y": (np.array([0.0, 1.0, 0.0]), math.pi),
        "Z": (np.array([0.0, 0.0, 1.0]), math.pi),
        "S": (np.array([0.0, 0.0, 1.0]), math.pi/2),
        "Sd":(np.array([0.0, 0.0, 1.0]), -math.pi/2),
        "T": (np.array([0.0, 0.0, 1.0]), math.pi/4),
        "Td":(np.array([0.0, 0.0, 1.0]), -math.pi/4),
        # H: π around (X+Z)/√2
        "H": (np.array([1/math.sqrt(2), 0.0, 1/math.sqrt(2)]), math.pi),
    }
    if token in mapping:
        return mapping[token]
    raise ValueError(f"Unsupported gate token: {token}")

def apply_axis_angle_to_bloch(r: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate Bloch vector r by angle theta around unit axis using Rodrigues' formula."""
    r = np.array(r, dtype=float)
    n = np.array(axis, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return r
    n = n / n_norm
    return (r*np.cos(theta) + np.cross(n, r)*np.sin(theta) + n*np.dot(n, r)*(1 - np.cos(theta)))

def state_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """Compare states up to global phase by normalizing and stripping phase."""
    aa = norm_phase_strip(a.copy(), tol=1e-12)
    bb = norm_phase_strip(b.copy(), tol=1e-12)
    return np.linalg.norm(aa - bb) < tol

def bloch_close(a: np.ndarray, b: np.ndarray, tol: float = 5e-3) -> bool:
    return np.linalg.norm(np.array(a) - np.array(b)) < tol


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Question:
    id: str
    template: str
    prompt: Optional[str]
    params: Dict[str, Any]
    assets: Dict[str, Any]
    difficulty: int = 1
    bloch_start: Optional[List[float]] = None
    bloch_end: Optional[List[float]] = None
    bloch_target: Optional[List[float]] = None
    state_start: Optional[List[str]] = None
    target_vector: Optional[List[str]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    exemplar_steps: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Question":
        return Question(
            id=obj["id"],
            template=obj.get("type") or obj["template"],
            prompt=obj.get("prompt"),
            params=obj.get("params", {}),
            assets=obj.get("assets", {}),
            difficulty=int(obj.get("difficulty", 1)),
            bloch_start=obj.get("bloch_start"),
            bloch_end=obj.get("bloch_end"),
            bloch_target=obj.get("bloch_target"),
            state_start=obj.get("state_start"),
            target_vector=obj.get("target_vector"),
            steps=obj.get("steps"),
            exemplar_steps=obj.get("exemplar_steps"),
        )


# -----------------------------
# Loading content
# -----------------------------

def load_questions(json_path: str) -> List[Question]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Question.from_json(x) for x in data]

def default_questions_path() -> str:
    # repository root (questions.json alongside build_content.py)
    here = os.path.abspath(os.getcwd())
    candidate = os.path.join(here, "questions.json")
    if os.path.exists(candidate):
        return candidate
    # try parent (if running from src/)
    parent = os.path.dirname(here)
    candidate2 = os.path.join(parent, "questions.json")
    if os.path.exists(candidate2):
        return candidate2
    raise FileNotFoundError("questions.json not found. Run build_content.py to generate it.")


# -----------------------------
# Parsing answers
# -----------------------------

def parse_state_input(s: str) -> Optional[np.ndarray]:
    """Accept 'ket0', '|1>', '|+>', 'plus', numeric '[a,b]' forms, etc."""
    t = s.strip().lower()
    if t in NAME_BY_VEC:
        return NAME_BY_VEC[t]
    # try list-like format: [a, b] where a,b are complex like '0.7071+0.0j'
    if t.startswith("[") and t.endswith("]"):
        body = t[1:-1].strip()
        parts = [p.strip() for p in body.split(",")]
        if len(parts) == 2:
            try:
                a = complex(parts[0])
                b = complex(parts[1])
                return np.array([a, b], dtype=complex)
            except Exception:
                return None
    return None

def parse_gate_sequence(s: str) -> List[str]:
    """Parse a user string like 'H;S;Td' or 'Ry(pi/3),H' into tokens."""
    raw = s.replace(",", ";")
    toks = [tok.strip() for tok in raw.split(";") if tok.strip()]
    return toks


# -----------------------------
# Game logic
# -----------------------------

def play_sq_gate_effect(q: Question) -> None:
    print(hr())
    print(f"[{q.id}] Single-qubit Gate Effect")
    gate = q.params["gate"] if "gate" in q.params else (q.steps[0]["gate"] if q.steps else "?")
    init_id = q.params.get("initial_state", "ket0")
    init = KET.get(init_id, KET["ket0"])
    print(f"Gate: {gate}")
    print(f"Initial state: {init_id} = {fmt_vec(init)}")
    if q.assets and "circuit" in q.assets:
        print(f"Circuit SVG: {q.assets['circuit']}")

    # expected bloch end (from JSON) if present
    expected_bloch = np.array(q.bloch_end) if q.bloch_end is not None else None

    ans = input("Predict final state (e.g., '|+>', 'ket1', or [a+0j, b+0j]): ").strip()
    user_state = parse_state_input(ans)
    if user_state is None:
        print("✗ Could not parse your answer. Try forms like |+>, ket1, plus, or [0.7071+0j, 0.7071+0j].")
        return
    # Compare by Bloch if provided; else compare canonical states up to global phase
    ok = False
    if expected_bloch is not None:
        ok = bloch_close(bloch_from_state(norm_phase_strip(user_state)), expected_bloch)
    else:
        # simulate quickly to get reference
        axis, th = axis_angle_from_gate(gate)
        r0 = bloch_from_state(norm_phase_strip(init))
        r1 = apply_axis_angle_to_bloch(r0, axis, th)
        ok = bloch_close(bloch_from_state(norm_phase_strip(user_state)), r1)

    print("✓ Correct!" if ok else "✗ Not quite.")
    if not ok:
        if expected_bloch is not None:
            print(f"Expected Bloch: {fmt_bloch(expected_bloch)}")
        else:
            print("Hint: Try visualizing the rotation axis and angle on the Bloch sphere.")
    print(hr())

def play_sq_build_target(q: Question) -> None:
    print(hr())
    print(f"[{q.id}] Single-qubit Target Build")
    target_desc = q.params.get("target_state", "?")
    print(q.prompt or "Build the target state.")
    print(f"Target: {target_desc}")
    if q.assets and "exemplar_circuit" in q.assets:
        print(f"Exemplar circuit SVG: {q.assets['exemplar_circuit']}")

    # derive target vector either from JSON 'target_vector' or by reconstructing
    if q.target_vector is not None:
        # entries were stored as repr strings like '1+0j', convert back
        ta = complex(q.target_vector[0])
        tb = complex(q.target_vector[1])
        target = norm_phase_strip(np.array([ta, tb], dtype=complex))
    else:
        # fallback for named target_state or ry(theta)
        target_token = q.params.get("target_state", "plus")
        if target_token in KET:
            target = norm_phase_strip(KET[target_token])
        elif target_token.lower().startswith("ry("):
            th = eval(target_token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
            target = norm_phase_strip(np.array([math.cos(th/2), math.sin(th/2)], dtype=complex))
        else:
            print("Unsupported target_state in this question.")
            return

    print(f"Target vector: {fmt_vec(target)}  |  Bloch: {fmt_bloch(bloch_from_state(target))}")
    user = input("Enter your gate sequence (e.g., 'H' or 'S;H' or 'Ry(pi/3);H'): ").strip()
    toks = parse_gate_sequence(user)
    if not toks:
        print("No gates entered.")
        return

    # Start from |0>
    state = norm_phase_strip(KET["ket0"])
    r = bloch_from_state(state)
    # Apply gates on Bloch vector (unitary action via Rodrigues on Bloch)
    try:
        for tok in toks:
            axis, th = axis_angle_from_gate(tok)
            r = apply_axis_angle_to_bloch(r, axis, th)
        # Convert Bloch back to a state (up to global phase choose real component)
        # For a Bloch vector (x,y,z), one valid state is:
        # |ψ> = [cos(θ/2), e^{iφ} sin(θ/2)] with θ = arccos(z), φ = atan2(y, x)
        x, y, z = r
        theta = math.acos(max(-1.0, min(1.0, z)))
        phi = math.atan2(y, x)
        candidate = np.array([math.cos(theta/2), math.e**(1j*phi)*math.sin(theta/2)], dtype=complex)
        candidate = norm_phase_strip(candidate)
    except Exception as e:
        print(f"Error parsing/applying gates: {e}")
        return

    ok = state_close(candidate, target, tol=1e-5)
    print("✓ Nice! You hit the target." if ok else "✗ Not yet.")
    print(f"Your state:   {fmt_vec(candidate)}  |  Bloch: {fmt_bloch(bloch_from_state(candidate))}")
    print(f"Target state: {fmt_vec(target)}     |  Bloch: {fmt_bloch(bloch_from_state(target))}")
    if not ok and q.exemplar_steps:
        seq = "; ".join(step["gate"] for step in q.exemplar_steps)
        print(f"Hint: exemplar sequence → {seq}")
    print(hr())


# -----------------------------
# Main entry
# -----------------------------

def menu(questions: List[Question]) -> None:
    by_type: Dict[str, List[Question]] = {"gate_effect": [], "build_target": []}
    for q in questions:
        t = q.template.lower()
        if t in ("sq_gate_effect", "gate_effect"):
            by_type["gate_effect"].append(q)
        elif t in ("sq_build_target", "build_target"):
            by_type["build_target"].append(q)

    while True:
        print(hr("="))
        print(" GameQ — Single‑Qubit Practice ")
        print(hr("="))
        print("1) Gate Effect (predict final state)")
        print("2) Build Target (enter gate sequence)")
        print("Q) Quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            print("Bye!")
            return
        if choice not in ("1", "2"):
            continue
        pool = by_type["gate_effect"] if choice == "1" else by_type["build_target"]
        if not pool:
            print("No questions available for this mode. Run build_content.py and try again.")
            continue
        print(hr())
        print(f"Select question (1..{len(pool)}):")
        for i, q in enumerate(pool, 1):
            title = q.prompt or q.id
            print(f"{i:2d}. {title}  [{q.id}]")
        sel = input("> ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(pool)):
            continue
        q = pool[int(sel) - 1]
        if choice == "1":
            play_sq_gate_effect(q)
        else:
            play_sq_build_target(q)

def main(argv: List[str]) -> int:
    try:
        path = argv[1] if len(argv) > 1 else default_questions_path()
        questions = load_questions(path)
    except Exception as e:
        print(f"Error loading questions: {e}")
        return 1
    menu(questions)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
