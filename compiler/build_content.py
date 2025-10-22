import json, math, pathlib
from typing import Any, Dict, List
import numpy as np
import yaml
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content" / "questions.yaml"
OUT_JSON = ROOT / "questions.json"
CIRCUITS = ROOT / "assets" / "circuits"
CIRCUITS.mkdir(parents=True, exist_ok=True)

# --- Basic states (α, β) as complex numpy arrays
KET = {
    "ket0": np.array([1+0j, 0+0j]),
    "ket1": np.array([0+0j, 1+0j]),
    "plus": (1/np.sqrt(2))*np.array([1, 1], dtype=complex),
    "minus": (1/np.sqrt(2))*np.array([1, -1], dtype=complex),
    "plus_i": (1/np.sqrt(2))*np.array([1, 1j], dtype=complex),
    "minus_i": (1/np.sqrt(2))*np.array([1, -1j], dtype=complex),
}

def norm_phase_strip(v: np.ndarray, tol=1e-12) -> np.ndarray:
    v = v/np.linalg.norm(v)
    idx = np.argmax(np.abs(v) > tol)
    if np.abs(v[idx]) > tol:
        v = v * np.exp(-1j*np.angle(v[idx]))
    v.real[np.abs(v.real)<tol]=0; v.imag[np.abs(v.imag)<tol]=0
    return v

# --- Axis/angle map for single-qubit gates
# Rx/Ry/Rz(θ) → axis=(1,0,0)/(0,1,0)/(0,0,1), angle=θ
# X=Rx(pi), Y=Ry(pi), Z=Rz(pi), S=Rz(pi/2), T=Rz(pi/4)
# H = rotation by pi around (1,0,1)/√2
def axis_angle_from_gate(token: str):
    token = token.strip()
    if token.startswith("Rx("):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return [1.0,0.0,0.0], float(th)
    if token.startswith("Ry("):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return [0.0,1.0,0.0], float(th)
    if token.startswith("Rz("):
        th = eval(token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        return [0.0,0.0,1.0], float(th)
    mapping = {
        "X": ([1.0,0.0,0.0], math.pi),
        "Y": ([0.0,1.0,0.0], math.pi),
        "Z": ([0.0,0.0,1.0], math.pi),
        "S": ([0.0,0.0,1.0], math.pi/2),
        "Sd":([0.0,0.0,1.0], -math.pi/2),
        "T": ([0.0,0.0,1.0], math.pi/4),
        "Td":([0.0,0.0,1.0], -math.pi/4),
        # Hadamard axis = (X+Z)/√2, angle=π
        "H": ([1/np.sqrt(2), 0.0, 1/np.sqrt(2)], math.pi),
    }
    if token in mapping: return mapping[token]
    raise ValueError(f"Unsupported gate token: {token}")

# --- Convert state (α,β) to Bloch vector r=(x,y,z)
def bloch_from_state(v: np.ndarray):
    a,b = v
    x = 2*np.real(a*np.conj(b))
    y = 2*np.imag(a*np.conj(b))
    z = (abs(a)**2 - abs(b)**2).real
    return [float(x), float(y), float(z)]

# --- Minimal circuit renderers (SVG)
def render_circuit_single(gate_token: str, out_path: pathlib.Path):
    qc = QuantumCircuit(1)
    name_map = {"Sd":"sdg", "Td":"tdg"}
    if gate_token in ["H","X","Y","Z","S","Sd","T","Td"]:
        getattr(qc, name_map.get(gate_token, gate_token.lower()))(0)
    elif gate_token.startswith("R"):
        base, th = gate_token[:2].lower(), eval(gate_token[3:-1], {"__builtins__": {}}, {"pi": math.pi})
        getattr(qc, base)(float(th), 0)
    circuit_drawer(qc, output="mpl", scale=1.2, idle_wires=False).savefig(out_path, bbox_inches="tight")

def render_exemplar(seq: List[str], out_path: pathlib.Path):
    qc = QuantumCircuit(1)
    name_map = {"Sd":"sdg", "Td":"tdg"}
    for tok in seq:
        if tok in ["H","X","Y","Z","S","Sd","T","Td"]:
            getattr(qc, name_map.get(tok, tok.lower()))(0)
        elif tok.startswith("R"):
            base, th = tok[:2].lower(), eval(tok[3:-1], {"__builtins__": {}}, {"pi": math.pi})
            getattr(qc, base)(float(th), 0)
    circuit_drawer(qc, output="mpl", scale=1.2, idle_wires=False).savefig(out_path, bbox_inches="tight")

# --- Build JSON
def main():
    if not CONTENT.exists():
        raise FileNotFoundError(f"Missing file: {CONTENT}")
    items_in_raw = yaml.safe_load(open(CONTENT, "r", encoding="utf-8"))
    if items_in_raw is None:
        raise ValueError(f"{CONTENT} is empty or could not be parsed.")
    if not isinstance(items_in_raw, list):
        raise ValueError(f"{CONTENT} must be a YAML list of questions (found {type(items_in_raw)}).")
    items_in = items_in_raw
    out=[]

    for q in items_in:
        qid = q["id"]
        tpl = q["template"]

        if tpl == "SQ_GATE_EFFECT":
            gate = q["params"]["gate"]
            init_id = q["params"]["initial_state"]
            init = norm_phase_strip(KET[init_id])
            axis, theta = axis_angle_from_gate(gate)

            # Final state is not required by the app to animate, but helpful:
            # apply rotation on Bloch vector (Rodrigues) to compute end for metadata
            def rodrigues(r, n, th):
                r = np.array(r, dtype=float); n = np.array(n, dtype=float)
                n = n/np.linalg.norm(n)
                return (r*np.cos(th)
                        + np.cross(n, r)*np.sin(th)
                        + n*np.dot(n, r)*(1-np.cos(th)))
            r0 = bloch_from_state(init)
            r1 = rodrigues(r0, axis, theta)

            # Render circuit
            render_circuit_single(gate, ROOT / q["visuals"]["circuit"])

            out.append({
                "id": qid,
                "type": "gate_effect",
                "prompt": f"Apply {gate} to |ψ⟩ = {init_id}.",
                "state_start": [complex(init[0]).__repr__(), complex(init[1]).__repr__()],
                "bloch_start": r0,
                "steps": [ { "gate": gate, "axis": axis, "theta": theta } ],
                "bloch_end": list(map(float, r1)),
                "assets": { "circuit": q["visuals"]["circuit"] },
                "difficulty": 1
            })

        elif tpl == "SQ_BUILD_TARGET":
            target = q["params"]["target_state"]
            if target in KET:
                target_vec = norm_phase_strip(KET[target])
            elif target.startswith("ry("):
                th = eval(target[3:-1], {"__builtins__": {}}, {"pi": math.pi})
                target_vec = norm_phase_strip(np.array([math.cos(th/2), math.sin(th/2)], dtype=complex))
            else:
                raise ValueError(f"Unsupported target_state: {target}")

            exemplar = q["params"]["exemplar"]  # list of tokens
            steps = [ dict(gate=t, axis=axis_angle_from_gate(t)[0], theta=axis_angle_from_gate(t)[1]) for t in exemplar ]
            render_exemplar(exemplar, ROOT / q["visuals"]["exemplar_circuit"])

            out.append({
                "id": qid,
                "type": "build_target",
                "prompt": q["prompt"],
                "state_start": [ "1+0j", "0+0j" ],  # |0>
                "target_vector": [complex(target_vec[0]).__repr__(), complex(target_vec[1]).__repr__()],
                "bloch_target": bloch_from_state(target_vec),
                "exemplar_steps": steps,
                "assets": { "exemplar_circuit": q["visuals"]["exemplar_circuit"] },
                "difficulty": 2
            })

        else:
            raise ValueError(f"Unknown template: {tpl}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Built {len(out)} items → {OUT_JSON}")

if __name__ == "__main__":
    main()
