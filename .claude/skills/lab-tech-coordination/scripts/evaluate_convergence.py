#!/usr/bin/env python3
"""
Evaluate convergence for lab-tech coordination testing.

Checks quality gates, calculates convergence score, and determines
if iteration 2 is needed.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple


def check_gate_in_artifact(gate_id: str, artifact_path: Path) -> bool:
    """
    Check if artifact addresses a specific gate.

    Args:
        gate_id: Gate identifier
        artifact_path: Path to artifact file

    Returns:
        True if gate is addressed
    """
    with open(artifact_path, 'r') as f:
        content = f.read()

    if gate_id == 'comprehensive_coverage':
        # Must have multiple sections (≥3 ## headings)
        section_count = content.count('##')
        return section_count >= 3

    elif gate_id == 'specific_examples':
        # Must have code blocks or concrete scenarios
        return '```' in content or 'Example:' in content

    elif gate_id == 'actionable_recommendations':
        # Must have imperative language
        keywords = ['should', 'must', 'implement', 'fix', 'add', 'remove']
        return any(kw in content.lower() for kw in keywords)

    return False


def find_latest_artifact(session_dir: Path, agent_name: str) -> Path:
    """
    Find the latest artifact from an agent.

    Args:
        session_dir: Session directory
        agent_name: Agent name (red or blue)

    Returns:
        Path to latest artifact, or None if not found
    """
    artifacts_dir = session_dir / "artifacts"
    if not artifacts_dir.exists():
        return None

    # Look for artifacts matching agent name
    artifacts = list(artifacts_dir.glob(f"*{agent_name}*.md"))
    if not artifacts:
        return None

    # Return most recently modified
    return max(artifacts, key=lambda p: p.stat().st_mtime)


def evaluate_convergence(session_dir: Path, verbose: bool = True) -> Tuple[bool, float, List[str]]:
    """
    Evaluate convergence based on quality gates.

    Args:
        session_dir: Session directory
        verbose: Print detailed evaluation

    Returns:
        (converged, quality_score, failed_gates)
    """
    session_dir = Path(session_dir)

    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating Convergence")
        print(f"{'='*60}")

    # Load convergence configuration
    convergence_file = session_dir / "convergence.json"
    with open(convergence_file, 'r') as f:
        convergence = json.load(f)

    # Find artifacts
    red_artifact = find_latest_artifact(session_dir, "red")
    blue_artifact = find_latest_artifact(session_dir, "blue")

    if not red_artifact:
        print("❌ No Red artifact found")
        return False, 0.0, []

    if not blue_artifact:
        print("❌ No Blue artifact found")
        return False, 0.0, []

    if verbose:
        print(f"\n📄 Red artifact: {red_artifact.name}")
        print(f"📄 Blue artifact: {blue_artifact.name}")

    # Evaluate each gate
    failed_gates = []
    gate_results = []

    for gate in convergence['convergence_gates']:
        gate_id = gate['id']

        # Check if BOTH agents addressed this gate
        red_addressed = check_gate_in_artifact(gate_id, red_artifact)
        blue_addressed = check_gate_in_artifact(gate_id, blue_artifact)

        gate_met = red_addressed and blue_addressed
        gate['met'] = gate_met

        if not gate_met:
            failed_gates.append(gate_id)

        gate_results.append({
            'gate_id': gate_id,
            'red': red_addressed,
            'blue': blue_addressed,
            'met': gate_met
        })

        if verbose:
            status = "✓" if gate_met else "✗"
            print(f"\n{status} {gate['id']}")
            print(f"   Description: {gate['description']}")
            print(f"   Red: {'✓' if red_addressed else '✗'}  Blue: {'✓' if blue_addressed else '✗'}")

    # Calculate quality score
    gates_met = sum(1 for g in convergence['convergence_gates'] if g['met'])
    total_gates = len(convergence['convergence_gates'])
    quality = gates_met / total_gates if total_gates > 0 else 0.0

    convergence['current_quality'] = quality

    # Determine convergence
    iteration = convergence['current_iteration']
    max_iterations = convergence['max_iterations']
    quality_threshold = convergence['quality_threshold']

    converged = (quality >= quality_threshold) or (iteration >= max_iterations)
    convergence['converged'] = converged

    if verbose:
        print(f"\n{'='*60}")
        print(f"Quality Score: {quality:.2f} (threshold: {quality_threshold})")
        print(f"Iteration: {iteration}/{max_iterations}")
        print(f"Gates Met: {gates_met}/{total_gates}")

        if converged:
            if quality >= quality_threshold:
                print(f"\n✅ CONVERGED - Quality threshold met!")
            else:
                print(f"\n✅ CONVERGED - Max iterations reached")
        else:
            print(f"\n⚠️  NOT CONVERGED - Iteration {iteration + 1} needed")
            print(f"\nFailed gates:")
            for gate_id in failed_gates:
                print(f"  - {gate_id}")

    # Update convergence file
    with open(convergence_file, 'w') as f:
        json.dump(convergence, f, indent=2)

    return converged, quality, failed_gates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_convergence.py <session_dir>")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    converged, quality, failed_gates = evaluate_convergence(session_dir, verbose=True)

    sys.exit(0 if converged else 1)
