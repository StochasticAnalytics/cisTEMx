#!/usr/bin/env python3
"""
Create tickets for lab-tech coordination testing.

Generates properly formatted tickets following ticket_schema.json
for iterative convergence workflow validation.
"""

import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime


def create_ticket(
    session_dir: Path,
    agent_name: str,
    iteration: int = 1,
    review_target: str = "methodological-skill-creation",
    prior_artifacts: list = None,
    failed_gates: list = None
) -> Path:
    """
    Create a ticket for an agent.

    Args:
        session_dir: Session directory path
        agent_name: "red" or "blue"
        iteration: Iteration number (1, 2, 3...)
        review_target: What to review
        prior_artifacts: List of artifact IDs from previous iteration
        failed_gates: Gates that failed in previous iteration

    Returns:
        Path to created ticket file
    """
    session_dir = Path(session_dir)

    # Generate ticket ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticket_id = f"ticket_{agent_name}_{iteration}_{timestamp}_{uuid.uuid4().hex[:8]}"

    # Base ticket structure
    ticket = {
        "id": ticket_id,
        "type": "initial_analysis" if iteration == 1 else "iteration",
        "iteration": iteration,
        "assigned_to": agent_name,
        "created_at": time.time(),
        "ttl_seconds": 300,  # 5 minutes
        "max_requeues": 1,
        "requeue_count": 0,
        "state": "pending",
        "review_target": review_target,
        "convergence_gates": [
            "comprehensive_coverage",
            "specific_examples",
            "actionable_recommendations"
        ]
    }

    # Iteration-specific instructions
    if iteration == 1:
        # Initial analysis instructions
        if agent_name == "red":
            ticket["instructions"] = f"""Perform critical analysis of the {review_target} skill.

Your role: Identify gaps, weaknesses, edge cases, and potential failures.

Focus areas:
1. Missing or incomplete coverage
2. Unclear or ambiguous guidance
3. Potential misuse patterns
4. Structural weaknesses
5. Gaps in progressive disclosure

Deliverables:
- Write comprehensive critical analysis artifact
- Provide specific examples of issues
- Categorize by severity (CRITICAL/MAJOR/MINOR)
- Include evidence from files analyzed

Convergence gates to address:
- comprehensive_coverage: Analyze multiple aspects (≥3 major areas)
- specific_examples: Provide concrete code/scenario examples
- actionable_recommendations: Suggest specific fixes
"""
        else:  # blue
            ticket["instructions"] = f"""Perform constructive analysis of the {review_target} skill.

Your role: Identify strengths, opportunities, and paths for improvement.

Focus areas:
1. Effective patterns and strong elements
2. Opportunities for enhancement
3. Potential for broader application
4. Good examples of progressive disclosure
5. Clear and helpful guidance

Deliverables:
- Write comprehensive constructive analysis artifact
- Celebrate specific strengths with examples
- Propose practical enhancements
- Identify patterns worth spreading

Convergence gates to address:
- comprehensive_coverage: Analyze multiple aspects (≥3 major areas)
- specific_examples: Provide concrete code/scenario examples
- actionable_recommendations: Suggest specific improvements
"""
    else:
        # Iteration 2+ refinement instructions
        ticket["prior_context"] = prior_artifacts or []

        refinement_focus = []
        if failed_gates:
            for gate in failed_gates:
                if gate == "comprehensive_coverage":
                    refinement_focus.append("- Expand analysis to cover more aspects (need ≥3 major sections)")
                elif gate == "specific_examples":
                    refinement_focus.append("- Add concrete code examples or specific scenarios")
                elif gate == "actionable_recommendations":
                    refinement_focus.append("- Strengthen actionable guidance (use: should, must, implement)")

        refinement_text = "\n".join(refinement_focus) if refinement_focus else "- Deepen and expand your previous analysis"

        if agent_name == "red":
            ticket["instructions"] = f"""Refine your critical analysis based on convergence evaluation.

Previous analysis: {', '.join(prior_artifacts or [])}

Refinement needed:
{refinement_text}

Maintain your critical perspective while addressing the gaps above.
Provide additional evidence and specific examples.
"""
        else:  # blue
            ticket["instructions"] = f"""Refine your constructive analysis based on convergence evaluation.

Previous analysis: {', '.join(prior_artifacts or [])}

Refinement needed:
{refinement_text}

Maintain your constructive perspective while addressing the gaps above.
Provide additional examples and concrete recommendations.
"""

    # Write ticket to pending directory
    pending_dir = session_dir / "tickets" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    ticket_path = pending_dir / f"{ticket_id}.json"
    with open(ticket_path, 'w') as f:
        json.dump(ticket, f, indent=2)

    print(f"✓ Created ticket: {ticket_id}")
    print(f"  Agent: {agent_name}")
    print(f"  Iteration: {iteration}")
    print(f"  Location: {ticket_path}")

    return ticket_path


def create_iteration_tickets(
    session_dir: Path,
    iteration: int = 1,
    prior_artifacts: dict = None,
    failed_gates: list = None
):
    """
    Create tickets for both Red and Blue for a given iteration.

    Args:
        session_dir: Session directory
        iteration: Iteration number
        prior_artifacts: Dict with 'red' and 'blue' artifact IDs
        failed_gates: List of gate IDs that failed
    """
    print(f"\n{'='*60}")
    print(f"Creating Iteration {iteration} Tickets")
    print(f"{'='*60}")

    # Create Red ticket
    red_prior = [prior_artifacts['red']] if prior_artifacts and 'red' in prior_artifacts else None
    create_ticket(
        session_dir=session_dir,
        agent_name="red",
        iteration=iteration,
        prior_artifacts=red_prior,
        failed_gates=failed_gates
    )

    # Create Blue ticket
    blue_prior = [prior_artifacts['blue']] if prior_artifacts and 'blue' in prior_artifacts else None
    create_ticket(
        session_dir=session_dir,
        agent_name="blue",
        iteration=iteration,
        prior_artifacts=blue_prior,
        failed_gates=failed_gates
    )

    print(f"\n✅ Iteration {iteration} tickets ready in tickets/pending/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_tickets.py <session_dir> [iteration]")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    create_iteration_tickets(session_dir, iteration=iteration)
