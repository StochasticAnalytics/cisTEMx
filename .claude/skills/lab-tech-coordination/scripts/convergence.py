"""
Convergence evaluation and coordination strategies.

Implements Purple Team Defenses 5, 7, 12:
- Session discovery via prompt injection
- Convergence quorum and timeout
- Pluggable convergence strategies
"""

import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from glob import glob
import json


def parse_session_from_prompt(prompt: str) -> Path:
    """
    Extract session directory from prompt prefix.

    Args:
        prompt: Agent prompt with [SESSION_DIR:/path] prefix

    Returns:
        Absolute path to session directory

    Raises:
        ValueError: If no session directory specified
    """
    match = re.search(r'\[SESSION_DIR:(.*?)\]', prompt)
    if not match:
        raise ValueError("No session directory specified in prompt")
    return Path(match.group(1))


def wait_for_agent_result(
    agent_name: str,
    session_dir: Path,
    timeout: int = 120
) -> Optional[Dict[str, Any]]:
    """
    Wait for agent to write result to outbox.

    Args:
        agent_name: Name of agent to wait for
        session_dir: Session directory
        timeout: Timeout in seconds

    Returns:
        Agent result dict or None on timeout
    """
    outbox_pattern = str(session_dir / "agents" / agent_name / "outbox" / "*.json")
    start = time.time()

    while time.time() - start < timeout:
        results = glob(outbox_pattern)
        if results:
            # Agent has written result - return latest
            latest = max(results, key=lambda p: Path(p).stat().st_mtime)
            with open(latest, 'r') as f:
                return json.load(f)
        time.sleep(1)

    return None  # Timeout


def check_gate_in_artifact(gate_id: str, artifact_path: Path) -> bool:
    """
    Check if artifact addresses a specific gate.
    Uses keyword matching + structure checks.

    Args:
        gate_id: Gate identifier
        artifact_path: Path to artifact

    Returns:
        True if gate is addressed, False otherwise
    """
    with open(artifact_path, 'r') as f:
        content = f.read()

    # Gate-specific checks
    if gate_id == 'comprehensive_coverage':
        # Must have multiple sections
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


class ConvergenceStrategy(ABC):
    """Base class for convergence strategies."""

    @abstractmethod
    def initialize(self, session_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize convergence criteria for this session."""
        pass

    @abstractmethod
    def check_convergence(
        self,
        session_dir: Path,
        iteration: int
    ) -> bool:
        """Check if coordination has converged."""
        pass

    @abstractmethod
    def synthesize(self, session_dir: Path) -> str:
        """Generate final synthesis from agent results."""
        pass


class AdversarialReviewConvergence(ConvergenceStrategy):
    """Convergence for Red/Blue adversarial review."""

    def initialize(self, session_dir: Path, config: Dict[str, Any]) -> None:
        self.session_dir = session_dir
        self.gates = [
            {'id': 'comprehensive_coverage', 'met': False},
            {'id': 'specific_examples', 'met': False},
            {'id': 'actionable_recommendations', 'met': False},
        ]
        self.max_iterations = config.get('max_iterations', 3)
        self.quality_threshold = config.get('quality_threshold', 0.8)

    def check_convergence(
        self,
        session_dir: Path,
        iteration: int
    ) -> bool:
        # Require both Red and Blue to submit
        red_result = wait_for_agent_result('red', session_dir)
        blue_result = wait_for_agent_result('blue', session_dir)

        if not (red_result and blue_result):
            return False  # Not converged, missing perspectives

        # Evaluate gates
        for gate in self.gates:
            gate_id = gate['id']
            red_addressed = check_gate_in_artifact(
                gate_id,
                Path(red_result['artifact_path'])
            )
            blue_addressed = check_gate_in_artifact(
                gate_id,
                Path(blue_result['artifact_path'])
            )

            # Gate met if BOTH agents addressed it
            gate['met'] = red_addressed and blue_addressed

        quality = sum(1 for g in self.gates if g['met']) / len(self.gates)
        return quality >= self.quality_threshold or iteration >= self.max_iterations

    def synthesize(self, session_dir: Path) -> str:
        """Aggregate Red and Blue artifacts."""
        # Simple aggregation - could be enhanced
        artifacts_dir = session_dir / "artifacts"
        artifacts = list(artifacts_dir.glob("*.md"))

        synthesis_parts = ["# Lab Tech Review Synthesis\n\n"]

        for artifact_path in artifacts:
            with open(artifact_path, 'r') as f:
                content = f.read()
            synthesis_parts.append(f"## {artifact_path.stem}\n\n")
            synthesis_parts.append(content)
            synthesis_parts.append("\n\n")

        return "".join(synthesis_parts)


class ParallelDecompositionConvergence(ConvergenceStrategy):
    """Convergence for N parallel agents (e.g., 12 file readers)."""

    def initialize(self, session_dir: Path, config: Dict[str, Any]) -> None:
        self.session_dir = session_dir
        self.num_agents = config['num_agents']
        self.quorum = config.get('quorum', self.num_agents)  # Default: all must complete
        self.timeout = config.get('timeout', 300)  # 5 min per agent

    def check_convergence(
        self,
        session_dir: Path,
        iteration: int
    ) -> bool:
        # Wait for quorum of agents to complete
        completed_agents = []
        start = time.time()

        while time.time() - start < self.timeout:
            for i in range(self.num_agents):
                agent_name = f"reader_{i}"
                result = wait_for_agent_result(agent_name, session_dir, timeout=1)
                if result and agent_name not in completed_agents:
                    completed_agents.append(agent_name)

            if len(completed_agents) >= self.quorum:
                return True  # Quorum reached

            time.sleep(1)

        # Timeout: Check if we have enough for partial synthesis
        return len(completed_agents) >= (self.quorum * 0.8)  # 80% quorum

    def synthesize(self, session_dir: Path) -> str:
        """Aggregate all available reader artifacts."""
        artifacts_dir = session_dir / "artifacts"
        artifacts = sorted(artifacts_dir.glob("*.md"))

        synthesis_parts = ["# Parallel Analysis Synthesis\n\n"]

        for i, artifact_path in enumerate(artifacts):
            with open(artifact_path, 'r') as f:
                content = f.read()
            synthesis_parts.append(f"## Section {i+1}\n\n")
            synthesis_parts.append(content)
            synthesis_parts.append("\n\n")

        return "".join(synthesis_parts)


class ConvergenceEvaluator:
    """Main convergence evaluation orchestrator."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

    def evaluate_gates(
        self,
        red_result: Dict[str, Any],
        blue_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate convergence gates using both perspectives.

        Args:
            red_result: Red agent result
            blue_result: Blue agent result

        Returns:
            Convergence data with gate status
        """
        convergence_path = self.session_dir / "convergence.json"

        if convergence_path.exists():
            with open(convergence_path, 'r') as f:
                convergence = json.load(f)
        else:
            # Default convergence gates
            convergence = {
                'convergence_gates': [
                    {'id': 'comprehensive_coverage', 'met': False},
                    {'id': 'specific_examples', 'met': False},
                    {'id': 'actionable_recommendations', 'met': False},
                ],
                'quality_threshold': 0.8,
            }

        for gate in convergence['convergence_gates']:
            gate_id = gate['id']

            # Check if both agents addressed this gate
            red_addressed = check_gate_in_artifact(
                gate_id,
                Path(red_result['artifact_path'])
            )
            blue_addressed = check_gate_in_artifact(
                gate_id,
                Path(blue_result['artifact_path'])
            )

            # Gate met if BOTH agents addressed it
            gate['met'] = red_addressed and blue_addressed

        # Calculate quality score
        num_met = sum(1 for g in convergence['convergence_gates'] if g['met'])
        quality = num_met / len(convergence['convergence_gates'])
        convergence['current_quality'] = quality

        return convergence


def select_convergence_strategy(
    task_type: str,
    config: Dict[str, Any]
) -> ConvergenceStrategy:
    """
    Select convergence strategy based on task type.

    Args:
        task_type: Type of coordination task
        config: Configuration for strategy

    Returns:
        Initialized convergence strategy

    Raises:
        ValueError: If unknown task type
    """
    strategies = {
        'adversarial_review': AdversarialReviewConvergence,
        'parallel_decomposition': ParallelDecompositionConvergence,
    }

    strategy_class = strategies.get(task_type)
    if not strategy_class:
        raise ValueError(f"Unknown task type: {task_type}")

    strategy = strategy_class()
    # Note: initialize() should be called with session_dir after creation
    return strategy
