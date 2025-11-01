#!/usr/bin/env python3
"""
Comprehensive tests for Phase 3: Convergence & Coordination Strategies

Tests all convergence defenses:
- Defense 5: Session discovery via prompt injection
- Defense 7: Convergence quorum and timeout
- Defense 12: Pluggable convergence strategies
"""

import sys
import unittest
import tempfile
import shutil
import os
import time
import json
from pathlib import Path
from threading import Thread

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from convergence import (
    parse_session_from_prompt,
    wait_for_agent_result,
    check_gate_in_artifact,
    select_convergence_strategy,
    AdversarialReviewConvergence,
    ParallelDecompositionConvergence,
    ConvergenceEvaluator
)


class TestSessionDiscovery(unittest.TestCase):
    """Test Defense 5: Session discovery via prompt injection"""

    def test_parse_valid_session_path(self):
        """Extract session directory from valid prompt"""
        prompt = "[SESSION_DIR:/tmp/test_session_123] Perform critical analysis..."
        session_dir = parse_session_from_prompt(prompt)

        self.assertEqual(session_dir, Path("/tmp/test_session_123"))

    def test_parse_session_with_spaces(self):
        """Handle session paths with spaces"""
        prompt = "[SESSION_DIR:/tmp/my session/dir] Analyze this..."
        session_dir = parse_session_from_prompt(prompt)

        self.assertEqual(session_dir, Path("/tmp/my session/dir"))

    def test_parse_relative_path(self):
        """Extract relative path from prompt"""
        prompt = "[SESSION_DIR:./relative/path] Review this code"
        session_dir = parse_session_from_prompt(prompt)

        self.assertEqual(session_dir, Path("./relative/path"))

    def test_parse_missing_session_raises(self):
        """Missing session directory raises ValueError"""
        prompt = "Perform analysis without session directory"

        with self.assertRaises(ValueError) as context:
            parse_session_from_prompt(prompt)

        self.assertIn("No session directory", str(context.exception))

    def test_parse_malformed_prefix(self):
        """Malformed prefix raises ValueError"""
        prompt = "SESSION_DIR:/tmp/bad Review this"  # Missing brackets

        with self.assertRaises(ValueError):
            parse_session_from_prompt(prompt)


class TestConvergenceQuorum(unittest.TestCase):
    """Test Defense 7: Convergence quorum and timeout"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create agent outbox directories
        (self.session_dir / "agents" / "red" / "outbox").mkdir(parents=True)
        (self.session_dir / "agents" / "blue" / "outbox").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_agent_result(self, agent_name: str, result: dict, delay: float = 0):
        """Helper to write agent result after delay"""
        time.sleep(delay)
        outbox_dir = self.session_dir / "agents" / agent_name / "outbox"
        result_path = outbox_dir / f"result_{int(time.time()*1000)}.json"

        with open(result_path, 'w') as f:
            json.dump(result, f)

    def test_wait_for_immediate_result(self):
        """Agent result available immediately"""
        result_data = {"status": "completed", "artifact_path": "/tmp/artifact.md"}
        self._write_agent_result("red", result_data)

        result = wait_for_agent_result("red", self.session_dir, timeout=5)

        self.assertIsNotNone(result)
        self.assertEqual(result['status'], "completed")

    def test_wait_for_delayed_result(self):
        """Agent result arrives after short delay"""
        result_data = {"status": "completed"}

        # Write result after 2 seconds in background
        thread = Thread(target=self._write_agent_result, args=("red", result_data, 2))
        thread.start()

        start = time.time()
        result = wait_for_agent_result("red", self.session_dir, timeout=10)
        elapsed = time.time() - start

        thread.join()

        self.assertIsNotNone(result)
        self.assertGreater(elapsed, 1.5)  # Waited for result
        self.assertLess(elapsed, 5)  # But didn't wait full timeout

    def test_wait_timeout_no_result(self):
        """Timeout when agent doesn't produce result"""
        start = time.time()
        result = wait_for_agent_result("red", self.session_dir, timeout=2)
        elapsed = time.time() - start

        self.assertIsNone(result)
        self.assertGreater(elapsed, 1.5)  # Waited full timeout

    def test_wait_returns_latest_result(self):
        """Returns latest result when multiple exist"""
        self._write_agent_result("red", {"iteration": 1})
        time.sleep(0.1)  # Ensure different mtimes
        self._write_agent_result("red", {"iteration": 2})

        result = wait_for_agent_result("red", self.session_dir, timeout=5)

        self.assertEqual(result['iteration'], 2)


class TestGateEvaluation(unittest.TestCase):
    """Test gate checking for convergence criteria"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_artifact(self, content: str) -> Path:
        """Helper to create test artifact"""
        artifact_path = self.test_dir / "test_artifact.md"
        with open(artifact_path, 'w') as f:
            f.write(content)
        return artifact_path

    def test_comprehensive_coverage_met(self):
        """Artifact with 3+ sections passes comprehensive_coverage gate"""
        content = """
# Analysis

## Finding 1
Details...

## Finding 2
More details...

## Finding 3
Even more...
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('comprehensive_coverage', artifact_path)

        self.assertTrue(result)

    def test_comprehensive_coverage_not_met(self):
        """Artifact with <3 sections fails comprehensive_coverage gate"""
        content = """
# Analysis

## Finding 1
Only one section
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('comprehensive_coverage', artifact_path)

        self.assertFalse(result)

    def test_specific_examples_code_blocks(self):
        """Artifact with code blocks passes specific_examples gate"""
        content = """
# Analysis

Example code:
```python
def example():
    pass
```
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('specific_examples', artifact_path)

        self.assertTrue(result)

    def test_specific_examples_example_keyword(self):
        """Artifact with 'Example:' keyword passes specific_examples gate"""
        content = """
# Analysis

Example: When the user clicks the button, the system should...
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('specific_examples', artifact_path)

        self.assertTrue(result)

    def test_specific_examples_not_met(self):
        """Artifact without examples fails specific_examples gate"""
        content = """
# Analysis

General discussion without concrete scenarios.
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('specific_examples', artifact_path)

        self.assertFalse(result)

    def test_actionable_recommendations_met(self):
        """Artifact with imperative keywords passes actionable_recommendations gate"""
        content = """
# Analysis

You should implement better error handling.
The system must validate inputs.
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('actionable_recommendations', artifact_path)

        self.assertTrue(result)

    def test_actionable_recommendations_not_met(self):
        """Artifact without actionable language fails actionable_recommendations gate"""
        content = """
# Analysis

This is an interesting observation about the system.
"""
        artifact_path = self._create_artifact(content)

        result = check_gate_in_artifact('actionable_recommendations', artifact_path)

        self.assertFalse(result)


class TestStrategySelection(unittest.TestCase):
    """Test Defense 12: Pluggable convergence strategies"""

    def test_select_adversarial_review(self):
        """Selecting 'adversarial_review' returns correct strategy"""
        strategy = select_convergence_strategy('adversarial_review', {})

        self.assertIsInstance(strategy, AdversarialReviewConvergence)

    def test_select_parallel_decomposition(self):
        """Selecting 'parallel_decomposition' returns correct strategy"""
        strategy = select_convergence_strategy('parallel_decomposition', {
            'num_agents': 10
        })

        self.assertIsInstance(strategy, ParallelDecompositionConvergence)

    def test_select_unknown_strategy_raises(self):
        """Unknown strategy type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            select_convergence_strategy('unknown_strategy', {})

        self.assertIn("Unknown task type", str(context.exception))


class TestAdversarialReviewStrategy(unittest.TestCase):
    """Test AdversarialReviewConvergence strategy"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create directory structure
        (self.session_dir / "agents" / "red" / "outbox").mkdir(parents=True)
        (self.session_dir / "agents" / "blue" / "outbox").mkdir(parents=True)
        (self.session_dir / "artifacts").mkdir()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_artifact(self, name: str, content: str) -> Path:
        """Helper to create artifact file"""
        artifact_path = self.session_dir / "artifacts" / f"{name}.md"
        with open(artifact_path, 'w') as f:
            f.write(content)
        return artifact_path

    def _create_agent_result(self, agent_name: str, artifact_name: str):
        """Helper to create agent result pointing to artifact"""
        artifact_path = str(self.session_dir / "artifacts" / f"{artifact_name}.md")
        result = {
            "status": "completed",
            "artifact_path": artifact_path
        }

        outbox_dir = self.session_dir / "agents" / agent_name / "outbox"
        result_path = outbox_dir / f"result_{int(time.time()*1000)}.json"

        with open(result_path, 'w') as f:
            json.dump(result, f)

    def test_initialization(self):
        """Strategy initializes with correct defaults"""
        strategy = AdversarialReviewConvergence()
        config = {
            'max_iterations': 5,
            'quality_threshold': 0.9
        }

        strategy.initialize(self.session_dir, config)

        self.assertEqual(strategy.max_iterations, 5)
        self.assertEqual(strategy.quality_threshold, 0.9)
        self.assertEqual(len(strategy.gates), 3)

    def test_convergence_both_agents_all_gates(self):
        """Convergence when both agents meet all gates"""
        strategy = AdversarialReviewConvergence()
        strategy.initialize(self.session_dir, {})

        # Create artifacts that meet all gates
        red_content = """
## Finding 1
You should fix this.

## Finding 2
Example: Test case

## Finding 3
Must implement validation.
"""
        blue_content = """
## Strength 1
Code example:
```python
pass
```

## Strength 2
Example scenario

## Strength 3
Should add tests.
"""

        self._create_artifact("red_analysis", red_content)
        self._create_artifact("blue_analysis", blue_content)
        self._create_agent_result("red", "red_analysis")
        self._create_agent_result("blue", "blue_analysis")

        converged = strategy.check_convergence(self.session_dir, iteration=1)

        self.assertTrue(converged)

    def test_convergence_missing_agent(self):
        """No convergence when one agent missing"""
        strategy = AdversarialReviewConvergence()
        strategy.initialize(self.session_dir, {})

        # Only Red submits
        self._create_artifact("red_analysis", "## Content\nshould fix")
        self._create_agent_result("red", "red_analysis")

        converged = strategy.check_convergence(self.session_dir, iteration=1)

        self.assertFalse(converged)

    def test_max_iterations_forces_convergence(self):
        """Max iterations forces convergence even with low quality"""
        strategy = AdversarialReviewConvergence()
        strategy.initialize(self.session_dir, {'max_iterations': 2})

        # Create minimal artifacts (won't meet quality threshold)
        self._create_artifact("red_analysis", "Analysis")
        self._create_artifact("blue_analysis", "Review")
        self._create_agent_result("red", "red_analysis")
        self._create_agent_result("blue", "blue_analysis")

        converged = strategy.check_convergence(self.session_dir, iteration=2)

        self.assertTrue(converged)  # Forced by max_iterations


class TestParallelDecompositionStrategy(unittest.TestCase):
    """Test ParallelDecompositionConvergence strategy"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create reader agent outboxes
        for i in range(5):
            (self.session_dir / "agents" / f"reader_{i}" / "outbox").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_reader_result(self, reader_id: int):
        """Helper to create reader result"""
        result = {"chunk": reader_id, "status": "completed"}
        outbox_dir = self.session_dir / "agents" / f"reader_{reader_id}" / "outbox"
        result_path = outbox_dir / f"result_{int(time.time()*1000)}.json"

        with open(result_path, 'w') as f:
            json.dump(result, f)

    def test_initialization(self):
        """Strategy initializes with correct config"""
        strategy = ParallelDecompositionConvergence()
        config = {
            'num_agents': 5,
            'quorum': 4,
            'timeout': 60
        }

        strategy.initialize(self.session_dir, config)

        self.assertEqual(strategy.num_agents, 5)
        self.assertEqual(strategy.quorum, 4)
        self.assertEqual(strategy.timeout, 60)

    def test_convergence_quorum_reached(self):
        """Convergence when quorum of agents complete"""
        strategy = ParallelDecompositionConvergence()
        strategy.initialize(self.session_dir, {
            'num_agents': 5,
            'quorum': 3,
            'timeout': 10
        })

        # Have 3 out of 5 agents complete
        for i in range(3):
            self._create_reader_result(i)

        converged = strategy.check_convergence(self.session_dir, iteration=1)

        self.assertTrue(converged)


def run_tests():
    """Run all Phase 3 convergence tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSessionDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergenceQuorum))
    suite.addTests(loader.loadTestsFromTestCase(TestGateEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategySelection))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialReviewStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelDecompositionStrategy))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
