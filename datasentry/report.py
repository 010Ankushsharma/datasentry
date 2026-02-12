from typing import Dict


class Report:
    """
    Inspection report returned by analyze().
    """

    def __init__(self, issues: Dict):
        self.issues = issues
        self.score = self._compute_score()
        self.fixer = None  # injected later

    def _compute_score(self) -> int:
        score = 100

        for issue in self.issues.values():
            if issue["is_problematic"]:
                score -= min(30, issue["severity"] * 50)

        return max(int(score), 0)

    def show(self) -> None:
        """
        Pretty-print report to console.
        """

        print("\n=== DataSentry Report ===\n")

        for name, result in self.issues.items():
            print(f"[{name.upper()}]")
            for k, v in result.items():
                print(f"  {k}: {v}")
            print()

        print(f"Overall Health Score: {self.score}/100\n")
