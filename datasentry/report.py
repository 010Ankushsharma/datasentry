# datasentry/report.py
class Report:

    def __init__(self, issues):
        self.issues = issues
        self.score = self._compute_score()

    def _compute_score(self):

        score = 100

        penalties = {
            "imbalance": 20,
            "outliers": 15,
            "distribution_shift": 25,
            "data_leakage": 30,
            "label_noise": 20
        }

        for key, penalty in penalties.items():
            issue = self.issues.get(key, {})
            if issue.get("status") == "warning":
                score -= penalty

        return max(score, 0)

    def show(self):

        print("\n=== DATA SENTRY REPORT ===\n")

        for name, result in self.issues.items():
            print(f"[{name.upper()}]")
            for k, v in result.items():
                print(f"{k}: {v}")
            print()

        print("DATA HEALTH SCORE:", self.score, "/ 100")
