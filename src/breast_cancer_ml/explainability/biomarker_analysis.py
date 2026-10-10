import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings, logger

class MutationAnalyzer:
    """
    Analyzes mutation frequencies and their relationship with survival and subtype.
    """
    def __init__(self):
        self.figs_dir = Path("reports/figures")
        self.tables_dir = Path("reports/tables")
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def analyze_mutations(self):
        """
        Calculates mutation frequencies and generates visualization.
        """
        raw_path = Path(settings.get("data.raw_path"))
        df = pd.read_csv(raw_path)

        # Identify mutation columns
        mut_cols = [col for col in df.columns if col.endswith('_mut')]

        # Calculate frequencies
        # Assuming mutations are 1 and wild-types are 0
        freqs = df[mut_cols].mean().sort_values(ascending=False)

        freq_df = pd.DataFrame({
            "mutation": freqs.index,
            "frequency": freqs.values
        })
        freq_df.to_csv(self.tables_dir / "mutation_frequency.csv", index=False)

        # Plot Top 20
        plt.figure(figsize=(12, 8))
        sns.barplot(data=freq_df.head(20), x="frequency", y="mutation", palette="viridis")
        plt.title("Top 20 Mutation Frequencies in METABRIC Dataset")
        plt.xlabel("Frequency")
        plt.ylabel("Gene Mutation")
        plt.tight_layout()
        plt.savefig(self.figs_dir / "mutation_frequency.png")
        plt.close()

        logger.info(f"Mutation frequency analysis complete. Top mutation: {freq_df.iloc[0]['mutation']}")
        return freq_df

if __name__ == "__main__":
    analyzer = MutationAnalyzer()
    analyzer.analyze_mutations()
