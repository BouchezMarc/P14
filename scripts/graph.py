from pathlib import Path
import json
import matplotlib.pyplot as plt


MODELS_DIR = Path("models")
OUTPUT_DIR = Path("results/training_curves")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trainer_state(model_dir: Path):
    """
    Cherche le trainer_state.json dans les checkpoints du modèle.
    Retourne l'historique complet.
    """
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )

    if not checkpoints:
        return None

    # Le dernier checkpoint contient normalement l'historique complet.
    trainer_state = checkpoints[-1] / "trainer_state.json"

    if not trainer_state.exists():
        return None

    with trainer_state.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_model(model_name: str, history):
    """
    Génère un graphique avec les métriques disponibles.
    """

    log_history = history.get("log_history", [])

    if not log_history:
        print(f"[SKIP] {model_name}: aucun log_history")
        return

    # Récupération des métriques disponibles
    train_loss = []
    eval_loss = []
    learning_rate = []

    for entry in log_history:
        step = entry.get("step")

        if step is None:
            continue

        if "loss" in entry:
            train_loss.append(
                (step, entry["loss"])
            )

        if "eval_loss" in entry:
            eval_loss.append(
                (step, entry["eval_loss"])
            )

        if "learning_rate" in entry:
            learning_rate.append(
                (step, entry["learning_rate"])
            )

    # Rien à afficher
    if not train_loss and not eval_loss and not learning_rate:
        print(f"[SKIP] {model_name}: aucune métrique exploitable")
        return

    # Nombre de graphiques nécessaires
    has_loss = bool(train_loss or eval_loss)
    has_lr = bool(learning_rate)

    n_plots = int(has_loss) + int(has_lr)

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(11, 5 * n_plots),
        squeeze=False
    )

    axes = axes.flatten()
    plot_index = 0

    # ---------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------

    if has_loss:
        ax = axes[plot_index]
        plot_index += 1

        if train_loss:
            steps, values = zip(*train_loss)

            ax.plot(
                steps,
                values,
                label="Train loss"
            )

        if eval_loss:
            steps, values = zip(*eval_loss)

            ax.plot(
                steps,
                values,
                marker="o",
                label="Eval loss"
            )

        ax.set_title(f"{model_name} — Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # ---------------------------------------------------------
    # LEARNING RATE
    # ---------------------------------------------------------

    if has_lr:
        ax = axes[plot_index]

        steps, values = zip(*learning_rate)

        ax.plot(
            steps,
            values,
            label="Learning rate"
        )

        ax.set_title(f"{model_name} — Learning rate")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Learning rate")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Training metrics — {model_name}",
        fontsize=16
    )

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{model_name}_training_curves.png"

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[OK] {model_name}")
    print(f"     -> {output_file}")


def main():

    if not MODELS_DIR.exists():
        print(f"Erreur : dossier introuvable : {MODELS_DIR}")
        return

    model_dirs = sorted(
        p for p in MODELS_DIR.iterdir()
        if p.is_dir()
    )

    if not model_dirs:
        print("Aucun modèle trouvé.")
        return

    print("=" * 70)
    print("TRAINING CURVES")
    print("=" * 70)

    for model_dir in model_dirs:

        history = load_trainer_state(model_dir)

        if history is None:
            print(
                f"[SKIP] {model_dir.name}: "
                "trainer_state.json introuvable"
            )
            continue

        plot_model(
            model_dir.name,
            history
        )

    print("=" * 70)
    print("TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()