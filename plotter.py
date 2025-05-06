import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Plotter:
    def __init__(self, json_path):
        self.jsonPath = json_path
        self.data = self._load_data()

    def _load_data(self):
        """Safely load valid JSON objects from a file with newline-separated entries."""
        data = []
        with open(self.jsonPath, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON on line {i}: {e}")
        return data

    def plot_loss(self):
        d_losses = []
        g_losses = []
        x = []

        for entry in self.data:
            kimg = entry.get("Progress/kimg", {}).get("mean", None)
            d_loss = entry.get("Loss/D/loss", {}).get("mean", None)
            g_loss = entry.get("Loss/G/loss", {}).get("mean", None)
            if kimg is not None and d_loss is not None and g_loss is not None:
                x.append(kimg)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

        plt.plot(x, d_losses, label="Discriminator Loss (D)", color='blue')
        plt.plot(x, g_losses, label="Generator Loss (G)", color='orange')
        plt.xlabel("kimg")
        plt.ylabel("Loss")
        plt.title("Discriminator and Generator Losses Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()


plotter = Plotter("C:\\Users\\doria\\Desktop\\Licenta\\Project\\models\\model_alpha\\stats.jsonl")
plotter.plot_loss()
