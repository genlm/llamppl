import json


class SMCRecord:
    def __init__(self, n):
        self.history = []
        self.most_recent_weights = [0.0 for _ in range(n)]
        self.step_num = 1

    def prepare_string(self, s):
        # If the string doesn't have <<< and >>>, prepend <<<>>> at the front.
        if "<<<" not in s and ">>>" not in s:
            return f"<<<>>>{s}"
        return s

    def particle_dict(self, particles):
        return [
            {
                "contents": self.prepare_string(p.string_for_serialization()),
                "logweight": (
                    "-Infinity" if p.weight == float("-inf") else str(float(p.weight))
                ),
                "weight_incr": str(
                    float(p.weight) - float(self.most_recent_weights[i])
                ),
            }
            for (i, p) in enumerate(particles)
        ]

    def add_init(self, particles):
        self.history.append(
            {
                "step": self.step_num,
                "mode": "init",
                "particles": self.particle_dict(particles),
            }
        )
        self.most_recent_weights = [p.weight for p in particles]

    def add_smc_step(self, particles):
        self.step_num += 1
        self.history.append(
            {
                "step": self.step_num,
                "mode": "smc_step",
                "particles": self.particle_dict(particles),
            }
        )
        self.most_recent_weights = [p.weight for p in particles]

    def add_resample(self, ancestor_indices, particles):
        self.step_num += 1
        self.most_recent_weights = [
            self.most_recent_weights[i] for i in ancestor_indices
        ]

        self.history.append(
            {
                "mode": "resample",
                "step": self.step_num,
                "ancestors": [int(a) for a in ancestor_indices],
                "particles": self.particle_dict(particles),
            }
        )

        self.most_recent_weights = [p.weight for p in particles]

    def to_json(self):
        return json.dumps(self.history)
