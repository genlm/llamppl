import json
import os


class SMCRecord:
    """JSON record of an SMC run for ``html/smc.html``.

    A step stores each particle's serialization as a diff against its ancestor's --
    ``keep`` shared leading characters plus ``contents_incr`` -- so the record stays
    linear in sequence length; the viewer rebuilds the strings along the ancestor
    chain.
    """

    def __init__(self, n):
        self.history = []
        self.most_recent_weights = [0.0 for _ in range(n)]
        self.recorded = ["" for _ in range(n)]
        self.step_num = 1

    def particle_dict(self, particles):
        out = []
        for i, p in enumerate(particles):
            s, prev = p.string_for_serialization(), self.recorded[i]
            keep = (
                len(prev)
                if s.startswith(prev)
                else len(os.path.commonprefix([prev, s]))
            )
            out.append(
                {
                    "keep": keep,
                    "contents_incr": s[keep:],
                    "logweight": (
                        "-Infinity"
                        if p.weight == float("-inf")
                        else str(float(p.weight))
                    ),
                    "weight_incr": str(
                        float(p.weight) - float(self.most_recent_weights[i])
                    ),
                }
            )
            self.recorded[i] = s
        return out

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
        # A forked row diffs against the string its ancestor recorded.
        self.recorded = [self.recorded[i] for i in ancestor_indices]

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
