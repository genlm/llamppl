import json

from llamppl.inference.smc_record import SMCRecord


class FakeParticle:
    """A particle whose serialization moves a `<<<...>>>` highlight, as `examples/haiku.py` does."""

    def __init__(self):
        self.weight = 0.0
        self.prefix = ""
        self.highlight = ""

    def grow(self, chunk):
        self.prefix += self.highlight
        self.highlight = chunk

    def string_for_serialization(self):
        return f"{self.prefix}<<<{self.highlight}>>>"


def rebuild(history):
    """Rebuild every step's full strings from the record, as `html/smc.html` does."""
    steps = []
    for i, step in enumerate(history):
        row = []
        for j, p in enumerate(step["particles"]):
            if i == 0:
                parent = None
            elif step["mode"] == "resample":
                parent = steps[i - 1][step["ancestors"][j]]
            else:
                parent = steps[i - 1][j]
            row.append((parent[: p["keep"]] if parent else "") + p["contents_incr"])
        steps.append(row)
    return steps


def test_record_round_trips_a_moving_highlight():
    particles = [FakeParticle() for _ in range(3)]
    record = SMCRecord(len(particles))
    ancestors = [2, 0, 2]
    truth = []

    for step, chunks in enumerate([["a ", "b ", "c "], ["dd ", "ee ", "ff "]]):
        for p, chunk in zip(particles, chunks):
            p.grow(chunk)
        truth.append([p.string_for_serialization() for p in particles])
        record.add_init(particles) if step == 0 else record.add_smc_step(particles)

    for p, chunk in zip(particles, ["g ", "h ", "i "]):
        p.grow(chunk)
    particles = [particles[i] for i in ancestors]
    truth.append([p.string_for_serialization() for p in particles])
    record.add_resample(ancestors, particles)

    assert rebuild(json.loads(record.to_json())) == truth


def test_record_stores_only_the_diff():
    p = FakeParticle()
    record = SMCRecord(1)
    for chunk, add in [
        ("hello ", record.add_init),
        ("world ", record.add_smc_step),
        ("again ", record.add_smc_step),
    ]:
        p.grow(chunk)
        add([p])

    history = json.loads(record.to_json())
    assert history[0]["particles"][0] == {
        "keep": 0,
        "contents_incr": "<<<hello >>>",
        "logweight": "0.0",
        "weight_incr": "0.0",
    }
    # Only what the settled prefix and the moved highlight cost is stored.
    assert history[2]["particles"][0]["keep"] == len("hello ")
    assert history[2]["particles"][0]["contents_incr"] == "world <<<again >>>"
