# Visualization

We provide a Web interface for visualizing the execution of a sequential Monte Carlo algorithm,
based on contributions from Maddy Bowers and Jacob Hoover.

First, update your model to support visualization by implementing the [`string_for_serialization`](llamppl.modeling.Model.string_for_serialization) method.
Return a string that summarizes the particle's current state.

To run the interface, change to the `html` directory and run `python -m http.server`. This will start serving
the files in the `html` directory at localhost:8000. (If you are SSH-ing onto a remote machine, you may need
port forwarding. Visual Studio Code automatically handles this for some ports, including 8000.)
Then, when calling [`smc_standard`](llamppl.inference.smc_standard), set `visualization_dir`
to the path to the `html` directory. A JSON record of the run will automatically be saved
to that directory, and a URL will be printed to the console (`http://localhost:8000/smc.html?path=$json_file`).
