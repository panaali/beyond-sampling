# Preserved environment failure and narrow recovery

Original attempt84989ffa reached its NumPy version assertion and stopped with
zero diagnostic calls, no case attempts, CPU0.068239874. The pinned binary was
invoked directly and loaded NumPy2.3.5 from base site-packages. The venv launcher
uses the same resolved executable but loads NumPy2.5.4. Therefore executable-byte
identity alone was insufficient runtime binding. No numerical claim follows.

The unchanged primitive and all ten/four fixtures are retained. controls_v2.py
changes only output/protocol/gate names and adds durable invoked/resolved Python,
venv prefix and NumPy path/version before checking them. The original failed
outputs are never overwritten. The recovery uses the venv launcher with the
same external20CPU/2GiB and review lock; it requires a published fresh source gate.
Prior CPU plus recovery CPU are reported. No completed scientific run is repeated.
