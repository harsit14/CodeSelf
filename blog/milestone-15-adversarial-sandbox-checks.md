# Milestone 15: Closing Some Easy Sandbox Escape Paths

This milestone was a reminder that "runs in a subprocess" is not the same thing
as "safe enough for execution-feedback RL."

The earlier sandbox step gave CodeSelf per-test outcomes and concurrent batch
execution. That made the feedback more useful. This step was about making a few
obvious adversarial cases less embarrassing.

## The `SystemExit(0)` problem

The most interesting bug was simple. If generated code can raise
`SystemExit(0)` during a test, the Python process may exit with status code 0.
To the outer runner, that can look like success even though the candidate never
actually satisfied the assertion.

That is exactly the kind of small harness issue that can poison an RL setup. The
model does not need to become generally malicious; it only needs to stumble into
a shortcut that the reward function accidentally treats as correct.

I fixed this in two layers:

- the static scan now rejects direct `SystemExit`, `exit`, and `quit` usage;
- the generated phase harness catches `SystemExit`, `KeyboardInterrupt`, and
  `GeneratorExit` and turns them into runtime errors.

The first layer catches ordinary generated code. The second layer is a backup
for cases where code reaches the harness through a path that did not go through
the scanner.

## Introspection is dangerous in Python

Python gives programs a lot of ways to look around. Attributes such as
`__class__`, `__subclasses__`, and `__globals__` can become ingredients for
escaping simple blocklists. I added conservative static rejection for those
surfaces.

This is not a proof. It is still a syntactic scan, and syntactic scans can be
bypassed. But it raises the floor for common object-introspection tricks while
we build out the container path.

## Runtime blocking for file access

The harness now also patches a small set of builtins at startup. The most
important one is `open`. Generated solutions for this benchmark distribution
are supposed to be pure Python functions, not programs that inspect the test
harness or read files from the temporary directory.

I learned a useful lesson here: blocking too many builtins can break Python
itself. My first attempt patched `exec`, which import machinery needs. The unit
tests caught that immediately because even a normal candidate stopped importing.
The final runtime block list is intentionally narrower.

## Docker runner becomes real enough to test

The Docker path used to be mostly a command builder. It now has a `run_phase`
method that writes the same `candidate.py`, `run_phase.py`, and
`sitecustomize.py` files as the local runner, then executes them through Docker.

The unit tests inject a fake command runner, so CI does not need Docker. But the
interface is now real: a final evaluator can call the same `run_phase` contract
used by the local subprocess runner.

The Docker command now asks for the isolation properties I want in final eval:
no network, read-only root, dropped capabilities, no-new-privileges, memory and
swap limits, pids limit, CPU limit, non-root user, read-only task mount, and a
small `/tmp` tmpfs.

## What still worries me

The local subprocess runner is still not a jail. Static scanning is still a
speed bump, not a security boundary. macOS resource limits are also not the same
as Linux cgroups, which matters because local development is happening on a
MacBook M5 Pro with 48GB unified memory.

So the next serious safety step is a slow integration test that builds the
Docker image and actually runs adversarial candidates inside it. The local path
is for fast iteration. The container path is where final claims should live.
