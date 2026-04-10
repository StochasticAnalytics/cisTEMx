# Tunneled Worker Fix — Investigation Notes & Change Log

**Status:** functional, validated on a real ~5hr Match Templates run.
**Branch of origin:** `claude/silly-roentgen` in the aws_tooling worktree.
**Target worktree:** `/sa_shared/git/cisTEM/worktrees/fix_volume_removal_crash`
  (branch `multiview_particlestacks_no_claude_for_review_with_ci`).
**Author of this note:** Claude, working under instruction. The user will
come back and clean this up directly in the cisTEM repo later.

This file exists because the changes in `guix_job_control.cpp`,
`match_template.cpp`, `MatchTemplatePanel.cpp`, and `defines.h` are wider
than the one "real fix" at the center, and parts of them are debug
instrumentation that should not be considered permanent. When you sit
down to tidy this up, read this file first.

---

## 1. What the fix actually fixes

Symptom (end of 2026-04-09 into early 2026-04-10):

- Multi-host cisTEM template-matching run.
- cisTEM GUI showed "N/N processes are connected."
- Workers on the **local VLAN** (salina/etna/siracusa) ran GPU code correctly.
- Workers behind SSH tunnels — `remote-grantahlquist1` (GA1 via firenze
  ProxyJump) and later `remote-aws-1..16` (AWS via Tailscale subnet router
  to VPC) — **connected once to the controller, then went silent**. Never
  hit the GPU, never produced results, never consumed GPU memory.
- Re-ordering the run profile didn't matter. Swapping position of the
  failing host had no effect.
- Position of failing host in run profile was irrelevant: when a tunneled
  host was the only non-local, only that host was silent.

Root cause (actually two stacked bugs):

**Bug A — IPv4/IPv6 silent bind fallback on the remote.**
When another user had a stale cisTEM worker bound to `0.0.0.0:3002` on
the remote host, `ssh -f -N -R 3002:localhost:<master_port> remote`
still returned **0** even with `ExitOnForwardFailure=yes`. Under the
hood sshd fell back to binding only `[::1]:3002` (the IPv6 loopback),
leaving `127.0.0.1:3002` held by the zombie. The cisTEM worker binary
uses `wxIPV4address`, so its `connect()` went to the zombie worker
instead of our tunnel. Result: silent "connected" without any data flow.

This was observed concretely on GA1: user `sshastri` had a leftover
cisTEM_job_control bound to `0.0.0.0:3002`, which silently absorbed every
IPv4 loopback connect. The tunnel *process* was alive, sshd reported
success, but no bytes reached salina.

**Bug B — master_port collision on the remote side after master election.**
Even after fixing A, the second reverse tunnel (for the elected
master's listen port) can land on a port that's in use on the remote,
silently fail the same way, and the newly-elected cisTEM master's port
would never be reachable from the tunneled worker — so the worker
"reconnects to master" succeeded at the TCP layer but hit nobody.
Worker then sat idle waiting for work that never arrived.

Fix strategy:

1. **Pre-scan ports on every remote target using a bash /dev/tcp probe
   *before* trusting `ssh -R`.** We require the remote's `127.0.0.1:<port>`
   to answer "free" (connection refused) before we even attempt the tunnel.
2. **Reserve a single uniform `master_tunnel_remote_port` that is free on
   *every* remote target**, picked high (>= 43210). This is used as the
   remote listen side of the master-port reverse tunnel, and as the port
   we tell tunneled workers to connect to when they reach the "go find the
   master" step. It is intentionally separate from `master_port` itself
   (which can, and does, collide with other cisTEM users).
3. **Set up the master-port reverse tunnel in `HandleNewSocketConnection`
   after master election**, one per remote target. The salina side still
   points at `127.0.0.1:<master_port>` (the master actually listens there);
   only the remote side uses `master_tunnel_remote_port`.
4. **In the worker branch of `HandleNewSocketConnection`, rewrite the
   master address sent to tunneled workers** (identified by
   `peer_ip == "127.0.0.1"`) to `127.0.0.1:<master_tunnel_remote_port>`.
   Local workers keep the original master_ip_address/master_port.

Validation — see `/sa_shared/.claude/projects/.../memory/project_tunneled_worker_fix_validated.md`.
Full 28-worker Match Templates run on project `first_process`, ~5hr ETA,
all 28 workers connected and running GPU code across local VLAN + GA1 +
16 AWS workers. `nvidia-smi` confirmed 100% utilization, 11145 MiB
resident on aws-11 (A10G), matching the 11329 MiB signature on GA1 (A100).

---

## 2. Files touched in this fix

### 2a. `src/programs/guix_job_control/guix_job_control.cpp` — THE FIX

This is the real code change. ~400 line diff but conceptually four
additions:

**(i) `JobControlApp` new member fields** (near `number_of_workers_already_connected`):

```cpp
wxArrayString remote_tunnel_targets;      // ssh_targets with a controller tunnel
bool          master_tunnels_established; // latch so we only do it once
wxString      master_tunnel_remote_port;  // reserved high port, free on all remotes
friend class LaunchJobThread;              // LaunchJobThread populates the above
```

Also `#include <map>` at the top.

Init in `OnInit()`: resets all three to empty/false.

**(ii) Two new static helpers + a constant block** tagged
`// BEGIN SSH_TUNNEL_HACK`:

- `RunAndCapture(cmd)` — `popen` + trim, small helper.
- `ExtractSSHTarget(cmd)` — parses `ssh [-flags...] <target>` out of a
  run_profile `command_to_run` template.
- `static const bool USE_SSH_TUNNEL = true;` — kill switch.

**(iii) `LaunchJobThread::LaunchRemoteJob()` tunnel setup block**
(right before "Build executable strings"):

- Scans `current_run_profile.run_commands` for SSH targets.
- SSH alias convention: hosts whose name starts with `remote-` are
  assumed tunneled; everything else is local. This is why the cisTEM
  run profile must use aliases like `remote-grantahlquist1`,
  `remote-aws-1..16` in the `command_to_run` template.
- For each unique remote target:
  - `ssh -o ConnectTimeout=10 <target> "echo ok"` — connectivity check,
    aborts the run with `SSH_TUNNEL_ERROR` if it fails.
  - Port-try loop starting at `next_tunnel_port = 43210`, 40 attempts.
    Each port is pre-checked by opening `/dev/tcp/127.0.0.1/<port>` on
    the remote via a one-shot SSH. Only if the probe says "free" do we
    run `ssh -f -N -o ExitOnForwardFailure=yes -R <port>:localhost:<port_number>`.
  - Populates `tunnel_map[target] = {"127.0.0.1", "<remote_port>"}`.
- After all tunnels are set up, a verification loop (5 attempts, 1s
  apart) SSHes each remote and checks the tunnel port is listening via
  another `/dev/tcp` probe. Non-verified tunnels emit a warning but
  don't abort (ssh fork may just need more time — seen in practice).
- Publishes the target list + false latch to
  `main_thread_pointer->remote_tunnel_targets` and
  `master_tunnels_established`.
- **Reserves `master_tunnel_remote_port`** — iterates `try_port` from
  `next_tunnel_port`, probes EVERY target until one port reports "free"
  on all of them, then saves it. Aborts the run with
  `SSH_TUNNEL_ERROR: could not reserve a master-tunnel port free on all remotes`
  if 80 attempts fail.

**(iv) Per-command executable selection** later in the same loop:

- `ExtractSSHTarget` on each `command_to_run` template.
- If the target has a tunnel in `tunnel_map`, build `executable` with
  `127.0.0.1:<tunnel_port>` instead of the default ip/port.
- Else use `executable_default`.

**(v) `HandleNewSocketConnection` master election branch**
(the existing `if ( have_assigned_master == false )` block):

Right after receiving `master_ip_address` and `master_port` from the
newly-elected master, iterate `remote_tunnel_targets` and for each:

```cpp
ssh -f -N -o ExitOnForwardFailure=yes \
    -R <master_tunnel_remote_port>:127.0.0.1:<master_port> <target>
```

Sets `master_tunnels_established = true` after the loop. Warnings on
ssh failure but doesn't abort.

**(vi) `HandleNewSocketConnection` worker branch**
(the `else // we have a master` block):

```cpp
wxString effective_master_ip   = master_ip_address;
wxString effective_master_port = master_port;
if ( peer_ip == "127.0.0.1" ) {                 // tunneled worker
    effective_master_ip = "127.0.0.1";
    if ( ! master_tunnel_remote_port.IsEmpty() ) {
        effective_master_port = master_tunnel_remote_port;
    }
}
SendwxStringToSocket(&effective_master_ip, new_connection);
SendwxStringToSocket(&effective_master_port, new_connection);
```

Local workers are untouched (their `peer_ip` is the local VLAN address).

**Grep marker for cleanup:** I tagged most of the new code with comments
starting `Revert-debug-ga1:` and `SSH_TUNNEL_HACK`. Search for either
string to find all the related points.

### 2b. `src/programs/guix_job_control/guix_job_control.cpp` — DEBUG NOISE

Heavily peppered with `MyDebugPrintGA1(...)` calls through `OnInit`,
`LaunchJobThread::Entry`, `LaunchRemoteJob`, `HandleNewSocketConnection`,
`HandleSocketJobPackage`, `HandleSocketDisconnect`. These were essential
during the investigation (the IPv4/IPv6 bug is invisible without per-step
logging) and now compile to no-ops because of the `defines.h` change.

**Cleanup TODO:** decide per-callsite whether to delete or keep. The
ones in `LaunchRemoteJob`'s tunnel scan/verify/port-reserve loop were
extremely valuable for debugging and I would keep them gated behind a
real debug flag — not MyDebugPrintGA1, because that name is specific to
the GA1 investigation. Maybe promote them to `wxLogDebug` or gate them
behind the existing `USE_SSH_TUNNEL` with a verbose mode.

### 2c. `src/core/defines.h` — MyDebugPrintGA1 → no-op

```cpp
// Revert-debug-ga1: no-op after GA1 silent-worker investigation was resolved.
// Call sites remain in the code but compile away. To re-enable, restore the
// original body: {wxPrintf("Revert-debug-ga1 [%s] %s:%i %s | ", ...); ...}
#define MyDebugPrintGA1(...)    do { } while (0)
```

Added in BOTH branches of the `#ifdef DEBUG` / `#else`, so call sites
compile away regardless of build mode. Call sites remain because they
were useful when the fix was being built. **Cleanup option:** delete
the macro and all call sites once you're satisfied the fix is stable.

### 2d. `src/programs/match_template/match_template.cpp` — pure debug noise

Only MyDebugPrintGA1 additions — **no functional changes**. Every call
site is at an entry/exit point or around a socket send, so they trace
worker lifecycle. Already no-op.

**Cleanup option:** delete them all. Zero risk — they don't affect
control flow.

Also note: some of these prints originally used `%ld` with `long` but
wxWidgets' `wxArgNormalizer` asserts strictly on type mismatches. I fixed
all of them to use `(int)` casts with `%d`. The user's memory
`feedback_wx_format_longs.md` covers this rule — see it for the pattern.

### 2e. `src/gui/MatchTemplatePanel.cpp` — mostly debug noise

Same story — MyDebugPrintGA1 call sites at entry/exit of most
MatchTemplatePanel member functions. No functional changes. One
noteworthy exception in `ProcessResult()`: there is an **inline comment**
explaining why the print there was intentionally omitted (it fires per
result, hundreds of thousands of times per run, floods the log).

Also fires explicitly NOT present in `OnUpdateUI` for the same
high-frequency reason — there's a commented-out line with a note.

**Cleanup option:** delete them all. Same zero-risk story as 2d.

**Left out of this commit:** a stray `#define BATCH_HIGH_RES_EXPERIMENT`
toggle at the top of the file was sitting in the working tree when I
got there — it is NOT related to the tunneled-worker fix and NOT mine.
I reverted it back to `// #define BATCH_HIGH_RES_EXPERIMENT` before
committing so nothing in my commit touches that feature flag.

---

## 3. Files NOT committed (user's unrelated in-progress work)

These had local modifications in the working tree when the tunneled-worker
fix work started. I did not stage or commit them:

- `src/core/conjugate_gradient_refactor2026.h` — user enabled
  `#define cisTEM_USE_CG_REFACTOR_2026`.
- `src/programs/refine3d/refine3d.cpp` — user added `#error` to the
  non-CG-refactor branch so the old path fails the build.
- `.mcp.json` (untracked) — Claude MCP config.
- `src/**/._*` macOS AppleDouble metadata files (untracked).

None of these are related to the template-matching / tunnel work. If
you're reading this after a `git pull` brought in the tunneled-worker
commit, those files should still be in your working tree as uncommitted
local changes.

---

## 4. Cleanup TODO for when you come back to this in the cisTEM repo

Ordered by "most obviously correct" first:

1. **Rename `MyDebugPrintGA1` to something project-neutral** or delete
   it entirely. The "GA1" tag was a debugging scar from the original
   remote-grantahlquist1 silent-worker investigation; it has no place
   in upstream cisTEM. Suggested: just delete the macro and remove all
   call sites.

2. **Decide the fate of the new fields on `JobControlApp`**:
   - `remote_tunnel_targets` — keep, rename if you like.
   - `master_tunnel_remote_port` — keep.
   - `master_tunnels_established` — keep (one-shot latch).
   - `friend class LaunchJobThread` — this was needed because
     `LaunchJobThread` populates those fields from outside
     `JobControlApp`. Consider refactoring to a proper accessor
     pair instead of friend, if that's the style preference.

3. **Factor the tunnel setup out of `LaunchRemoteJob`** — it's ~150
   lines now, mostly tunnel orchestration, before any job launching
   starts. A `SetupRemoteTunnels(run_profile, out_tunnel_map,
   out_master_tunnel_remote_port)` helper would make the function
   much more readable.

4. **`USE_SSH_TUNNEL` gate** — either promote to a runtime option
   driven by the run profile (so a user without any `remote-` aliases
   doesn't even hit the scan loop), or gate it behind a compile-time
   feature flag that matches cisTEM project conventions.

5. **`remote-` SSH alias prefix convention** — should be documented
   somewhere the user can find. There's nothing in the cisTEM run
   profile UI that explains it; a user adding a remote worker for the
   first time needs to know this rule. Candidates:
   - A one-line hint on the run profile panel.
   - A section in the cisTEM user guide.
   - At minimum a comment block at the top of `guix_job_control.cpp`
     summarizing the tunnel model.

6. **IPv6 bind hardening** — the `/dev/tcp` precheck catches the
   specific zombie-worker case but the underlying sshd fallback behavior
   is nasty. Consider passing `-o AddressFamily=inet` or explicit
   `127.0.0.1` in the forward spec (some ssh versions need
   `-R 127.0.0.1:<port>:localhost:<port_number>` to prevent IPv6 fallback).
   I did NOT make this change in the working fix because the precheck
   is sufficient in practice and the explicit bind address behaves
   differently across OpenSSH versions — test on the specific
   grantahlquist1 ssh version before relying on it.

7. **Tunnel process lifetime** — the `ssh -f -N` processes are orphaned
   when cisTEM_job_control exits (they don't have a parent to signal
   them). There's a companion `awst tunnel-cleanup` command on the
   aws_tooling side that reads `~/.config/aws_tooling/tunnel_pids.txt`
   to clean them up, but that file is populated by a different commit
   (`ab14d7a` in the aws_tooling worktree) which is NOT in cisTEM.
   Decide: write the PID file from guix_job_control itself, or rely on
   an external sweeper, or teach the controller to kill its own
   tunnels on exit (signal handler). The current code leaks
   `ssh -f -N` processes on controller exit.

8. **Run profile pathing hack** — the AWS worker binary path is
   `/home/ubuntu/bin/match_template_gpu` while GA1/local is
   `/scratch/salina/bin/match_template_gpu`. This is encoded per-host
   in the cisTEM run profile `command_to_run` templates. Works but
   fragile. Long-term, a templated binary path per-host alias would be
   cleaner.

9. **Thread count** — `cisTEM` warned "you are only using one thread on
   the GPU. Suggested minimum is 2" during the validation run. The
   trailing `4` on each worker exec line is the thread-per-GPU arg from
   the run profile. Bump on the next run. Not a correctness issue.

---

## 5. How to reproduce the validation

Preconditions:
- `~/.ssh/config` has `remote-grantahlquist1` (via firenze ProxyJump)
  and `remote-aws-1..16` (via awst gateway VPC CIDR + key).
- `gpu-workers` fleet is running (see `awst fleet status`).
- `/scratch/salina/bin/match_template_gpu` exists on GA1 and local hosts.
- `/home/ubuntu/bin/match_template_gpu` exists on all 16 AWS workers.
- cisTEM run profile `first_process` configured with SSH `command_to_run`
  lines using the `remote-*` aliases for the tunneled hosts.

Procedure:
1. Open cisTEM GUI on salina.
2. Load project `first_process`.
3. Go to Match Templates panel.
4. Start a Match Templates run using the 28-worker run profile.

Expected confirmation lines in the GUI log:
- `SSH_TUNNEL: Setting up tunnel for remote-grantahlquist1`
- `SSH_TUNNEL: Setting up tunnel for remote-aws-*`
- `SSH_TUNNEL: reserved master-tunnel port <N>` (usually 43225 or nearby)
- `All 28 processes are connected.`
- `All workers have re-connected to the master.`
- Worker exec lines showing `ssh remote-aws-15 "... /home/ubuntu/bin/match_template_gpu 127.0.0.1 43225 <job_code> 4"`

Expected GPU signal (on any tunneled host):
- `nvidia-smi` shows 100% util, ~11 GB used by `match_template_gpu`.

---

## 6. Related aws_tooling files

- `/Users/himesb/git/aws_tooling/.claude/worktrees/silly-roentgen/REMOTE_WORKERS.md` —
  design doc explaining the tunnel architecture (user-written).
- `/Users/himesb/git/aws_tooling/.claude/worktrees/silly-roentgen/src/aws_tooling/cli/tunnel.py` —
  `awst tunnel-cleanup` CLI command. Also a `claude/silly-roentgen` branch addition.
- `/sa_shared/.claude/projects/-Users-himesb-git-aws-tooling/memory/project_tunneled_worker_fix_validated.md` —
  validation record for the 2026-04-10 28-worker run.

---

## 7. Who/what to blame in commit archaeology

The original three-executable strategy (hardcoded `Contains("grantahlquist1")`
and `aws-host-` prefixes) was commits `a0265ee` → `0405f08` on the
aws_tooling `claude/silly-roentgen` branch (NOT in cisTEM). Those are
historical and have been superseded by the cleaner uniform-port approach
documented above. Do not resurrect them. The stale binary at
`/sa_shared/software/current_cistem_dir/cisTEM_job_control` with mtime
2026-04-09 12:47:00 was the `0405f08` binary — it was the root cause of a
full evening of confusion because the user was running it while we were
reading HEAD source that contained later commits.

The "forensic bisect" plan at
`/sa_shared/.claude/plans/soft-wandering-wirth.md` captures the misguided
turns taken before the stale-binary issue was noticed. Worth reading if
you're ever in a "code and symptoms don't match" situation — the answer
might be that the running binary isn't what you think it is.
