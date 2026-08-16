{
  # Keep this line accurate and one line long: `nix flake metadata` prints it,
  # and it is the first thing a cold agent learns about the repo.
  description = "ir_racecar -- OpenCV image-join / IR lane-detection experiments for an MIT-RACEcar-style robot, in a deprecated ROS 1 catkin workspace. Run `nix flake show` for the command map.";

  # nixpkgs is the only input, and nothing here wants a second one: the canonical
  # block below already defines `systems` and `forAllSystems`, so the system list
  # lives in this file rather than in another input's copy of it.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    # `...` rather than a closed pattern, so a second input can be added later
    # without editing this line. `self` is mandatory: the canonical block anchors
    # every verb on it.
    { self, nixpkgs, ... }:
    let
      lib = nixpkgs.lib;

      # ======================================================================
      # PER-REPO BLOCK 5 -- the name in the interactive dev-shell banner
      # ======================================================================
      repoName = "ir_racecar";

      # ======================================================================
      # PER-REPO BLOCK 1 -- the toolchain
      # ======================================================================
      # This repo ships no Python manifest: `git ls-files` matches no
      # requirements.txt, setup.py, pyproject.toml, Pipfile or .github/ workflow.
      # The dependency set below was derived by grepping the imports of the 22
      # tracked .py files (`git ls-files '*.py' | wc -l` -> 22). The third-party
      # names they import are cv2, numpy, scipy, serial, yaml and tkinter, plus
      # rospy / roslib / cv_bridge / sensor_msgs -- see GAP 1.
      #
      # Every one of the first six exists in nixpkgs, so the interpreter is built
      # with withPackages instead of bootstrapping a .venv. Consequence, and it is
      # the whole point: this shell needs no network and there is no `setup` verb
      # to forget to run. Do NOT "improve" this by adding uv and a
      # requirements.txt -- there is no upstream pin to be faithful to, so a
      # generated requirements.txt would be a new source of truth nobody wrote.
      #
      # python313 is pinned rather than python3 because in the locked nixpkgs
      # `python3` already resolves to 3.14.7 while `python313` is 3.13.15, and the
      # bytecode committed under scripts/__pycache__ is cpython-38. Pinning the
      # major keeps the shell from moving under the next agent.
      #
      # GAP 1 -- ROS is not covered, and no verb pretends otherwise. Eight tracked
      # scripts import rospy, roslib, cv_bridge or sensor_msgs: scripts/
      # camera_join.py, scripts/publish_camera_info.py, and under
      # scripts/to_be_deleted/ the files camera_join_subscribe.py,
      # image_stitching_ROS.py, image_stitching_ROS_concat.py,
      # image_stitching_ROS_openCV.py, image_stitching_ROS_staticMatrix.py and
      # writing_image_test.py. catkin_ws/src/camera_tests is a catkin package
      # (CMakeLists.txt + package.xml) and scripts/ros_start.batch runs
      # `catkin_make`, then `source devel/setup.bash`, then `roslaunch
      # camera_processor.launch`. None of that is reachable from here: the locked
      # nixpkgs has no `rospy` attribute at all (`nix eval nixpkgs#rospy` and
      # `nix eval nixpkgs#python313Packages.rospy` both fail with "does not
      # provide attribute"), so nothing in this shell can import those modules.
      #
      # GAP 2 -- OpenCV highgui is absent. Measured in this shell, with cv2
      # 4.13.0, `cv2.namedWindow("t")` raises:
      #   error: (-2:Unspecified error) The function is not implemented. Rebuild
      #   the library with Windows, GTK+ 2.x or Cocoa support.
      # That hits the 8 of 22 scripts that call imshow/namedWindow/waitKey:
      # camera_test_no_ROS.py, showCamNoGUI.py, showFPS.py, and under
      # to_be_deleted/ image_stitching_feature.py, image_stitching_test.py,
      # image_test.py, showCam.py and writing_image_test.py.
      #
      # `pkgs.python313Packages.opencv4.override { enableGtk3 = true; }` does
      # evaluate, to a different derivation than the default opencv4, and
      # cache.nixos.org does not have that output -- `nix path-info --store
      # https://cache.nixos.org <outPath>` answers "is not valid" for the override
      # and succeeds for the default -- so switching it on means building OpenCV
      # from source locally. Left off on purpose: the affected scripts also need
      # real cameras (camera_test_no_ROS.py opens VideoCapture(0) and
      # VideoCapture(2)) and a display, so a headless agent gains nothing from the
      # rebuild. Turn it on locally if you have the hardware in front of you.
      #
      # Not covered and not coverable: scripts/pwm_Gen.ino and
      # scripts/pwm_Gen_frame_sync.ino (Arduino sketches for the external IR PWM
      # generator) and the four cad/*.kicad_* files -- those need the Arduino IDE
      # and KiCad plus real hardware, not a dev shell.
      #
      # Explicit `pkgs.foo`, never `with pkgs; [ ... ]`: when an attr disappears in
      # a nixpkgs bump, `with` reports a bare undefined identifier with no hint of
      # which set it came from, and the name is not greppable.
      toolchain = pkgs: [
        # ---- this repo's ecosystem ----
        (pkgs.python313.withPackages (ps: [
          ps.numpy
          ps.opencv4
          ps.pyserial
          ps.pyyaml
          ps.scipy
          # tkinter is a separate derivation in nixpkgs, not part of the base
          # interpreter. Three scripts need it: showCamNoGUI.py and
          # to_be_deleted/showCam.py do `from tkinter import *` (showCam.py also
          # `from tkinter import ttk`), toSerialApp.py does `import tkinter as
          # tk`. Without this the interpreter alone would fail there.
          ps.tkinter
        ]))
        pkgs.ruff

        # ---- generic helpers, on PATH for whoever is working in here ----
        # No verb below invokes any of these; they are here so the shell is a
        # usable place to stand.
        pkgs.git
        pkgs.jq
        pkgs.gnumake
      ];

      # ======================================================================
      # PER-REPO BLOCK 2 -- libraries that get dlopened, not linked
      # ======================================================================
      # Nothing in the toolchain above needs this. Measured: inside the dev shell,
      # `unset LD_LIBRARY_PATH; python3 -c 'import cv2, numpy, scipy, serial,
      # yaml, tkinter'` succeeds -- nixpkgs has already linked those extension
      # modules against their store copies.
      #
      # The list is kept anyway as the escape hatch for a binary that dlopens
      # libstdc++ or libz at runtime and was not built by nixpkgs. That is a
      # choice, not a measurement; keep it at these two, because LD_LIBRARY_PATH
      # is a blunt instrument.
      nativeLibs = pkgs: [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];

      # ======================================================================
      # PER-REPO BLOCK 3 -- constant environment variables
      # ======================================================================
      # Constants only, applied identically to the dev shell and to every wrapper.
      envVars = pkgs: {
        # scripts/__pycache__ holds five cpython-38 .pyc files that are TRACKED in
        # git. Running anything under this shell's 3.13 would drop cpython-313
        # files beside them. Those new files would not show up in `git status`
        # (.gitignore line 18 is `__pycache__/`, and `nix flake metadata` still
        # reports the tree clean with one present -- measured), so this is
        # tidiness rather than a dirty-tree fix.
        PYTHONDONTWRITEBYTECODE = "1";

        # Part of the anchoring invariant, not a performance knob. Measured with
        # ruff 0.16.2: `ruff check sub/a.py` writes its `.ruff_cache` beside the
        # PROCESS's cwd, not beside the file it was handed, so a verb run from
        # somebody else's directory littered a cache there -- a write outside this
        # repo. And when the cwd is not writable (which $REPO_ROOT is not, when it
        # falls back to this flake's store snapshot) ruff aborts before reporting a
        # single finding -- measured in the snapshot: "error: Failed to initialize
        # cache at /nix/store/...-source/.ruff_cache: Read-only file system (os
        # error 30)".
        #
        # Note the value: ruff parses this as a bool and rejects "1" with
        # "error: invalid value '1' for '--no-cache'". Nothing is lost by disabling
        # it -- a cold `ruff check .` over this repo's 22 files measured 0.017 s
        # wall.
        RUFF_NO_CACHE = "true";
      };

      # ======================================================================
      # PER-REPO BLOCK 4 -- the command map
      # ======================================================================
      # THE single source of truth: it generates `apps` (so `nix run .#lint`
      # works), the `dev-*` wrappers on PATH inside the shell, and `dev-help`.
      # Nothing is written twice, so `nix flake show` can never disagree with what
      # `dev-lint` actually runs.
      #
      # The house vocabulary is setup/build/test/lint/fmt/run, and a verb the repo
      # has no meaning for is OMITTED rather than stubbed, because absence is
      # information. Missing here, and why:
      #   setup  nothing to install -- the interpreter above is complete offline
      #   build  the only buildable artifact is the catkin package, and that needs
      #          ROS 1 (GAP 1). `catkin_make` cannot run in here.
      #   test   there is no test suite. No test runner is configured anywhere, no
      #          CI workflow exists, and CMakeLists.txt line 209 leaves
      #          `# catkin_add_nosetests(test)` commented out. (Three files under
      #          to_be_deleted/ carry "test" in the name -- image_stitching_test.py,
      #          image_test.py, writing_image_test.py -- but they are scratch
      #          scripts, not tests.) A `test` verb would be a lie.
      #
      # `text` is bash under `set -euo pipefail`, shellcheck'd at BUILD time. Each
      # text below `cd "$REPO_ROOT"` first, which is what makes a tool's own "no
      # path given, use the current directory" default mean THIS repo rather than
      # the caller's directory. The consequence to know: a relative path argument
      # is then resolved against the repo root, not against where you typed it.
      # Absolute paths and flags are unaffected.
      commands = pkgs: {
        lint = {
          # Measured today with the pinned ruff 0.16.2: 137 findings across 23 rule
          # codes, led by I001 (20), UP032 (17), BLE001 (16), E722 (11), F841 (9)
          # and F401 (9), spread over all 22 tracked .py files. That non-zero exit
          # is the honest state of 2022-era lab code, not a broken flake -- do not
          # chase it by weakening the verb.
          #
          # Note for anyone reading an older copy of this comment: the enabled rule
          # set is NOT E4/E7/E9/F. `ruff check --show-settings` on this pin lists a
          # far wider default selection (I, UP, BLE, TRY, SIM, S, RUF, PIE, PL ...)
          # with preview disabled and no config file anywhere above this directory.
          description = "ruff check the whole repo (exits non-zero today: pre-existing findings)";
          text = ''
            cd "$REPO_ROOT"
            ruff check "$@"
          '';
        };
        fmt = {
          description = "ruff format the whole repo (rewrites files)";
          # Mutating, so it refuses rather than guessing when $REPO_ROOT fell back
          # to the read-only store snapshot. Without the guard `ruff format` walks
          # the whole tree first and then emits, measured against that snapshot,
          # one line per file -- "error: Failed to write <path>: Read-only file
          # system (os error 30)", 22 of them -- to convey the single fact the
          # guard states once.
          text = ''
            need_writable_checkout
            cd "$REPO_ROOT"
            ruff format "$@"
          '';
        };
        run = {
          # A bare `python3` is correct here: the store interpreter on PATH is the
          # one carrying cv2 and numpy, and there is no .venv for it to shadow.
          #
          # Read-only with respect to the tree: camera_test_no_ROS.py contains no
          # `imwrite` and no `open(`, and PYTHONDONTWRITEBYTECODE keeps CPython
          # from dropping .pyc files beside it. So no need_writable_checkout here.
          #
          # Verified to reach the repo's own code and then fail on the repo's own
          # bug: line 11 calls
          # `ImageJoinFactory.create_instance(joinType=2, ...)` with keywords, but
          # image_join.py line 40 declares `create_instance(dict)` taking one
          # positional dict, so it dies with
          #   TypeError: ImageJoinFactory.create_instance() got an unexpected
          #   keyword argument 'joinType'
          # That the traceback gets that far also proves the script's own directory
          # landed on sys.path[0], which is what makes its `import image_join`
          # resolve. Pre-existing defect in a deprecated repo, not a toolchain
          # problem -- and it fails fast rather than hanging, which is what matters
          # for an unattended agent.
          description = "start the non-ROS two-camera join demo (needs 2 USB cameras + a display; hits a pre-existing repo bug, see flake comment)";
          text = ''
            cd "$REPO_ROOT"
            python3 "$REPO_ROOT/catkin_ws/src/camera_tests/scripts/camera_test_no_ROS.py" "$@"
          '';
        };
      };

      # ======================================================================
      # PER-REPO BLOCK 6 -- checks beyond the canonical two
      # ======================================================================
      extraChecks = pkgs: {
        # The cheap real check available in a repo with no test suite: import the
        # modules the scripts actually need. It would catch a nixpkgs bump that
        # drops cv2's python bindings or moves tkinter out of the interpreter.
        # tkinter is imported without instantiating Tk. Measured with DISPLAY
        # unset, `tkinter.Tk()` raises "_tkinter.TclError: no display name and no
        # $DISPLAY environment variable" -- a build sandbox has no display, so
        # calling it here would fail for a reason unrelated to the toolchain.
        pythonImports = pkgs.runCommand "python-imports-check" { nativeBuildInputs = toolchain pkgs; } ''
          set -euo pipefail
          python3 -c 'import cv2, numpy, scipy, serial, tkinter, yaml; print(cv2.__version__)'
          touch "$out"
        '';

        # The canonical `anchoring` check proves rootPreamble and guardPreamble
        # behave. This one proves THIS repo's verbs actually use them.
        verbAnchoring =
          pkgs.runCommand "verb-anchoring-check" { nativeBuildInputs = lib.attrValues (wrappers pkgs); }
            ''
              set -euo pipefail

              # A decoy carrying what a naive anchor would accept for a python
              # repo, plus one filename this repo does not contain.
              mkdir decoy
              cd decoy
              printf 'import os\nx  =1\n' > sibling_only.py
              printf 'opencv-python\n' > requirements.txt
              printf '{\n  description = "a different repo";\n  outputs = _: { };\n}\n' > flake.nix
              cp -r . ../decoy.orig

              # Grep by NAME, not by directory: a wrongly anchored ruff is also
              # STANDING in the decoy and prints bare relative paths, so a grep
              # for "decoy" would match nothing and the leak would sail through.
              dev-lint > lint.log 2>&1 || true
              if grep -q sibling_only lint.log; then
                echo "dev-lint graded the decoy" >&2
                cat lint.log >&2
                exit 1
              fi
              # ...and it must have graded SOMETHING: a verb that read nothing at
              # all also passes the test above. This path exists only in this repo.
              if ! grep -q 'catkin_ws/src/camera_tests/scripts/image_join.py' lint.log; then
                echo "dev-lint graded neither the decoy nor this repo" >&2
                cat lint.log >&2
                exit 1
              fi
              # Exit codes are deliberately not asserted: dev-lint is non-zero only
              # because the findings are real, and would flip to zero the day
              # somebody fixes them.

              # Refusal, not silence, and not a reformat of somebody else's tree.
              if dev-fmt > fmt.log 2>&1; then
                echo "dev-fmt succeeded in a foreign tree; it must refuse" >&2
                exit 1
              fi

              # `*.log`, and every log file must match it -- a file named plainly
              # `log` is not excluded by `--exclude='*.log'` and fails this diff.
              diff -r --exclude='*.log' . ../decoy.orig
              touch "$out"
            '';
      };

      # >>>>> BEGIN CANONICAL MACHINERY v1 <<<<<
      # ======================================================================
      # Everything from the BEGIN sentinel above to the END sentinel on the last
      # line of this file is fleet-canonical text: the same bytes in every repo
      # that carries this flake style. That is a checkable claim, not a boast --
      #
      #   sed -n '/BEGIN CANONICAL MACHINERY v1/,$p' flake.nix | sha256sum
      #
      # prints the same digest in every repo, or one of them has been edited.
      # (`,$p`, not a range ending on the END sentinel: a range whose closing
      # pattern were spelled out here would terminate on this very comment.)
      # Nothing here names a repository, a language, a tool or a project file.
      # If you find such a name below, it is contamination: the fix is to move
      # it into the per-repo section above, never to special-case it here.
      #
      # This region READS exactly these names from the per-repo section:
      #   nixpkgs  self  lib  repoName  toolchain  nativeLibs  envVars
      #   commands  extraChecks
      # and DEFINES exactly these:
      #   systems  forAllSystems  ldPreamble  rootPreamble  guardPreamble
      #   wrappers  helpFor  anchorCheck
      # plus the four flake outputs apps / devShells / checks / formatter.
      # Anything else in scope is invisible to it. The types of those eight
      # inputs, and the shell variables this region exports into command texts,
      # are specified in INTERFACE.md, which travels with this block.
      #
      # To change behaviour here you change it in every repo at once and bump
      # the version in both sentinels. A local edit is a bug by construction:
      # the digest above stops matching, and -- because rootPreamble anchors on
      # flake.nix byte-identity -- an edited working tree also stops being
      # recognised by wrappers built from the previous revision.
      # ======================================================================

      # ---- systems policy: decided once for the whole fleet ----
      #
      # Read this list as "evaluated on three, built on one". That is what was
      # measured, and it is all it means:
      #   * `nix flake check --all-systems` passes, so every output attribute
      #     below EVALUATES on all three systems.
      #   * only x86_64-linux has ever been BUILT. The machine this was verified
      #     on has no aarch64 emulation -- no binfmt handler, and `extra-
      #     platforms` is x86-only -- so aarch64 cannot be built there at all.
      # It is not a statement that anything works on aarch64. Do not upgrade it
      # into one in a README.
      #
      # Evaluating all three is still worth its seconds, because the failure it
      # catches is an eval-time failure: a `pkgs.<attr>` that exists on Linux
      # and not on darwin (`stdenv.cc.cc.lib` is the usual one) throws during
      # evaluation, and `nix flake check` without --all-systems checks only the
      # current system and sails straight past it.
      #
      # x86_64-darwin is deliberately absent. nixpkgs 26.11 replaced that whole
      # attribute set with a `throw`. genAttrs is lazy, so plain `nix develop`
      # on Linux would not notice -- it detonates later, on the --all-systems
      # run this policy requires. Add it back only against a separate
      # nixpkgs-26.05-darwin input.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      # Stand-in for flake-utils.lib.eachDefaultSystem. Passes `pkgs` rather
      # than a system string, because that is what every call site wants, and
      # keeps the system list in this file rather than in a second input's
      # hardcoded copy of it.
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # Prepend, never assign: a host LD_LIBRARY_PATH may be carrying something
      # the user needs, and clobbering it breaks binaries they launch from here.
      # Linux only -- on darwin the loader variable is DYLD_*, and exporting a
      # Linux-shaped value there is at best useless.
      #
      # `&&` short-circuits in Nix, so on darwin `nativeLibs pkgs` is never
      # forced. That is load-bearing for the systems policy above: it is what
      # lets a repo list Linux-only attrs in nativeLibs and still evaluate on
      # aarch64-darwin. Do not reorder the two operands.
      ldPreamble =
        pkgs:
        lib.optionalString (pkgs.stdenv.hostPlatform.isLinux && nativeLibs pkgs != [ ]) ''
          export LD_LIBRARY_PATH="${lib.makeLibraryPath (nativeLibs pkgs)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';

      # Every command gets $SRC_ROOT and $REPO_ROOT. `nix run` and `nix develop`
      # both start in whatever directory they were invoked from, and no verb may
      # act on that directory -- these two are what it acts on instead.
      #
      # $SRC_ROOT is this flake's own source, snapshotted into the store when
      # the flake was evaluated. It is the one anchor that is always available:
      # `nix run /path/to/repo#lint` tells the running program nothing whatever
      # about /path/to/repo (flake refs are location-independent by design, and
      # there is no $FLAKE_DIR to read), so without `self` a wrapper invoked
      # that way has literally no way to name the repo it belongs to. Two
      # limitations worth knowing: it is read-only, being a store path, and in a
      # git checkout it contains only TRACKED files.
      #
      # $REPO_ROOT is the writable checkout when the caller is standing in one,
      # and $SRC_ROOT when they are not. Three things this deliberately is NOT:
      #
      #   * NOT `pwd`. A fallback to the caller's directory is how `fmt`
      #     rewrites a stranger's source tree and how `lint` prints "all checks
      #     passed" having read none of this repo.
      #   * NOT `git rev-parse --show-toplevel`. Run from inside some OTHER git
      #     repo it cheerfully answers with THAT repo's top level. It also needs
      #     git on PATH and a .git directory, so it fails on an export and in
      #     any wrapper whose toolchain omits git.
      #   * NOT an inherited $REPO_ROOT from the environment. The dev shell
      #     EXPORTS this variable, so honouring it would mean that running
      #     `nix run /path/to/B#fmt` from inside repo A's dev shell points B's
      #     formatter at A. An explicit path argument is how a caller overrides
      #     a verb's target; an ambient variable is how they do it by accident.
      #
      # Instead: walk up from $PWD and take the first ancestor that IS this
      # repo, proved by carrying a byte-identical flake.nix. A single tracked
      # filename, a marker directory, or a set of them is not proof -- sibling
      # repos in a fleet share those, and a decoy can be built to carry any list
      # of names you care to publish. The whole flake.nix is what distinguishes
      # repos, because description, toolchain and command map all differ, so the
      # whole flake.nix is what gets compared. Compared with bash's own
      # `$(<file)` rather than cmp or sha256sum, so the check depends on no
      # package at all -- pure builtins, correct even in a wrapper whose PATH
      # carries nothing but the repo's own toolchain.
      #
      # Consequence worth knowing: edit flake.nix and the dev-* wrappers in an
      # already-open `nix develop` stop recognising the tree, because they were
      # built from the previous flake.nix. That is a stale shell telling you so
      # -- re-enter it. `nix run` re-evaluates every time and never sees this.
      rootPreamble = ''
        SRC_ROOT=${lib.escapeShellArg "${self}"}
        export SRC_ROOT

        _dev_find_root() {
          local dir ref
          ref=$(<"$SRC_ROOT/flake.nix") || return 1
          dir=$(
            unset CDPATH
            cd -P -- "''${1:-.}" 2>/dev/null && pwd
          ) || return 1
          while [ -n "$dir" ]; do
            if [ -f "$dir/flake.nix" ] && [ "$(<"$dir/flake.nix")" = "$ref" ]; then
              printf '%s\n' "$dir"
              return 0
            fi
            dir=''${dir%/*}
          done
          return 1
        }

        REPO_ROOT="$(_dev_find_root "$PWD" || printf '%s\n' "$SRC_ROOT")"
        export REPO_ROOT
      '';

      # Wrappers only, not the shellHook -- an interactive shell has no business
      # carrying this function around. Any command text that writes files calls
      # it first, and it is the reason a mutating verb can fail loudly instead
      # of falling back to "well, the cwd then".
      #
      # The test is $REPO_ROOT != $SRC_ROOT, i.e. "rootPreamble found a real
      # checkout", not a permission or a store-path-prefix test. Both of those
      # answer a narrower question: a checkout may be read-only for unrelated
      # reasons, and a store path is not the only tree we must refuse to write.
      guardPreamble = ''
        need_writable_checkout() {
          if [ "$REPO_ROOT" != "$SRC_ROOT" ]; then
            return 0
          fi
          echo "''${0##*/}: this command rewrites files, so it needs a writable" >&2
          echo "checkout of this repo -- and standing in $PWD there is none: no" >&2
          echo "parent directory carries this flake's flake.nix. The only tree in" >&2
          echo "reach is the read-only store snapshot $SRC_ROOT, and rewriting" >&2
          echo "$PWD instead is exactly the bug this guard exists to prevent." >&2
          echo "cd into the repo (or \`nix develop\` it), or pass an explicit path." >&2
          exit 1
        }
      '';

      # One derivation per command, reused by both `apps` and the dev shell, so
      # the two can never diverge. `dev-` prefixed because a bare `test` binary
      # earlier on PATH would shadow the POSIX shell builtin and quietly break
      # every script in the repo that uses it.
      #
      # writeShellApplication, not writeShellScriptBin: it runs shellcheck at
      # BUILD time and sets `set -euo pipefail`, so an unquoted $@ or a silently
      # ignored failure is a `nix flake check` failure rather than a surprise in
      # front of an agent.
      wrappers =
        pkgs:
        lib.mapAttrs (
          name: cmd:
          pkgs.writeShellApplication {
            name = "dev-${name}";
            runtimeInputs = toolchain pkgs;
            runtimeEnv = envVars pkgs;
            meta.description = cmd.description;
            text = ''
              ${rootPreamble}
              ${guardPreamble}
              ${ldPreamble pkgs}
              ${cmd.text}
            '';
          }
        ) (commands pkgs);

      # `dev-help` is generated from the same attrset as everything else, so it
      # cannot describe a verb that does not exist or miss one that does. No
      # runtimeInputs: printing the map must work with nothing installed.
      helpFor =
        pkgs:
        let
          cmds = commands pkgs;
          names = lib.attrNames cmds;
          width = lib.foldl' (a: n: lib.max a (builtins.stringLength n)) 0 names;
          pad = n: n + lib.concatStrings (lib.genList (_: " ") (width - builtins.stringLength n));
          line = n: c: "  dev-${pad n}  ${c.description}";
        in
        pkgs.writeShellApplication {
          name = "dev-help";
          meta.description = "print this repo's command map (works offline)";
          text = ''
            cat <<'EOF'
            ${lib.concatStringsSep "\n" (lib.mapAttrsToList line cmds)}
            EOF
          '';
        };

      # The regression gate for rootPreamble and guardPreamble, which are the
      # two pieces of this flake that can silently damage a tree that is not
      # this repo. It tests the MECHANISM, not any verb, which is precisely what
      # makes it fleet-generic: it needs to know nothing about what this repo
      # does, only that the anchor resolves and the guard refuses.
      #
      # The decoy is a real directory carrying a real flake.nix that differs.
      # Marker-file anchors pass a decoy like this -- that is the whole point of
      # the probe -- and so does any anchor that trusts `pwd`. Probe 2 is the
      # other half, and without it a guard that refused everything would score a
      # perfect pass: a tree that IS byte-identical must still be adopted, or
      # every mutating verb in the repo is dead. Probe 3 pins the subdirectory
      # case, which is the normal one for an agent working inside a repo.
      #
      # A per-repo probe that drives the actual verbs is strictly better and
      # cannot live here -- it has to know which verb writes and which needs a
      # network. INTERFACE.md shows how to add one via `extraChecks`.
      anchorCheck =
        pkgs:
        pkgs.runCommand "anchor-check" { } ''
          set -euo pipefail

          # The two preambles under test, verbatim, in a file the probes source.
          # A quoted heredoc, so every $ below is the bash the wrappers see.
          cat > preamble.sh <<'CANONICAL_PREAMBLE_EOF'
          ${rootPreamble}
          ${guardPreamble}
          CANONICAL_PREAMBLE_EOF

          mkdir decoy
          printf '{\n  description = "a different repo";\n  outputs = _: { };\n}\n' > decoy/flake.nix
          printf 'do not touch me\n' > decoy/victim.txt
          cp -r decoy decoy.orig

          # ---- probe 1: a foreign tree must not be adopted ----
          if ! ( cd decoy && . ../preamble.sh && [ "$REPO_ROOT" = "$SRC_ROOT" ] ); then
            echo "anchor adopted a directory that is not this repo" >&2
            exit 1
          fi
          # In a subshell: need_writable_checkout ends in `exit`, which would
          # otherwise take this whole build down instead of failing a condition.
          if ( cd decoy && . ../preamble.sh && need_writable_checkout ) > guard.log 2>&1; then
            echo "need_writable_checkout accepted a tree that is not this repo" >&2
            exit 1
          fi
          if ! diff -r decoy decoy.orig; then
            echo "the probes modified the foreign tree" >&2
            exit 1
          fi

          # ---- probe 2: a byte-identical checkout must be adopted ----
          cp -r ${lib.escapeShellArg "${self}"} checkout
          chmod -R u+w checkout
          if ! ( cd checkout && . ../preamble.sh &&
                 [ "$REPO_ROOT" = "$(pwd -P)" ] && need_writable_checkout ); then
            echo "anchor refused a byte-identical checkout of this repo" >&2
            exit 1
          fi

          # ---- probe 3: from a subdirectory, still the checkout root ----
          mkdir -p checkout/probe3/deeper
          if ! ( cd checkout/probe3/deeper && . ../../../preamble.sh &&
                 [ "$REPO_ROOT" = "$(cd -P ../.. && pwd)" ] ); then
            echo "anchor did not walk up to the checkout root from a subdirectory" >&2
            exit 1
          fi

          touch "$out"
        '';
    in
    {
      # `nix flake show` -- the discovery entrypoint, and deliberately the whole
      # machine-facing contract: every app carries a meta.description, which
      # `nix flake show` prints inline and `nix flake show --json` exposes at
      # .apps.<system>.<name>.description. Pure evaluation, so an agent gets the
      # entire command map in one cheap call without reading a README.
      #
      # Do NOT invent a top-level output for this (`agentManifest`, `probeThing`
      # ...). Nix answers with `warning: unknown flake output '<name>'` on every
      # single `nix flake check`, forever.
      apps = forAllSystems (
        pkgs:
        lib.mapAttrs (name: cmd: {
          type = "app";
          program = "${(wrappers pkgs).${name}}/bin/dev-${name}";
          meta.description = cmd.description;
        }) (commands pkgs)
      );

      # `nix develop` -- the toolchain, plus a dev-<verb> for every app.
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = toolchain pkgs ++ lib.attrValues (wrappers pkgs) ++ [ (helpFor pkgs) ];

          env = envVars pkgs;

          # Natively-compiled extension modules are routinely built at -O0,
          # where glibc's _FORTIFY_SOURCE stops being a warning and becomes a
          # hard error.
          hardeningDisable = [ "fortify" ];

          shellHook = ''
            # mkShell inherits SOURCE_DATE_EPOCH=315532800 (1980-01-01) from
            # stdenv, and any wheel or zip built in here then dies with "ZIP does
            # not support timestamps before 1980".
            unset SOURCE_DATE_EPOCH

            # $REPO_ROOT and $SRC_ROOT are exported here as a convenience for
            # the human at the prompt. Every wrapper re-resolves them from
            # scratch and none of them reads these, on purpose: a stale value
            # exported by one repo's shell must never steer another repo's verb.
            ${rootPreamble}
            ${ldPreamble pkgs}

            # Nothing networked, nothing stateful and nothing interactive above
            # this line, and nothing below it either. No environment
            # bootstrapping, no dependency installation, no `read`, no
            # `exec $SHELL`. Bootstrapping in the hook makes a cold
            # `nix develop -c <anything>` start downloading before it runs
            # anything, on EVERY invocation -- the exact failure an unattended
            # agent cannot diagnose. That is what a `setup` verb is for.

            # The banner is interactive-only, and this guard is load-bearing:
            # shellHook output lands on the STDOUT of `nix develop -c <cmd>`, so
            # an unguarded echo corrupts anything parsing it
            # (`nix develop -c cat x.json | jq` fails to parse). $- is the only
            # reliable discriminator here -- it lacks `i` for `nix develop -c`
            # and has it at an interactive prompt. Do not test $PS1 (unset in
            # both) or $IN_NIX_SHELL (set in both). >&2 is the second layer, for
            # the case where a caller runs us on a pty.
            case $- in
              *i*) echo "${repoName} dev shell -- 'dev-help' for the command map" >&2 ;;
            esac
          '';
        };
      });

      # `nix flake check` -- honest by construction, and the only gate this
      # style has. `toolchain` realises the whole toolchain closure (so a typo'd
      # or currently-broken attr fails here, not halfway through a task) and
      # builds every wrapper, which runs shellcheck over every command text.
      # `anchoring` is the regression test described above.
      #
      # Repo-specific checks go in `extraChecks`, never here. They may not
      # shadow either canonical name: silently replacing `anchoring` with
      # something weaker is the exact failure this whole file exists to make
      # impossible, so a collision is an eval error with both names in it.
      #
      # NEVER add a check that always passes. An agent reads "all checks
      # passed!" as a signal, and a fake check makes `nix flake check` a liar.
      checks = forAllSystems (
        pkgs:
        let
          canonical = {
            toolchain =
              pkgs.runCommand "toolchain-check"
                {
                  nativeBuildInputs = toolchain pkgs ++ lib.attrValues (wrappers pkgs) ++ [ (helpFor pkgs) ];
                }
                ''
                  set -euo pipefail
                  dev-help > help.txt

                  # A while-read over a heredoc rather than `for x in <list>`,
                  # which is a bash syntax error when the list is empty -- and a
                  # repo with no verbs yet is a legitimate state.
                  while IFS= read -r verb; do
                    [ -n "$verb" ] || continue
                    command -v "dev-$verb" > /dev/null || {
                      echo "dev-$verb is not on PATH" >&2
                      exit 1
                    }
                    grep -q -- "dev-$verb" help.txt || {
                      echo "dev-$verb is missing from the dev-help map" >&2
                      exit 1
                    }
                  done <<'CANONICAL_VERBS_EOF'
                  ${lib.concatStringsSep "\n" (lib.attrNames (commands pkgs))}
                  CANONICAL_VERBS_EOF

                  touch "$out"
                '';
            anchoring = anchorCheck pkgs;
          };
          extra = extraChecks pkgs;
          clash = lib.intersectLists (lib.attrNames canonical) (lib.attrNames extra);
        in
        if clash != [ ] then
          throw "extraChecks must not redefine canonical checks: ${lib.concatStringsSep ", " clash}"
        else
          canonical // extra
      );

      # `nix fmt` -- formats the *Nix* in this repo; project code gets a `fmt`
      # verb. nixfmt-tree (the treefmt wrapper) rather than bare nixfmt, because
      # bare nixfmt tries to parse every path handed to it and fails on non-Nix
      # files. This file ships already formatted, so `nix fmt` is a no-op rather
      # than a diff across the fleet.
      #
      # This is the one verb here NOT anchored to $REPO_ROOT, and it cannot be:
      # `nix fmt` is nix's own verb, and nix -- not this flake -- decides which
      # paths the formatter receives, passing the cwd when the user names none.
      # A wrapper that overrode them would break `nix fmt path/to/one/file.nix`,
      # and it cannot tell that "." apart from the default. So `nix fmt` formats
      # where you stand, by design; the `fmt` verb is the anchored one.
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
# >>>>> END CANONICAL MACHINERY v1 <<<<<
